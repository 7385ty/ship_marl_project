import os
import sys
import copy
import numpy as np
import torch
import torch.optim as optim

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from envs.multi_ship_marl_wrapper import MultiShipMARLWrapper
from models.actor_critic import Actor, Critic
from utils.rollout_buffer import RolloutBuffer
from utils.ppo_utils import (
    compute_returns,
    compute_advantages,
    normalize_advantages,
    flatten_marl_batch,
)


def collect_rollout(env, actor, critic, buffer, scenario="head_on", rollout_steps=10, device="cpu"):
    obs, share_obs, info = env.reset(scenario=scenario, seed=123)

    for _ in range(rollout_steps):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
        share_obs_tensor = torch.tensor(share_obs, dtype=torch.float32, device=device)

        with torch.no_grad():
            actions_tensor, log_probs_tensor = actor.sample_action(obs_tensor)
            values_tensor = critic(share_obs_tensor)

        actions = actions_tensor.cpu().numpy().astype(np.float32)
        log_probs = log_probs_tensor.cpu().numpy().astype(np.float32)
        values = values_tensor.cpu().numpy().astype(np.float32)

        next_obs, next_share_obs, rewards, dones, info = env.step(actions)

        buffer.add(
            obs=obs,
            share_obs=share_obs,
            actions=actions,
            rewards=rewards,
            dones=dones,
            log_probs=log_probs,
            values=values,
        )

        obs = next_obs
        share_obs = next_share_obs

        if np.all(dones):
            break


def test_ppo_update(env, actor, critic, actor_optimizer, critic_optimizer, scenario="head_on", rollout_steps=10, device="cpu"):
    buffer = RolloutBuffer()

    # 1. Collect rollout
    collect_rollout(env, actor, critic, buffer, scenario=scenario, rollout_steps=rollout_steps, device=device)

    data = buffer.as_dict()

    print("\n" + "=" * 80)
    print("Collected rollout data shapes:")
    for k, v in data.items():
        print(f"{k}: {v.shape}")

    # 2. Compute returns and advantages
    returns = compute_returns(data["rewards"], data["dones"], gamma=0.99)
    advantages = compute_advantages(returns, data["values"])
    advantages = normalize_advantages(advantages)

    print("\nComputed:")
    print("returns shape:", returns.shape)
    print("advantages shape:", advantages.shape)

    # 3. Flatten batch
    batch = flatten_marl_batch(
        obs=data["obs"],
        share_obs=data["share_obs"],
        actions=data["actions"],
        log_probs=data["log_probs"],
        values=data["values"],
        returns=returns,
        advantages=advantages,
    )

    print("\nFlattened batch shapes:")
    for k, v in batch.items():
        print(f"{k}: {v.shape}")

    # Convert to tensors
    obs_tensor = torch.tensor(batch["obs"], dtype=torch.float32, device=device)
    share_obs_tensor = torch.tensor(batch["share_obs"], dtype=torch.float32, device=device)
    actions_tensor = torch.tensor(batch["actions"], dtype=torch.float32, device=device)
    old_log_probs_tensor = torch.tensor(batch["old_log_probs"], dtype=torch.float32, device=device)
    returns_tensor = torch.tensor(batch["returns"], dtype=torch.float32, device=device)
    advantages_tensor = torch.tensor(batch["advantages"], dtype=torch.float32, device=device)

    # Save parameters before update
    actor_params_before = copy.deepcopy([p.detach().cpu().clone() for p in actor.parameters()])
    critic_params_before = copy.deepcopy([p.detach().cpu().clone() for p in critic.parameters()])

    # 4. Forward again for update
    new_log_probs, entropy = actor.evaluate_actions(obs_tensor, actions_tensor)
    new_values = critic(share_obs_tensor)

    # 5. Minimal PPO-style losses
    ratio = torch.exp(new_log_probs - old_log_probs_tensor)

    # No clipping first? We include minimal PPO clipping here
    clip_eps = 0.2
    surr1 = ratio * advantages_tensor
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages_tensor
    actor_loss = -torch.min(surr1, surr2).mean()

    critic_loss = ((returns_tensor - new_values) ** 2).mean()

    entropy_loss = entropy.mean()

    total_actor_loss = actor_loss - 0.01 * entropy_loss
    total_critic_loss = critic_loss

    print("\nLoss values before update:")
    print("actor_loss:", actor_loss.item())
    print("critic_loss:", critic_loss.item())
    print("entropy:", entropy_loss.item())
    print("total_actor_loss:", total_actor_loss.item())
    print("total_critic_loss:", total_critic_loss.item())

    # 6. Backward and optimize actor
    actor_optimizer.zero_grad()
    total_actor_loss.backward()
    actor_grad_norm = 0.0
    for p in actor.parameters():
        if p.grad is not None:
            actor_grad_norm += p.grad.data.norm(2).item()
    actor_optimizer.step()

    # 7. Backward and optimize critic
    critic_optimizer.zero_grad()
    total_critic_loss.backward()
    critic_grad_norm = 0.0
    for p in critic.parameters():
        if p.grad is not None:
            critic_grad_norm += p.grad.data.norm(2).item()
    critic_optimizer.step()

    print("\nGradient norms:")
    print("actor_grad_norm:", actor_grad_norm)
    print("critic_grad_norm:", critic_grad_norm)

    # 8. Check parameter update
    actor_params_after = [p.detach().cpu().clone() for p in actor.parameters()]
    critic_params_after = [p.detach().cpu().clone() for p in critic.parameters()]

    actor_param_changed = any(
        not torch.allclose(before, after)
        for before, after in zip(actor_params_before, actor_params_after)
    )

    critic_param_changed = any(
        not torch.allclose(before, after)
        for before, after in zip(critic_params_before, critic_params_after)
    )

    print("\nParameter update check:")
    print("Actor parameters changed:", actor_param_changed)
    print("Critic parameters changed:", critic_param_changed)


if __name__ == "__main__":
    device = "cpu"

    env_config = {
        "num_ships": 2,
        "world_size": 20.0,
        "dt": 1.0,
        "max_steps": 50,
        "ship_speed": 0.8,
        "max_delta_heading_deg": 10.0,
        "collision_radius": 1.0,
        "goal_radius": 1.0,
        "position_noise": 0.5,
        "heading_noise_deg": 5.0,
        "seed": 42,
    }

    env = MultiShipMARLWrapper(env_config=env_config)
    env_info = env.get_env_info()

    print("Environment info:", env_info)

    actor = Actor(
        obs_dim=env_info["obs_dim"],
        action_dim=env_info["action_dim"],
        hidden_dim=64
    ).to(device)

    critic = Critic(
        state_dim=env_info["state_dim"],
        hidden_dim=64
    ).to(device)

    actor_optimizer = optim.Adam(actor.parameters(), lr=1e-3)
    critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)

    test_ppo_update(
        env=env,
        actor=actor,
        critic=critic,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        scenario="head_on",
        rollout_steps=10,
        device=device
    )

    env.close()