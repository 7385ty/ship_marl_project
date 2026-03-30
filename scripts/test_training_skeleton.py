import os
import sys
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from envs.multi_ship_marl_wrapper import MultiShipMARLWrapper
from models.actor_critic import Actor, Critic
from utils.rollout_buffer import RolloutBuffer


def run_training_skeleton(env, actor, critic, buffer, scenario="head_on", rollout_steps=10, device="cpu"):
    obs, share_obs, info = env.reset(scenario=scenario, seed=123)

    print("\n" + "=" * 80)
    print(f"Scenario: {scenario}")
    print("Reset info:", info)

    for step in range(rollout_steps):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)           # (n_agents, obs_dim)
        share_obs_tensor = torch.tensor(share_obs, dtype=torch.float32, device=device)  # (n_agents, state_dim)

        with torch.no_grad():
            actions_tensor, log_probs_tensor = actor.sample_action(obs_tensor)       # (n_agents, action_dim), (n_agents, 1)
            values_tensor = critic(share_obs_tensor)                                 # (n_agents, 1)

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
            values=values
        )

        print(f"\nStep {step + 1}")
        print("obs shape:", obs.shape)
        print("share_obs shape:", share_obs.shape)
        print("actions shape:", actions.shape)
        print("log_probs shape:", log_probs.shape)
        print("values shape:", values.shape)
        print("rewards shape:", rewards.shape)
        print("dones shape:", dones.shape)

        print("actions:", actions.squeeze())
        print("log_probs:", log_probs.squeeze())
        print("values:", values.squeeze())
        print("rewards:", rewards)
        print("dones:", dones)
        print("collision:", info["collision"], "| timeout:", info["timeout"], "| reached:", info["reached_goals"])

        obs = next_obs
        share_obs = next_share_obs

        if np.all(dones):
            print("\nEpisode ended early because environment reached terminal state.")
            break

    print("\nRollout collection finished.")
    print("Buffer length:", len(buffer))

    data = buffer.as_dict()

    print("\nStored buffer shapes:")
    for k, v in data.items():
        print(f"{k}: {v.shape}")

    return data


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

    actor.eval()
    critic.eval()

    print("\nActor network:")
    print(actor)

    print("\nCritic network:")
    print(critic)

    buffer = RolloutBuffer()

    data = run_training_skeleton(
        env=env,
        actor=actor,
        critic=critic,
        buffer=buffer,
        scenario="head_on",
        rollout_steps=10,
        device=device
    )

    env.close()