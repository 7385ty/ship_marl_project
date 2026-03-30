import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


def collect_rollout(env, actor, critic, scenario="head_on", rollout_steps=20, device="cpu"):
    buffer = RolloutBuffer()

    obs, share_obs, info = env.reset(scenario=scenario)

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

    return buffer


def ppo_update(actor, critic, actor_optimizer, critic_optimizer, buffer, device="cpu", gamma=0.99, clip_eps=0.2):
    data = buffer.as_dict()

    returns = compute_returns(data["rewards"], data["dones"], gamma=gamma)
    advantages = compute_advantages(returns, data["values"])
    advantages = normalize_advantages(advantages)

    batch = flatten_marl_batch(
        obs=data["obs"],
        share_obs=data["share_obs"],
        actions=data["actions"],
        log_probs=data["log_probs"],
        values=data["values"],
        returns=returns,
        advantages=advantages,
    )

    obs_tensor = torch.tensor(batch["obs"], dtype=torch.float32, device=device)
    share_obs_tensor = torch.tensor(batch["share_obs"], dtype=torch.float32, device=device)
    actions_tensor = torch.tensor(batch["actions"], dtype=torch.float32, device=device)
    old_log_probs_tensor = torch.tensor(batch["old_log_probs"], dtype=torch.float32, device=device)
    returns_tensor = torch.tensor(batch["returns"], dtype=torch.float32, device=device)
    advantages_tensor = torch.tensor(batch["advantages"], dtype=torch.float32, device=device)

    new_log_probs, entropy = actor.evaluate_actions(obs_tensor, actions_tensor)
    new_values = critic(share_obs_tensor)

    ratio = torch.exp(new_log_probs - old_log_probs_tensor)

    surr1 = ratio * advantages_tensor
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages_tensor
    actor_loss = -torch.min(surr1, surr2).mean()

    critic_loss = ((returns_tensor - new_values) ** 2).mean()
    entropy_mean = entropy.mean()

    total_actor_loss = actor_loss - 0.01 * entropy_mean
    total_critic_loss = critic_loss

    actor_optimizer.zero_grad()
    total_actor_loss.backward()
    actor_optimizer.step()

    critic_optimizer.zero_grad()
    total_critic_loss.backward()
    critic_optimizer.step()

    update_info = {
        "actor_loss": float(actor_loss.item()),
        "critic_loss": float(critic_loss.item()),
        "entropy": float(entropy_mean.item()),
        "mean_return": float(np.mean(returns)),
        "mean_advantage": float(np.mean(advantages)),
        "buffer_steps": len(buffer),
    }

    return update_info


def evaluate_policy(env, actor, scenario="head_on", num_episodes=10, max_steps=50, device="cpu"):
    """
    Evaluate current actor deterministically using action mean.
    """
    episode_results = []

    actor.eval()

    for _ in range(num_episodes):
        obs, share_obs, info = env.reset(scenario=scenario)

        done = False
        episode_return = 0.0
        step_count = 0
        final_info = None

        while not done and step_count < max_steps:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)

            with torch.no_grad():
                mean, std = actor(obs_tensor)
                actions = mean.cpu().numpy().astype(np.float32)

            next_obs, next_share_obs, rewards, dones, info = env.step(actions)

            episode_return += float(np.mean(rewards))
            step_count += 1
            final_info = info

            obs = next_obs
            share_obs = next_share_obs
            done = bool(np.all(dones))

        episode_results.append({
            "success": bool(final_info["all_reached"]),
            "collision": bool(final_info["collision"]),
            "timeout": bool(final_info["timeout"]),
            "episode_return": float(episode_return),
            "episode_length": int(step_count),
        })

    summary = {
        "success_rate": float(np.mean([r["success"] for r in episode_results])),
        "collision_rate": float(np.mean([r["collision"] for r in episode_results])),
        "timeout_rate": float(np.mean([r["timeout"] for r in episode_results])),
        "avg_return": float(np.mean([r["episode_return"] for r in episode_results])),
        "avg_episode_length": float(np.mean([r["episode_length"] for r in episode_results])),
    }

    return summary


def plot_training_curves(log_df, figures_dir):
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(log_df["iteration"], log_df["actor_loss"], label="Actor Loss")
    plt.plot(log_df["iteration"], log_df["critic_loss"], label="Critic Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss Curves")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "minimal_rl_loss_curves.png"), dpi=300)
    plt.close()

    # 2. Evaluation curves
    plt.figure(figsize=(10, 6))
    plt.plot(log_df["iteration"], log_df["eval_success_rate"], label="Success Rate")
    plt.plot(log_df["iteration"], log_df["eval_collision_rate"], label="Collision Rate")
    plt.plot(log_df["iteration"], log_df["eval_timeout_rate"], label="Timeout Rate")
    plt.xlabel("Iteration")
    plt.ylabel("Rate")
    plt.title("Evaluation Metrics Curves")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "minimal_rl_eval_curves.png"), dpi=300)
    plt.close()

    # 3. Return curve
    plt.figure(figsize=(10, 6))
    plt.plot(log_df["iteration"], log_df["eval_avg_return"], label="Eval Avg Return")
    plt.xlabel("Iteration")
    plt.ylabel("Return")
    plt.title("Evaluation Average Return")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "minimal_rl_return_curve.png"), dpi=300)
    plt.close()


def main():
    device = "cpu"

    logs_dir = os.path.join(PROJECT_ROOT, "logs")
    figures_dir = os.path.join(PROJECT_ROOT, "figures")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

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

    train_env = MultiShipMARLWrapper(env_config=env_config)
    eval_env = MultiShipMARLWrapper(env_config=env_config)

    env_info = train_env.get_env_info()
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

    num_iterations = 50
    rollout_steps = 20
    eval_interval = 5
    train_scenario = "head_on"
    eval_scenario = "head_on"

    train_logs = []

    for iteration in range(1, num_iterations + 1):
        actor.train()
        critic.train()

        buffer = collect_rollout(
            env=train_env,
            actor=actor,
            critic=critic,
            scenario=train_scenario,
            rollout_steps=rollout_steps,
            device=device
        )

        update_info = ppo_update(
            actor=actor,
            critic=critic,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            buffer=buffer,
            device=device,
            gamma=0.99,
            clip_eps=0.2
        )

        # Periodic evaluation
        if iteration % eval_interval == 0:
            eval_summary = evaluate_policy(
                env=eval_env,
                actor=actor,
                scenario=eval_scenario,
                num_episodes=10,
                max_steps=50,
                device=device
            )
        else:
            eval_summary = {
                "success_rate": np.nan,
                "collision_rate": np.nan,
                "timeout_rate": np.nan,
                "avg_return": np.nan,
                "avg_episode_length": np.nan,
            }

        log_row = {
            "iteration": iteration,
            "actor_loss": update_info["actor_loss"],
            "critic_loss": update_info["critic_loss"],
            "entropy": update_info["entropy"],
            "mean_return": update_info["mean_return"],
            "mean_advantage": update_info["mean_advantage"],
            "buffer_steps": update_info["buffer_steps"],
            "eval_success_rate": eval_summary["success_rate"],
            "eval_collision_rate": eval_summary["collision_rate"],
            "eval_timeout_rate": eval_summary["timeout_rate"],
            "eval_avg_return": eval_summary["avg_return"],
            "eval_avg_episode_length": eval_summary["avg_episode_length"],
        }
        train_logs.append(log_row)

        print("\n" + "=" * 80)
        print(f"Iteration {iteration}/{num_iterations}")
        print(f"Actor loss: {update_info['actor_loss']:.6f}")
        print(f"Critic loss: {update_info['critic_loss']:.6f}")
        print(f"Entropy: {update_info['entropy']:.6f}")
        print(f"Mean return: {update_info['mean_return']:.6f}")
        print(f"Buffer steps: {update_info['buffer_steps']}")

        if iteration % eval_interval == 0:
            print("Evaluation:")
            print(f"  Success rate:   {eval_summary['success_rate']:.3f}")
            print(f"  Collision rate: {eval_summary['collision_rate']:.3f}")
            print(f"  Timeout rate:   {eval_summary['timeout_rate']:.3f}")
            print(f"  Avg return:     {eval_summary['avg_return']:.3f}")
            print(f"  Avg ep length:  {eval_summary['avg_episode_length']:.3f}")

    # Save training log
    log_df = pd.DataFrame(train_logs)
    log_csv_path = os.path.join(logs_dir, "minimal_rl_train_log.csv")
    log_df.to_csv(log_csv_path, index=False)
    print("\nTraining log saved to:", log_csv_path)

    # Plot curves
    plot_training_curves(log_df, figures_dir)
    print("Training curves saved to figures directory.")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()