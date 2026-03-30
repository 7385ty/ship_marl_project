import os
import sys
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from envs.multi_ship_marl_wrapper import MultiShipMARLWrapper
from models.actor_critic import Actor, Critic


def load_checkpoint(checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint["config"]
    env_config = config["env_config"]
    env_info = config["env_info"]

    actor = Actor(
        obs_dim=env_info["obs_dim"],
        action_dim=env_info["action_dim"],
        hidden_dim=64
    ).to(device)

    critic = Critic(
        state_dim=env_info["state_dim"],
        hidden_dim=64
    ).to(device)

    actor.load_state_dict(checkpoint["actor_state_dict"])
    critic.load_state_dict(checkpoint["critic_state_dict"])

    actor.eval()
    critic.eval()

    env = MultiShipMARLWrapper(env_config=env_config)

    return checkpoint, env, actor, critic


def deterministic_rollout(env, actor, scenario="head_on", max_steps=60, device="cpu"):
    obs, share_obs, info = env.reset(scenario=scenario, seed=123)

    print("\n" + "=" * 80)
    print(f"Deterministic diagnosis rollout | Scenario: {scenario}")
    print("Reset info:", info)

    step_count = 0
    done = False
    all_actions = []

    while not done and step_count < max_steps:
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)

        with torch.no_grad():
            mean, std = actor(obs_tensor)
            actions = mean.cpu().numpy().astype(np.float32)

        all_actions.append(actions.copy())

        next_obs, next_share_obs, rewards, dones, info = env.step(actions)

        print(f"\nStep {step_count + 1}")
        print("actions:", actions.squeeze())
        print("action mean abs:", np.mean(np.abs(actions)))
        print("rewards:", rewards)
        print("dones:", dones)
        print("collision:", info["collision"], "| timeout:", info["timeout"], "| reached:", info["reached_goals"])

        obs = next_obs
        share_obs = next_share_obs
        done = bool(np.all(dones))
        step_count += 1

    print("\nDiagnosis rollout finished.")
    print("Final step count:", step_count)
    print("Final collision:", info["collision"])
    print("Final timeout:", info["timeout"])
    print("Final reached:", info["reached_goals"])

    all_actions = np.array(all_actions)
    print("Stored action trajectory shape:", all_actions.shape)
    print("Overall action mean:", np.mean(all_actions))
    print("Overall action std:", np.std(all_actions))
    print("Overall abs(action) mean:", np.mean(np.abs(all_actions)))

    env.render()


def main():
    device = "cpu"

    checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints")

    # ===== IMPORTANT: choose version here =====
    # version_prefix = "minimal_rl_v3"
    version_prefix = "minimal_rl_v4"
    # version_prefix = "minimal_rl_v2"

    checkpoint_files = [
        f"{version_prefix}_iter_0.pt",
        f"{version_prefix}_iter_50.pt",
        f"{version_prefix}_iter_100.pt",
        f"{version_prefix}_iter_200.pt",
        f"{version_prefix}_iter_300.pt",
    ]

    scenario = "head_on"

    print("\n" + "#" * 100)
    print(f"Diagnosing checkpoints with prefix: {version_prefix}")
    print("#" * 100)

    for ckpt_name in checkpoint_files:
        checkpoint_path = os.path.join(checkpoint_dir, ckpt_name)

        if not os.path.exists(checkpoint_path):
            print(f"\nCheckpoint not found, skipped: {checkpoint_path}")
            continue

        print("\n" + "#" * 100)
        print(f"Loading checkpoint: {checkpoint_path}")

        checkpoint, env, actor, critic = load_checkpoint(checkpoint_path, device=device)
        print(f"Checkpoint iteration: {checkpoint['iteration']}")

        deterministic_rollout(
            env=env,
            actor=actor,
            scenario=scenario,
            max_steps=60,
            device=device
        )

        env.close()


if __name__ == "__main__":
    main()