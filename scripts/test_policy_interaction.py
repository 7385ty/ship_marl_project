import os
import sys
import numpy as np
import torch

# 让一个随机初始化的 MLP policy 跟环境交互，验证整条链路是通的。
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from envs.multi_ship_marl_wrapper import MultiShipMARLWrapper
from models.mlp_policy import MLPPolicy


def run_policy_rollout(env, policy, scenario="head_on", max_steps=20, device="cpu"):
    obs, share_obs, info = env.reset(scenario=scenario, seed=123)

    print("\n" + "=" * 80)
    print(f"Scenario: {scenario}")
    print("Reset info:", info)
    print("Initial obs shape:", obs.shape)
    print("Initial share_obs shape:", share_obs.shape)

    step_count = 0
    done = False

    while not done and step_count < max_steps:
        # Convert obs to tensor: shape (n_agents, obs_dim)
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)

        # Forward through policy
        with torch.no_grad():
            actions_tensor = policy(obs_tensor)  # shape (n_agents, action_dim)

        # Convert to numpy for env step
        actions = actions_tensor.cpu().numpy().astype(np.float32)

        next_obs, next_share_obs, rewards, dones, info = env.step(actions)

        print(f"\nStep {step_count + 1}")
        print("obs_tensor shape:", tuple(obs_tensor.shape))
        print("actions shape:", actions.shape)
        print("actions min/max:", actions.min(), actions.max())
        print("next_obs shape:", next_obs.shape)
        print("next_share_obs shape:", next_share_obs.shape)
        print("rewards shape:", rewards.shape)
        print("dones shape:", dones.shape)

        print("actions:", actions.squeeze())
        print("rewards:", rewards)
        print("dones:", dones)
        print("collision:", info["collision"], "| timeout:", info["timeout"], "| reached:", info["reached_goals"])

        obs = next_obs
        share_obs = next_share_obs
        done = bool(np.all(dones))
        step_count += 1

    print("\nPolicy rollout finished.")
    print("Final step count:", step_count)
    print("Environment truly done:", done)

    if done:
        print("Episode ended because the environment reached terminal state.")
    else:
        print("Episode stopped due to test max_steps limit.")

    env.render()


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

    obs_dim = env_info["obs_dim"]
    action_dim = env_info["action_dim"]

    # Create minimal MLP policy
    policy = MLPPolicy(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=64).to(device)
    policy.eval()

    print("\nPolicy network:")
    print(policy)

    run_policy_rollout(env, policy, scenario="head_on", max_steps=20, device=device)
    run_policy_rollout(env, policy, scenario="crossing", max_steps=20, device=device)
    run_policy_rollout(env, policy, scenario="overtaking", max_steps=20, device=device)

    env.close()