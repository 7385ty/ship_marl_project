import os
import sys
import numpy as np


# 先不训练，只验证一遍“MARL 数据流是否畅通
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from envs.multi_ship_marl_wrapper import MultiShipMARLWrapper


def run_random_marl_rollout(env, scenario="head_on", max_steps=10):
    obs, share_obs, info = env.reset(scenario=scenario, seed=123)

    print("\n" + "=" * 80)
    print(f"Scenario: {scenario}")
    print("Reset info:", info)

    print("\nInitial shapes:")
    print("obs shape:", obs.shape)
    print("share_obs shape:", share_obs.shape)

    step_count = 0
    done = False

    while not done and step_count < max_steps:
        # Random continuous actions for each agent
        actions = np.random.uniform(
            low=-1.0,
            high=1.0,
            size=(env.get_num_agents(), env.get_action_dim())
        ).astype(np.float32)

        next_obs, next_share_obs, rewards, dones, info = env.step(actions)

        print(f"\nStep {step_count + 1}")
        print("actions shape:", actions.shape)
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

    print("\nRollout finished.")
    print("Final step count:", step_count)
    print("Final dones:", dones)
    print("Final collision:", info["collision"])
    print("Final timeout:", info["timeout"])
    print("Final reached:", info["reached_goals"])
    if bool(np.all(dones)):
        print("Episode ended because the environment reached a terminal state.")
    else:
        print("Episode did NOT end naturally; rollout stopped due to test max_steps limit.")

    env.render()


if __name__ == "__main__":
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

    print("Environment info:", env.get_env_info())

    run_random_marl_rollout(env, scenario="head_on", max_steps=15)
    run_random_marl_rollout(env, scenario="crossing", max_steps=15)
    run_random_marl_rollout(env, scenario="overtaking", max_steps=15)

    env.close()