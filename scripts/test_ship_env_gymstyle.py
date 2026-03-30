import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from envs.multi_ship_env import MultiShipEnv


def test_env_once(env, scenario):
    obs, info = env.reset(scenario=scenario, seed=123)

    print("\n" + "=" * 70)
    print(f"Scenario: {scenario}")
    print("Reset info:", info)
    print("Num agents:", env.get_num_agents())
    print("Obs dim:", env.get_obs_dim())
    print("Action dim:", env.get_action_dim())
    print("State dim:", env.get_state_dim())
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    print("Global state shape:", env.get_state().shape)

    terminated = False
    truncated = False
    total_rewards = np.zeros(env.get_num_agents(), dtype=np.float32)

    while not (terminated or truncated):
        # sample one action for each ship
        actions = np.array([env.action_space.sample() for _ in range(env.get_num_agents())])
        obs, rewards, terminated, truncated, info = env.step(actions)

        total_rewards += np.array(rewards, dtype=np.float32)

        print(f"Step {info['current_step']:02d}")
        print("Actions:", actions.squeeze())
        print("Rewards:", rewards)
        print("Terminated:", terminated, "| Truncated:", truncated)
        print("Collision:", info["collision"], "| Reached:", info["reached_goals"], "| Timeout:", info["timeout"])

    print("Episode finished.")
    print("Total rewards:", total_rewards)
    print("Final global state:", env.get_state())
    env.render()


if __name__ == "__main__":
    env = MultiShipEnv(
        num_ships=2,
        world_size=20.0,
        dt=1.0,
        max_steps=50,
        ship_speed=0.8,
        max_delta_heading_deg=10.0,
        collision_radius=1.0,
        goal_radius=1.0,
        position_noise=0.5,
        heading_noise_deg=5.0,
        seed=42,
    )

    test_env_once(env, "head_on")
    test_env_once(env, "crossing")
    test_env_once(env, "overtaking")

    env.close()