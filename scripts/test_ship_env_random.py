import os
import sys
import numpy as np

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from envs.multi_ship_env import MultiShipEnv


def run_random_episode(env, scenario):
    obs, info = env.reset(scenario=scenario)

    print("\n" + "=" * 60)
    print(f"Scenario: {scenario}")
    print("Reset info:", info)
    print("Observation dimension:", env.get_obs_dim())
    print("Action dimension:", env.get_action_dim())

    done = False
    total_rewards = np.zeros(env.num_ships, dtype=np.float32)

    while not done:
        # Random actions in [-1, 1]
        actions = np.random.uniform(-1.0, 1.0, size=(env.num_ships,))
        obs, rewards, done, info = env.step(actions)

        total_rewards += np.array(rewards, dtype=np.float32)

        print(f"Step {info['current_step']:02d} | Actions: {actions} | Rewards: {rewards}")
        print(f"Collision: {info['collision']} | Reached: {info['reached_goals']} | Timeout: {info['timeout']}")

    print("Episode finished.")
    print("Total rewards:", total_rewards)
    print("Final info:", info)

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

    # You can test multiple scenarios one by one
    run_random_episode(env, scenario="head_on")
    run_random_episode(env, scenario="crossing")
    run_random_episode(env, scenario="overtaking")

    env.close()