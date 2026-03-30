import os
import sys
import numpy as np

# Add project root to Python path
# 用手工动作模式验证环境行为
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from envs.multi_ship_env import MultiShipEnv


def run_scenario(scenario_name, action_mode="straight"):
    env = MultiShipEnv(
        num_ships=2,
        world_size=20.0,
        dt=1.0,
        max_steps=50,
        ship_speed=0.8,
        max_delta_heading_deg=10.0,
        collision_radius=1.0,
        goal_radius=1.0,
    )

    obs, info = env.reset(scenario=scenario_name)
    print(f"\n===== Scenario: {scenario_name} =====")
    print("Reset info:", info)
    print("Initial observations:")
    for i, o in enumerate(obs):
        print(f"Ship {i} obs: {o}")

    done = False
    step_count = 0

    while not done:
        if action_mode == "straight":
            actions = [0.0, 0.0]

        elif action_mode == "turn_right":
            actions = [-0.5, -0.5]

        elif action_mode == "ship0_right_ship1_left":
            actions = [-0.5, 0.5]

        else:
            raise ValueError(f"Unknown action mode: {action_mode}")

        obs, rewards, done, info = env.step(actions)
        step_count += 1

        print(f"\nStep {step_count}")
        print("Actions:", actions)
        print("Rewards:", rewards)
        print("Info:", info)

        if done:
            print("\nEpisode finished.")

    env.render()


if __name__ == "__main__":
    run_scenario("head_on", action_mode="straight")
    run_scenario("head_on", action_mode="turn_right")
    run_scenario("crossing", action_mode="ship0_right_ship1_left")