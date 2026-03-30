import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from envs.multi_ship_env import MultiShipEnv
from utils.rule_policy_v2 import simple_rule_policy_v2


def run_one_episode(env, scenario, episode_id):
    """
    Run one episode with improved rule-based policy v2.

    Returns
    -------
    result : dict
    """
    obs, info = env.reset(scenario=scenario)

    terminated = False
    truncated = False

    episode_return = 0.0
    episode_length = 0
    final_info = None

    while not (terminated or truncated):
        actions = simple_rule_policy_v2(obs)
        obs, rewards, terminated, truncated, info = env.step(actions)

        episode_return += float(np.mean(rewards))
        episode_length += 1
        final_info = info

    collision = final_info["collision"]
    timeout = final_info["timeout"]
    success = final_info["all_reached"]

    result = {
        "scenario": scenario,
        "episode_id": episode_id,
        "success": bool(success),
        "collision": bool(collision),
        "timeout": bool(timeout),
        "episode_return": float(episode_return),
        "episode_length": int(episode_length),
    }
    return result


def evaluate_rule_policy_v2(env, scenario, num_episodes=20):
    """
    Evaluate improved rule-based policy v2 on one scenario.
    """
    episode_results = []

    for ep in range(num_episodes):
        result = run_one_episode(env, scenario, episode_id=ep)
        episode_results.append(result)

    success_rate = np.mean([r["success"] for r in episode_results])
    collision_rate = np.mean([r["collision"] for r in episode_results])
    timeout_rate = np.mean([r["timeout"] for r in episode_results])
    avg_return = np.mean([r["episode_return"] for r in episode_results])
    avg_episode_length = np.mean([r["episode_length"] for r in episode_results])

    summary = {
        "scenario": scenario,
        "num_episodes": num_episodes,
        "success_rate": float(success_rate),
        "collision_rate": float(collision_rate),
        "timeout_rate": float(timeout_rate),
        "avg_return": float(avg_return),
        "avg_episode_length": float(avg_episode_length),
    }

    return episode_results, summary


def print_summary(summary):
    print("\n" + "=" * 70)
    print(f"Scenario: {summary['scenario']}")
    print(f"Number of episodes: {summary['num_episodes']}")
    print(f"Success rate:      {summary['success_rate']:.3f}")
    print(f"Collision rate:    {summary['collision_rate']:.3f}")
    print(f"Timeout rate:      {summary['timeout_rate']:.3f}")
    print(f"Average return:    {summary['avg_return']:.3f}")
    print(f"Average length:    {summary['avg_episode_length']:.3f}")


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

    scenarios = ["head_on", "crossing", "overtaking"]

    all_episode_results = []
    all_summary_results = []

    for scenario in scenarios:
        episode_results, summary = evaluate_rule_policy_v2(env, scenario, num_episodes=20)

        all_episode_results.extend(episode_results)
        all_summary_results.append(summary)

        print_summary(summary)

    logs_dir = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    episode_df = pd.DataFrame(all_episode_results)
    episode_csv_path = os.path.join(logs_dir, "rule_policy_v2_episode_results.csv")
    episode_df.to_csv(episode_csv_path, index=False)

    summary_df = pd.DataFrame(all_summary_results)
    summary_csv_path = os.path.join(logs_dir, "rule_policy_v2_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)

    print("\n" + "=" * 70)
    print("Rule-policy v2 CSV files saved successfully!")
    print("Episode results:", episode_csv_path)
    print("Summary results:", summary_csv_path)

    env.close()