import os
import json
import numpy as np
import matplotlib.pyplot as plt


def load_summary_json(summary_path):
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"summary.json not found: {summary_path}")

    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def print_all_keys(summary_data):
    print("\n" + "=" * 80)
    print("Available keys in summary.json:")
    print("=" * 80)

    for i, key in enumerate(summary_data.keys()):
        print(f"{i + 1}. {key}")


def inspect_key_shapes(summary_data):
    print("\n" + "=" * 80)
    print("Key shape inspection:")
    print("=" * 80)

    for key, value in summary_data.items():
        try:
            arr = np.array(value)
            print(f"{key}: shape={arr.shape}")
        except Exception as e:
            print(f"{key}: cannot convert to numpy array ({e})")


def extract_curve(summary_data, key):
    """
    Handle summary.json formats:
    1. shape (N, 2): [step, value]
    2. shape (N, 3): [wall_time, step, value]
    """
    if key not in summary_data:
        return None, None

    arr = np.array(summary_data[key], dtype=np.float32)

    if arr.ndim != 2:
        print(f"[Warning] Key '{key}' has unexpected ndim={arr.ndim}, shape={arr.shape}")
        return None, None

    if arr.shape[1] == 2:
        x = arr[:, 0]
        y = arr[:, 1]
        return x, y

    elif arr.shape[1] == 3:
        x = arr[:, 1]   # step
        y = arr[:, 2]   # value
        return x, y

    else:
        print(f"[Warning] Key '{key}' has unexpected shape={arr.shape}")
        return None, None


def plot_single_curve(summary_data, key, save_path, title=None):
    x, y = extract_curve(summary_data, key)

    if x is None or y is None:
        print(f"[Skip] Could not plot key: {key}")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label=key)
    plt.xlabel("Training Step")
    plt.ylabel("Value")
    plt.title(title if title else key)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[Saved] {save_path}")


def plot_multi_curves(summary_data, keys, save_path, title="Training Curves"):
    plt.figure(figsize=(10, 6))

    plotted = False
    for key in keys:
        x, y = extract_curve(summary_data, key)
        if x is not None and y is not None:
            plt.plot(x, y, label=key)
            plotted = True
        else:
            print(f"[Skip] Key not plotted: {key}")

    if not plotted:
        print(f"[Skip] No valid keys found for {save_path}")
        plt.close()
        return

    plt.xlabel("Training Step")
    plt.ylabel("Value")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[Saved] {save_path}")


def main():
    summary_path = r"D:\ship_marl_project\external\on-policy\onpolicy\scripts\results\Ship\head_on\rmappo\debug_ship_rmappo\run1\logs\summary.json"
    figures_dir = r"D:\ship_marl_project\figures\rmappo_analysis"
    os.makedirs(figures_dir, exist_ok=True)

    summary_data = load_summary_json(summary_path)

    print_all_keys(summary_data)
    inspect_key_shapes(summary_data)

    important_keys = [
        r"D:\ship_marl_project\external\on-policy\onpolicy\scripts\results\Ship\head_on\rmappo\debug_ship_rmappo\run1\logs\average_episode_rewards\average_episode_rewards",
        r"D:\ship_marl_project\external\on-policy\onpolicy\scripts\results\Ship\head_on\rmappo\debug_ship_rmappo\run1\logs\eval_average_episode_rewards\eval_average_episode_rewards",
        r"D:\ship_marl_project\external\on-policy\onpolicy\scripts\results\Ship\head_on\rmappo\debug_ship_rmappo\run1\logs\policy_loss\policy_loss",
        r"D:\ship_marl_project\external\on-policy\onpolicy\scripts\results\Ship\head_on\rmappo\debug_ship_rmappo\run1\logs\value_loss\value_loss",
        r"D:\ship_marl_project\external\on-policy\onpolicy\scripts\results\Ship\head_on\rmappo\debug_ship_rmappo\run1\logs\dist_entropy\dist_entropy",
        r"D:\ship_marl_project\external\on-policy\onpolicy\scripts\results\Ship\head_on\rmappo\debug_ship_rmappo\run1\logs\ratio\ratio",
        r"D:\ship_marl_project\external\on-policy\onpolicy\scripts\results\Ship\head_on\rmappo\debug_ship_rmappo\run1\logs\actor_grad_norm\actor_grad_norm",
        r"D:\ship_marl_project\external\on-policy\onpolicy\scripts\results\Ship\head_on\rmappo\debug_ship_rmappo\run1\logs\critic_grad_norm\critic_grad_norm",
    ]

    print("\n" + "=" * 80)
    print("Plotting important single curves...")
    print("=" * 80)

    for key in important_keys:
        safe_name = key.split("\\")[-1].replace("/", "_")
        save_path = os.path.join(figures_dir, f"{safe_name}.png")
        plot_single_curve(
            summary_data,
            key=key,
            save_path=save_path,
            title=f"RMAPPO Ship - {safe_name}"
        )

    print("\n" + "=" * 80)
    print("Plotting combined curves...")
    print("=" * 80)

    plot_multi_curves(
        summary_data,
        keys=[
            r"D:\ship_marl_project\external\on-policy\onpolicy\scripts\results\Ship\head_on\rmappo\debug_ship_rmappo\run1\logs\average_episode_rewards\average_episode_rewards",
            r"D:\ship_marl_project\external\on-policy\onpolicy\scripts\results\Ship\head_on\rmappo\debug_ship_rmappo\run1\logs\eval_average_episode_rewards\eval_average_episode_rewards",
        ],
        save_path=os.path.join(figures_dir, "reward_curves_combined.png"),
        title="RMAPPO Ship - Reward Curves"
    )

    plot_multi_curves(
        summary_data,
        keys=[
            r"D:\ship_marl_project\external\on-policy\onpolicy\scripts\results\Ship\head_on\rmappo\debug_ship_rmappo\run1\logs\policy_loss\policy_loss",
            r"D:\ship_marl_project\external\on-policy\onpolicy\scripts\results\Ship\head_on\rmappo\debug_ship_rmappo\run1\logs\value_loss\value_loss",
        ],
        save_path=os.path.join(figures_dir, "loss_curves_combined.png"),
        title="RMAPPO Ship - Loss Curves"
    )

    plot_multi_curves(
        summary_data,
        keys=[
            r"D:\ship_marl_project\external\on-policy\onpolicy\scripts\results\Ship\head_on\rmappo\debug_ship_rmappo\run1\logs\dist_entropy\dist_entropy",
            r"D:\ship_marl_project\external\on-policy\onpolicy\scripts\results\Ship\head_on\rmappo\debug_ship_rmappo\run1\logs\ratio\ratio",
        ],
        save_path=os.path.join(figures_dir, "ppo_stats_combined.png"),
        title="RMAPPO Ship - PPO Statistics"
    )

    print("\nAnalysis completed.")
    print(f"Figures saved to: {figures_dir}")


if __name__ == "__main__":
    main()
    