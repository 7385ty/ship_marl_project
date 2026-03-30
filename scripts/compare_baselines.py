import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 合并三个csv表格

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)


def load_and_tag_csv(csv_path, method_name):
    """
    Load one summary CSV file and add a 'method' column.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df["method"] = method_name
    return df


def save_comparison_csv(df, save_path):
    """
    Save merged comparison dataframe to CSV.
    """
    df.to_csv(save_path, index=False)
    print(f"Comparison CSV saved to: {save_path}")


def plot_metric_bar(df, metric, save_path, title=None):
    """
    Plot grouped bar chart for one metric across scenarios and methods.
    """
    plt.figure(figsize=(9, 6))
    sns.barplot(data=df, x="scenario", y=metric, hue="method")

    plt.title(title if title else metric)
    plt.xlabel("Scenario")
    plt.ylabel(metric)
    plt.ylim(0, 1.0 if "rate" in metric else None)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(title="Method")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Figure saved to: {save_path}")


def main():
    logs_dir = os.path.join(PROJECT_ROOT, "logs")
    figures_dir = os.path.join(PROJECT_ROOT, "figures")

    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # Input summary CSV files
    random_csv = os.path.join(logs_dir, "random_policy_summary.csv")
    rule_v1_csv = os.path.join(logs_dir, "rule_policy_summary.csv")
    rule_v2_csv = os.path.join(logs_dir, "rule_policy_v2_summary.csv")

    # Load and tag
    df_random = load_and_tag_csv(random_csv, "Random")
    df_rule_v1 = load_and_tag_csv(rule_v1_csv, "Rule_v1")
    df_rule_v2 = load_and_tag_csv(rule_v2_csv, "Rule_v2")

    # Merge
    comparison_df = pd.concat([df_random, df_rule_v1, df_rule_v2], ignore_index=True)

    # Reorder columns for readability
    preferred_columns = [
        "method",
        "scenario",
        "num_episodes",
        "success_rate",
        "collision_rate",
        "timeout_rate",
        "avg_return",
        "avg_episode_length",
    ]
    comparison_df = comparison_df[preferred_columns]

    # Sort rows
    method_order = ["Random", "Rule_v1", "Rule_v2"]
    scenario_order = ["head_on", "crossing", "overtaking"]

    comparison_df["method"] = pd.Categorical(comparison_df["method"], categories=method_order, ordered=True)
    comparison_df["scenario"] = pd.Categorical(comparison_df["scenario"], categories=scenario_order, ordered=True)

    comparison_df = comparison_df.sort_values(["scenario", "method"]).reset_index(drop=True)

    # Print to terminal
    print("\n" + "=" * 80)
    print("Baseline Comparison Table")
    print("=" * 80)
    print(comparison_df)

    # Save merged comparison CSV
    comparison_csv_path = os.path.join(logs_dir, "baseline_comparison.csv")
    save_comparison_csv(comparison_df, comparison_csv_path)

    # Set seaborn style
    sns.set(style="whitegrid", font_scale=1.1)

    # Plot metrics
    plot_metric_bar(
        comparison_df,
        metric="success_rate",
        save_path=os.path.join(figures_dir, "baseline_success_rate.png"),
        title="Baseline Comparison - Success Rate"
    )

    plot_metric_bar(
        comparison_df,
        metric="collision_rate",
        save_path=os.path.join(figures_dir, "baseline_collision_rate.png"),
        title="Baseline Comparison - Collision Rate"
    )

    plot_metric_bar(
        comparison_df,
        metric="timeout_rate",
        save_path=os.path.join(figures_dir, "baseline_timeout_rate.png"),
        title="Baseline Comparison - Timeout Rate"
    )

    print("\nAll baseline comparison results have been generated successfully!")


if __name__ == "__main__":
    main()