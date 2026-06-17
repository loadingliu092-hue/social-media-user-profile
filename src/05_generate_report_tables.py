from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils import CLUSTERED_DATA_PATH, CLUSTER_SUMMARY_PATH, FEATURE_COLUMNS, PROFILE_PATH, ensure_directories

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(
    style="whitegrid",
    font="Microsoft YaHei",
    rc={"axes.unicode_minus": False},
)


def infer_profile_notes(row: pd.Series) -> str:
    if row["cluster_name"] == "Night Owl":
        return "\u591c\u95f4\u6d3b\u8dc3\u5ea6\u9ad8\u3001\u5355\u6b21\u505c\u7559\u65f6\u95f4\u957f\u3001\u89c6\u9891\u89c2\u770b\u65f6\u957f\u660e\u663e\u504f\u9ad8\u3002"
    if row["cluster_name"] == "Interactive User":
        return "\u4e92\u52a8\u53c2\u4e0e\u5ea6\u8f83\u9ad8\uff0c\u4f46\u5185\u5bb9\u751f\u4ea7\u548c\u7c89\u4e1d\u89c4\u6a21\u4f4e\u4e8e\u5185\u5bb9\u521b\u4f5c\u8005\u578b\uff0c\u66f4\u9002\u5408\u5f52\u4e3a\u4e2d\u5ea6\u4e92\u52a8\u578b\u7528\u6237\u3002"
    if row["cluster_name"] == "Content Creator":
        return "\u53d1\u5e16\u548c\u5206\u4eab\u66f4\u79ef\u6781\u3001\u7c89\u4e1d\u89c4\u6a21\u66f4\u5927\u3001\u5185\u5bb9\u4f20\u64ad\u80fd\u529b\u7a81\u51fa\u3002"
    return "\u6574\u4f53\u6d3b\u8dc3\u9891\u7387\u8f83\u4f4e\u3001\u4f7f\u7528\u65f6\u957f\u8f83\u77ed\u3001\u4ee5\u8f7b\u5ea6\u6d4f\u89c8\u4e3a\u4e3b\u3002"


def save_profile_comparison_figure(summary: pd.DataFrame) -> None:
    comparison_columns = [
        "posts_per_week",
        "likes_per_week",
        "followers",
        "avg_session_minutes",
        "night_activity_ratio",
        "video_watch_minutes",
    ]
    plot_df = summary[["cluster_name"] + comparison_columns].melt(
        id_vars="cluster_name",
        var_name="feature",
        value_name="value",
    )

    plt.figure(figsize=(12, 7))
    sns.barplot(data=plot_df, x="feature", y="value", hue="cluster_name")
    plt.xticks(rotation=25, ha="right")
    plt.title("Cluster Profile Comparison")
    plt.tight_layout()
    plt.savefig(PROFILE_PATH, dpi=220, bbox_inches="tight")
    plt.close()


def main() -> None:
    ensure_directories()
    clustered_df = pd.read_csv(CLUSTERED_DATA_PATH)

    cluster_summary = (
        clustered_df.groupby(["cluster", "cluster_name"])[FEATURE_COLUMNS]
        .mean()
        .round(2)
        .reset_index()
        .sort_values("cluster")
        .reset_index(drop=True)
    )
    cluster_summary["user_count"] = clustered_df.groupby("cluster").size().values
    cluster_summary["profile_note"] = cluster_summary.apply(infer_profile_notes, axis=1)
    cluster_summary.to_csv(CLUSTER_SUMMARY_PATH, index=False, encoding="utf-8-sig")

    save_profile_comparison_figure(cluster_summary)
    print(f"Cluster summary saved to: {CLUSTER_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
