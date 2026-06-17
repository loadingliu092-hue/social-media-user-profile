from __future__ import annotations

import math

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils import (
    BEHAVIOR_SCATTER_PATH,
    CLEANED_DATA_PATH,
    CORRELATION_PATH,
    DISTRIBUTION_PATH,
    EDA_SUMMARY_PATH,
    FEATURE_COLUMNS,
    ensure_directories,
)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(
    style="whitegrid",
    font="Microsoft YaHei",
    rc={"axes.unicode_minus": False},
)


def save_distribution_figure(df: pd.DataFrame) -> None:
    rows = math.ceil(len(FEATURE_COLUMNS) / 3)
    fig, axes = plt.subplots(rows, 3, figsize=(18, 4.5 * rows))
    axes = axes.flatten()

    for ax, column in zip(axes, FEATURE_COLUMNS):
        sns.histplot(df[column], bins=24, kde=True, ax=ax, color="#2a9d8f")
        ax.set_title(f"{column} Distribution")
        ax.set_xlabel(column)

    for ax in axes[len(FEATURE_COLUMNS):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(DISTRIBUTION_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_correlation_heatmap(df: pd.DataFrame) -> None:
    corr_matrix = df[FEATURE_COLUMNS].corr(numeric_only=True)
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="YlGnBu", square=True)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(CORRELATION_PATH, dpi=220, bbox_inches="tight")
    plt.close()


def save_engagement_scatter(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=df,
        x="likes_per_week",
        y="comments_per_week",
        size="followers",
        hue="active_days_per_week",
        palette="viridis",
        sizes=(20, 220),
        alpha=0.72,
    )
    plt.title("User Engagement Scatter Plot")
    plt.xlabel("Likes Per Week")
    plt.ylabel("Comments Per Week")
    plt.tight_layout()
    plt.savefig(BEHAVIOR_SCATTER_PATH, dpi=220, bbox_inches="tight")
    plt.close()


def main() -> None:
    ensure_directories()
    df = pd.read_csv(CLEANED_DATA_PATH)

    eda_summary = df[FEATURE_COLUMNS].describe().T
    eda_summary["skew"] = df[FEATURE_COLUMNS].skew(numeric_only=True)
    eda_summary["missing_after_cleaning"] = df[FEATURE_COLUMNS].isna().sum()
    eda_summary.to_csv(EDA_SUMMARY_PATH, encoding="utf-8-sig")

    save_distribution_figure(df)
    save_correlation_heatmap(df)
    save_engagement_scatter(df)

    print("EDA summary and figures generated successfully.")


if __name__ == "__main__":
    main()
