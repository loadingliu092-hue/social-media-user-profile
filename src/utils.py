from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
REAL_WEIBO_DATA_DIR = DATA_DIR / "real_weibo"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODEL_DIR = OUTPUTS_DIR / "model"
REAL_WEIBO_OUTPUTS_DIR = OUTPUTS_DIR / "real_weibo"
REAL_WEIBO_FIGURES_DIR = REAL_WEIBO_OUTPUTS_DIR / "figures"

RAW_DATA_PATH = DATA_DIR / "raw_social_media_users.csv"
CLEANED_DATA_PATH = DATA_DIR / "cleaned_social_media_users.csv"
CLUSTERED_DATA_PATH = OUTPUTS_DIR / "clustered_social_media_users.csv"
CLUSTER_SUMMARY_PATH = OUTPUTS_DIR / "cluster_summary.csv"
KMEANS_MODEL_PATH = MODEL_DIR / "kmeans_model.pkl"
DBSCAN_SUMMARY_PATH = OUTPUTS_DIR / "dbscan_cluster_summary.csv"
EDA_SUMMARY_PATH = OUTPUTS_DIR / "eda_summary.csv"
K_SELECTION_PATH = FIGURES_DIR / "k_selection.png"
K_SELECTION_SCORES_PATH = OUTPUTS_DIR / "k_selection_scores.csv"
DISTRIBUTION_PATH = FIGURES_DIR / "distribution.png"
CORRELATION_PATH = FIGURES_DIR / "correlation_heatmap.png"
BEHAVIOR_SCATTER_PATH = FIGURES_DIR / "engagement_scatter.png"
PCA_CLUSTER_PATH = FIGURES_DIR / "pca_clusters.png"
PROFILE_PATH = FIGURES_DIR / "cluster_profile.png"
TRAINING_LOG_PATH = OUTPUTS_DIR / "training_log.txt"

REAL_WEIBO_PROFILE_PATH = REAL_WEIBO_OUTPUTS_DIR / "real_user_profile.csv"
REAL_WEIBO_NETWORK_PATH = REAL_WEIBO_OUTPUTS_DIR / "real_network_features.csv"
REAL_WEIBO_SUMMARY_PATH = REAL_WEIBO_OUTPUTS_DIR / "real_weibo_summary.csv"
REAL_WEIBO_REPORT_PATH = REAL_WEIBO_OUTPUTS_DIR / "real_weibo_report.txt"
REAL_WEIBO_GENDER_FIGURE_PATH = REAL_WEIBO_FIGURES_DIR / "gender_distribution.png"
REAL_WEIBO_AGE_FIGURE_PATH = REAL_WEIBO_FIGURES_DIR / "age_distribution.png"
REAL_WEIBO_PROVINCE_FIGURE_PATH = REAL_WEIBO_FIGURES_DIR / "province_top10.png"
REAL_WEIBO_NETWORK_FIGURE_PATH = REAL_WEIBO_FIGURES_DIR / "network_degree_distribution.png"

FEATURE_COLUMNS = [
    "posts_per_week",
    "likes_per_week",
    "comments_per_week",
    "shares_per_week",
    "followers",
    "following",
    "avg_session_minutes",
    "active_days_per_week",
    "night_activity_ratio",
    "video_watch_minutes",
    "purchase_clicks_per_week",
]


def ensure_directories() -> None:
    for path in (
        DATA_DIR,
        REAL_WEIBO_DATA_DIR,
        OUTPUTS_DIR,
        FIGURES_DIR,
        MODEL_DIR,
        REAL_WEIBO_OUTPUTS_DIR,
        REAL_WEIBO_FIGURES_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
