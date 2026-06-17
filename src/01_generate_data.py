from __future__ import annotations

import numpy as np
import pandas as pd

from utils import FEATURE_COLUMNS, RAW_DATA_PATH, ensure_directories


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-values))


def min_max_scale(values: np.ndarray) -> np.ndarray:
    value_min = values.min()
    value_max = values.max()
    if value_max == value_min:
        return np.zeros_like(values)
    return (values - value_min) / (value_max - value_min)


def add_collection_noise(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    noisy = df.copy()

    for column in FEATURE_COLUMNS:
        missing_index = rng.choice(noisy.index, size=max(8, len(noisy) // 50), replace=False)
        noisy.loc[missing_index[: len(missing_index) // 2], column] = np.nan

    outlier_index = rng.choice(noisy.index, size=15, replace=False)
    noisy.loc[outlier_index[:5], "followers"] = noisy["followers"].max() * 6
    noisy.loc[outlier_index[5:10], "video_watch_minutes"] = 2400
    noisy.loc[outlier_index[10:], "night_activity_ratio"] = 1.35

    duplicate_rows = noisy.sample(12, random_state=42).copy()
    noisy_duplicates = noisy.sample(8, random_state=43).copy()
    noisy_duplicates["likes_per_week"] = noisy_duplicates["likes_per_week"].fillna(0) + 3

    return pd.concat([noisy, duplicate_rows, noisy_duplicates], ignore_index=True)


def generate_social_media_data(sample_size: int = 1200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Latent behavioral factors are based on common user profiling dimensions:
    # activity, interaction, creation, night use, video consumption, commerce, and social scale.
    activity = rng.normal(0, 1, sample_size)
    interaction = 0.45 * activity + rng.normal(0, 0.9, sample_size)
    creator = 0.35 * activity + 0.35 * interaction + rng.normal(0, 0.95, sample_size)
    night = rng.normal(0, 1, sample_size)
    video = 0.30 * activity + 0.25 * night + rng.normal(0, 0.95, sample_size)
    commerce = 0.30 * video + 0.25 * interaction + rng.normal(0, 0.9, sample_size)
    social_scale = 0.55 * creator + 0.30 * interaction + rng.normal(0, 0.85, sample_size)

    activity_score = sigmoid(activity)
    interaction_score = sigmoid(interaction)
    creator_score = sigmoid(creator)
    night_score = sigmoid(night)
    video_score = sigmoid(video)
    commerce_score = sigmoid(commerce)
    social_score = sigmoid(social_scale)

    posts_lambda = np.clip(0.6 + 4.2 * creator_score + 2.2 * activity_score, 0.2, 18)
    posts_per_week = rng.poisson(posts_lambda)

    followers = np.round(
        rng.lognormal(
            mean=6.2 + 1.35 * social_score + 0.75 * creator_score,
            sigma=0.55,
            size=sample_size,
        )
    ).astype(int)

    following = np.round(
        80
        + 900 * interaction_score
        + 420 * activity_score
        + rng.normal(0, 120, sample_size)
    ).astype(int)

    likes_per_week = np.round(
        20
        + 260 * interaction_score
        + 0.020 * followers
        + 35 * posts_per_week
        + rng.normal(0, 45, sample_size)
    ).astype(int)

    comments_per_week = np.round(
        4
        + 70 * interaction_score
        + 8 * posts_per_week
        + rng.normal(0, 14, sample_size)
    ).astype(int)

    shares_per_week = np.round(
        2
        + 55 * creator_score
        + 35 * interaction_score
        + rng.normal(0, 11, sample_size)
    ).astype(int)

    avg_session_minutes = np.round(
        8
        + 60 * activity_score
        + 45 * video_score
        + 22 * night_score
        + rng.normal(0, 10, sample_size),
        1,
    )

    active_days_per_week = np.round(
        1 + 6 * sigmoid(1.4 * activity + rng.normal(0, 0.45, sample_size))
    ).astype(int)

    night_activity_ratio = np.round(
        rng.beta(
            1.2 + 8 * night_score,
            1.2 + 8 * (1 - night_score),
            size=sample_size,
        ),
        3,
    )

    video_watch_minutes = np.round(
        25
        + 760 * video_score
        + 130 * activity_score
        + rng.normal(0, 70, sample_size),
        1,
    )

    purchase_clicks_per_week = np.round(
        rng.poisson(np.clip(0.4 + 12 * commerce_score + 2 * activity_score, 0.1, 28))
    ).astype(int)

    df = pd.DataFrame(
        {
            "user_id": np.arange(100000, 100000 + sample_size),
            "posts_per_week": np.clip(posts_per_week, 0, 35),
            "likes_per_week": np.clip(likes_per_week, 0, 1200),
            "comments_per_week": np.clip(comments_per_week, 0, 300),
            "shares_per_week": np.clip(shares_per_week, 0, 220),
            "followers": np.clip(followers, 1, 25000),
            "following": np.clip(following, 1, 5000),
            "avg_session_minutes": np.clip(avg_session_minutes, 5, 240),
            "active_days_per_week": np.clip(active_days_per_week, 1, 7),
            "night_activity_ratio": np.clip(night_activity_ratio, 0, 1),
            "video_watch_minutes": np.clip(video_watch_minutes, 5, 1500),
            "purchase_clicks_per_week": np.clip(purchase_clicks_per_week, 0, 35),
        }
    )

    # Shuffle users so row order does not carry any latent generation pattern.
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df = add_collection_noise(df, rng)
    df["user_id"] = df["user_id"].map(lambda value: f"U{int(value):06d}")
    return df


def main() -> None:
    ensure_directories()
    data = generate_social_media_data()
    data.to_csv(RAW_DATA_PATH, index=False, encoding="utf-8-sig")
    print(f"Raw dataset saved to: {RAW_DATA_PATH}")
    print(f"Generated rows: {len(data)}")


if __name__ == "__main__":
    main()
