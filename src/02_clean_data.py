from __future__ import annotations

import pandas as pd

from utils import CLEANED_DATA_PATH, FEATURE_COLUMNS, RAW_DATA_PATH, ensure_directories


INTEGER_COLUMNS = [
    "posts_per_week",
    "likes_per_week",
    "comments_per_week",
    "shares_per_week",
    "followers",
    "following",
    "active_days_per_week",
    "purchase_clicks_per_week",
]

FLOAT_COLUMNS = [
    "avg_session_minutes",
    "night_activity_ratio",
    "video_watch_minutes",
]


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    for column in FEATURE_COLUMNS:
        median_value = cleaned[column].median()
        cleaned[column] = cleaned[column].fillna(median_value)
    return cleaned


def cap_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    for column in FEATURE_COLUMNS:
        q1 = cleaned[column].quantile(0.25)
        q3 = cleaned[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        cleaned[column] = cleaned[column].clip(lower=lower, upper=upper)
    return cleaned


def standardize_formats(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned["user_id"] = cleaned["user_id"].astype(str).str.strip().str.upper()

    for column in INTEGER_COLUMNS:
        cleaned[column] = cleaned[column].round().astype(int)

    for column in FLOAT_COLUMNS:
        cleaned[column] = cleaned[column].astype(float).round(3)

    cleaned["night_activity_ratio"] = cleaned["night_activity_ratio"].clip(0, 1).round(3)
    cleaned["active_days_per_week"] = cleaned["active_days_per_week"].clip(1, 7)

    for column in INTEGER_COLUMNS:
        cleaned[column] = cleaned[column].clip(lower=0)

    return cleaned


def clean_dataset() -> pd.DataFrame:
    ensure_directories()
    df = pd.read_csv(RAW_DATA_PATH)
    df = df.drop_duplicates()
    df = fill_missing_values(df)
    df = cap_outliers_iqr(df)
    df = standardize_formats(df)
    df = df.drop_duplicates(subset=["user_id"], keep="first").sort_values("user_id").reset_index(drop=True)
    return df


def main() -> None:
    cleaned = clean_dataset()
    cleaned.to_csv(CLEANED_DATA_PATH, index=False, encoding="utf-8-sig")
    print(f"Cleaned dataset saved to: {CLEANED_DATA_PATH}")
    print(f"Remaining rows: {len(cleaned)}")


if __name__ == "__main__":
    main()
