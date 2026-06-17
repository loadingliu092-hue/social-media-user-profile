from __future__ import annotations

from datetime import date
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils import (
    REAL_WEIBO_AGE_FIGURE_PATH,
    REAL_WEIBO_DATA_DIR,
    REAL_WEIBO_GENDER_FIGURE_PATH,
    REAL_WEIBO_NETWORK_FIGURE_PATH,
    REAL_WEIBO_NETWORK_PATH,
    REAL_WEIBO_PROFILE_PATH,
    REAL_WEIBO_PROVINCE_FIGURE_PATH,
    REAL_WEIBO_REPORT_PATH,
    REAL_WEIBO_SUMMARY_PATH,
    ensure_directories,
)


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(
    style="whitegrid",
    font="Microsoft YaHei",
    rc={"axes.unicode_minus": False},
)


REQUIRED_FILES = [
    "train_info.txt",
    "train_labels.txt",
    "train_links.txt",
    "valid_info.txt",
    "valid_links.txt",
    "valid_nolabel.txt",
]


def check_required_files() -> None:
    missing_files = [name for name in REQUIRED_FILES if not (REAL_WEIBO_DATA_DIR / name).exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing files in data/real_weibo/: {', '.join(missing_files)}")


def split_line(line: str) -> list[str]:
    text = line.strip()
    if not text:
        return []
    return re.split(r"[\t, ]+", text)


def parse_label_line(line: str) -> dict[str, str | None] | None:
    text = line.strip()
    if not text:
        return None

    parts = text.split("||")
    if len(parts) < 4:
        return None

    user_id = parts[0].strip()
    gender = parts[1].strip()
    birth_year = parts[2].strip()
    location_text = parts[3].strip()
    location_parts = location_text.split(maxsplit=1)
    province = location_parts[0] if location_parts else "None"
    city = location_parts[1] if len(location_parts) > 1 else "None"

    return {
        "user_id": user_id,
        "gender": gender,
        "birth_year": birth_year,
        "province": province,
        "city": city,
    }


def read_labels_txt(file_path) -> pd.DataFrame:
    records: list[dict[str, str | None]] = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        for raw_line in file:
            parsed = parse_label_line(raw_line)
            if parsed is not None:
                records.append(parsed)
    return pd.DataFrame(records, columns=["user_id", "gender", "birth_year", "province", "city"])


def read_links_txt(file_path) -> pd.DataFrame:
    records: list[dict[str, str]] = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        for raw_line in file:
            parts = split_line(raw_line)
            if len(parts) < 2:
                continue
            source = parts[0]
            for target in parts[1:]:
                records.append({"source": source, "target": target})
    return pd.DataFrame(records, columns=["source", "target"])


def clean_user_profile(labels_df: pd.DataFrame) -> pd.DataFrame:
    cleaned = labels_df.copy()
    cleaned["gender"] = cleaned["gender"].replace({"": np.nan, "None": np.nan, "none": np.nan})
    cleaned = cleaned.dropna(subset=["gender"]).copy()

    cleaned["birth_year"] = pd.to_numeric(cleaned["birth_year"], errors="coerce")
    cleaned = cleaned.dropna(subset=["birth_year"]).copy()

    current_year = date.today().year
    cleaned["age"] = current_year - cleaned["birth_year"].astype(int)
    cleaned = cleaned[(cleaned["age"] >= 10) & (cleaned["age"] <= 100)].copy()

    cleaned["province"] = cleaned["province"].replace({None: "None", "": "None", "none": "None", "NONE": "None"}).fillna("None")
    cleaned["city"] = cleaned["city"].replace({None: "None", "": "None", "none": "None", "NONE": "None"}).fillna("None")
    cleaned = cleaned.drop_duplicates(subset=["user_id"]).reset_index(drop=True)
    return cleaned


def build_network_features(links_df: pd.DataFrame) -> pd.DataFrame:
    links = links_df.dropna(subset=["source", "target"]).copy()
    links["source"] = links["source"].astype(str)
    links["target"] = links["target"].astype(str)

    out_degree = links.groupby("source").size().rename("out_degree")
    in_degree = links.groupby("target").size().rename("in_degree")
    all_users = pd.Index(sorted(set(links["source"]).union(set(links["target"]))), name="user_id")

    network_features = pd.DataFrame(index=all_users).join(out_degree).join(in_degree).fillna(0)
    network_features[["out_degree", "in_degree"]] = network_features[["out_degree", "in_degree"]].astype(int)
    network_features["total_degree"] = network_features["out_degree"] + network_features["in_degree"]
    return network_features.reset_index()


def save_gender_distribution(profile_df: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 5))
    sns.countplot(data=profile_df, x="gender", hue="gender", palette="Set2", legend=False)
    plt.title("\u5fae\u535a\u7528\u6237\u6027\u522b\u5206\u5e03")
    plt.xlabel("\u6027\u522b")
    plt.ylabel("\u4eba\u6570")
    plt.tight_layout()
    plt.savefig(REAL_WEIBO_GENDER_FIGURE_PATH, dpi=220, bbox_inches="tight")
    plt.close()


def save_age_distribution(profile_df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    sns.histplot(profile_df["age"], bins=20, kde=True, color="#2a9d8f")
    plt.title("\u5fae\u535a\u7528\u6237\u5e74\u9f84\u5206\u5e03")
    plt.xlabel("\u5e74\u9f84")
    plt.ylabel("\u4eba\u6570")
    plt.tight_layout()
    plt.savefig(REAL_WEIBO_AGE_FIGURE_PATH, dpi=220, bbox_inches="tight")
    plt.close()


def save_province_top10(profile_df: pd.DataFrame) -> pd.Series:
    province_counts = profile_df["province"].fillna("None").value_counts().head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=province_counts.values, y=province_counts.index, hue=province_counts.index, palette="Blues_r", legend=False)
    plt.title("\u5fae\u535a\u7528\u6237\u7701\u4efd Top10")
    plt.xlabel("\u4eba\u6570")
    plt.ylabel("\u7701\u4efd")
    plt.tight_layout()
    plt.savefig(REAL_WEIBO_PROVINCE_FIGURE_PATH, dpi=220, bbox_inches="tight")
    plt.close()
    return province_counts


def save_degree_distribution(network_df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    sns.histplot(network_df["total_degree"], bins=30, kde=True, color="#e76f51")
    plt.title("\u5fae\u535a\u7528\u6237\u7f51\u7edc\u5ea6\u5206\u5e03")
    plt.xlabel("\u603b\u5ea6\u6570")
    plt.ylabel("\u4eba\u6570")
    plt.tight_layout()
    plt.savefig(REAL_WEIBO_NETWORK_FIGURE_PATH, dpi=220, bbox_inches="tight")
    plt.close()


def write_report(summary_df: pd.DataFrame, province_top10: pd.Series) -> None:
    total_users = int(summary_df["user_id"].nunique())
    gender_ratio = summary_df["gender"].value_counts(normalize=True).mul(100).round(2)
    average_age = float(summary_df["age"].mean())
    average_degree = float(summary_df["total_degree"].fillna(0).mean())

    lines = [
        f"\u603b\u7528\u6237\u6570: {total_users}",
        "\u7537\u5973\u6bd4\u4f8b:",
    ]
    for gender, ratio in gender_ratio.items():
        lines.append(f"  {gender}: {ratio:.2f}%")
    lines.append(f"\u5e73\u5747\u5e74\u9f84: {average_age:.2f}")
    lines.append("Top10\u7701\u4efd:")
    for province, count in province_top10.items():
        lines.append(f"  {province}: {count}")
    lines.append(f"\u5e73\u5747\u5ea6\u6570: {average_degree:.2f}")

    REAL_WEIBO_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_directories()
    check_required_files()

    labels_df = read_labels_txt(REAL_WEIBO_DATA_DIR / "train_labels.txt")
    links_df = read_links_txt(REAL_WEIBO_DATA_DIR / "train_links.txt")

    profile_df = clean_user_profile(labels_df)
    network_df = build_network_features(links_df)

    profile_df.to_csv(REAL_WEIBO_PROFILE_PATH, index=False, encoding="utf-8-sig")
    network_df.to_csv(REAL_WEIBO_NETWORK_PATH, index=False, encoding="utf-8-sig")

    save_gender_distribution(profile_df)
    save_age_distribution(profile_df)
    province_top10 = save_province_top10(profile_df)
    save_degree_distribution(network_df)

    summary_df = profile_df.merge(network_df, on="user_id", how="left")
    summary_df[["out_degree", "in_degree", "total_degree"]] = summary_df[["out_degree", "in_degree", "total_degree"]].fillna(0).astype(int)
    summary_df.to_csv(REAL_WEIBO_SUMMARY_PATH, index=False, encoding="utf-8-sig")

    write_report(summary_df, province_top10)
    print("Real Weibo analysis completed.")


if __name__ == "__main__":
    main()
