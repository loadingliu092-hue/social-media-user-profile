from __future__ import annotations

import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from utils import (
    CLEANED_DATA_PATH,
    CLUSTERED_DATA_PATH,
    DBSCAN_SUMMARY_PATH,
    FEATURE_COLUMNS,
    K_SELECTION_SCORES_PATH,
    KMEANS_MODEL_PATH,
    K_SELECTION_PATH,
    PCA_CLUSTER_PATH,
    TRAINING_LOG_PATH,
    ensure_directories,
)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(
    style="whitegrid",
    font="Microsoft YaHei",
    rc={"axes.unicode_minus": False},
)


PROFILE_NAME_ORDER = [
    "Content Creator",
    "Interactive User",
    "Night Owl",
    "Low-frequency Browser",
]

EXTRA_PROFILE_NAMES = [
    "General Active User",
    "Regular User",
]


def evaluate_k_values(scaled_data: np.ndarray, k_range: range) -> pd.DataFrame:
    records = []
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(scaled_data)
        records.append(
            {
                "k": k,
                "silhouette_score": silhouette_score(scaled_data, labels),
            }
        )
    return pd.DataFrame(records)


def save_k_selection_figure(scores: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 6))
    sns.lineplot(data=scores, x="k", y="silhouette_score", marker="o", color="#e76f51")
    plt.title("Silhouette Score Comparison for K=2 to K=8")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.xticks(scores["k"].tolist())
    plt.tight_layout()
    plt.savefig(K_SELECTION_PATH, dpi=220, bbox_inches="tight")
    plt.close()


def search_dbscan_parameters(scaled_data: np.ndarray) -> tuple[DBSCAN, np.ndarray, pd.DataFrame, dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for eps in np.round(np.arange(1.0, 4.01, 0.25), 2):
        for min_samples in (5, 10, 14, 20):
            model = DBSCAN(eps=float(eps), min_samples=min_samples)
            labels = model.fit_predict(scaled_data)
            cluster_labels = sorted(label for label in set(labels) if label != -1)
            cluster_count = len(cluster_labels)
            noise_count = int((labels == -1).sum())
            noise_ratio = noise_count / len(labels)
            candidates.append(
                {
                    "eps": float(eps),
                    "min_samples": min_samples,
                    "model": model,
                    "labels": labels,
                    "cluster_count": cluster_count,
                    "noise_count": noise_count,
                    "noise_ratio": noise_ratio,
                    "is_valid": cluster_count >= 2 and cluster_count <= 6 and noise_count < len(labels),
                }
            )

    valid_candidates = [candidate for candidate in candidates if candidate["is_valid"]]
    if valid_candidates:
        selected = sorted(valid_candidates, key=lambda item: (item["noise_ratio"], -item["cluster_count"]))[0]
        note = (
            f"DBSCAN selected eps={selected['eps']}, min_samples={selected['min_samples']}; "
            f"found {selected['cluster_count']} clusters with noise ratio {selected['noise_ratio']:.2%}."
        )
    else:
        selected = next(
            candidate
            for candidate in candidates
            if candidate["eps"] == 1.25 and candidate["min_samples"] == 14
        )
        note = (
            "DBSCAN did not find a suitable non-noise clustering result with 2 to 6 clusters "
            "in the tested parameter grid; DBSCAN is not suitable for the current standardized "
            "high-dimensional behavior features under tested parameters."
        )

    summary = (
        pd.Series(selected["labels"])
        .value_counts()
        .sort_index()
        .rename_axis("dbscan_cluster")
        .reset_index(name="user_count")
    )
    info = {
        "eps": selected["eps"],
        "min_samples": selected["min_samples"],
        "cluster_count": selected["cluster_count"],
        "noise_count": selected["noise_count"],
        "noise_ratio": selected["noise_ratio"],
        "note": note,
        "found_suitable_result": bool(valid_candidates),
    }
    return selected["model"], selected["labels"], summary, info


def safe_pick(df: pd.DataFrame, score_column: str, used_clusters: set[int]) -> int | None:
    available = df.loc[~df.index.isin(used_clusters)]
    if available.empty:
        return None
    return int(available[score_column].idxmax())


def reorder_clusters(cluster_profile: pd.DataFrame) -> dict[int, str]:
    scored_profile = cluster_profile.assign(
        creator_score=cluster_profile["posts_per_week"] + cluster_profile["shares_per_week"] * 0.8,
        social_score=cluster_profile["likes_per_week"] + cluster_profile["comments_per_week"] * 2,
        night_score=cluster_profile["night_activity_ratio"] * 100 + cluster_profile["video_watch_minutes"] * 0.05,
        low_freq_score=-(cluster_profile["active_days_per_week"] * 10 + cluster_profile["avg_session_minutes"]),
    )

    mapping: dict[int, str] = {}
    used_clusters: set[int] = set()

    selection_rules = [
        ("creator_score", PROFILE_NAME_ORDER[0]),
        ("social_score", PROFILE_NAME_ORDER[1]),
        ("night_score", PROFILE_NAME_ORDER[2]),
        ("low_freq_score", PROFILE_NAME_ORDER[3]),
    ]

    for score_column, profile_name in selection_rules:
        picked_cluster = safe_pick(scored_profile, score_column, used_clusters)
        if picked_cluster is None:
            continue
        mapping[picked_cluster] = profile_name
        used_clusters.add(picked_cluster)

    remaining_clusters = [int(cluster_id) for cluster_id in scored_profile.index if int(cluster_id) not in used_clusters]
    for idx, cluster_id in enumerate(remaining_clusters):
        mapping[cluster_id] = EXTRA_PROFILE_NAMES[idx % len(EXTRA_PROFILE_NAMES)]

    return mapping


def save_pca_cluster_figure(pca_features: np.ndarray, labels: np.ndarray, label_names: list[str]) -> None:
    plot_df = pd.DataFrame(
        {
            "PCA1": pca_features[:, 0],
            "PCA2": pca_features[:, 1],
            "cluster": labels,
            "cluster_name": label_names,
        }
    )

    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=plot_df,
        x="PCA1",
        y="PCA2",
        hue="cluster_name",
        palette="Set2",
        s=55,
        alpha=0.8,
    )
    plt.title("PCA 2D Cluster Visualization")
    plt.tight_layout()
    plt.savefig(PCA_CLUSTER_PATH, dpi=220, bbox_inches="tight")
    plt.close()


def write_training_log(
    sample_count: int,
    silhouette: float,
    explained_variance_ratio: list[float],
    k_scores: pd.DataFrame,
    dbscan_info: dict[str, object],
    dbscan_summary: pd.DataFrame,
    fixed_k: int,
) -> None:
    lines = [
        f"Data sample count: {sample_count}",
        f"Feature columns: {', '.join(FEATURE_COLUMNS)}",
        f"KMeans K used: {fixed_k}",
        f"KMeans silhouette score: {silhouette:.4f}",
        "K selection comparison:",
    ]
    for row in k_scores.itertuples(index=False):
        lines.append(f"  K={row.k}: silhouette_score={row.silhouette_score:.4f}")
    lines.extend(
        [
            "K=2 has the highest silhouette score, but K=4 is selected for finer-grained user profiling and business interpretability.",
            f"PCA explained variance ratio: {explained_variance_ratio}",
            f"DBSCAN eps: {dbscan_info['eps']}",
            f"DBSCAN min_samples: {dbscan_info['min_samples']}",
            f"DBSCAN cluster count excluding noise: {dbscan_info['cluster_count']}",
            f"DBSCAN noise count: {dbscan_info['noise_count']}",
            f"DBSCAN noise ratio: {dbscan_info['noise_ratio']:.2%}",
            f"DBSCAN result note: {dbscan_info['note']}",
            "DBSCAN cluster distribution:",
        ]
    )
    for row in dbscan_summary.itertuples(index=False):
        lines.append(f"  cluster {row.dbscan_cluster}: {row.user_count}")

    TRAINING_LOG_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_clustering() -> dict[str, object]:
    ensure_directories()
    df = pd.read_csv(CLEANED_DATA_PATH)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[FEATURE_COLUMNS])

    k_scores = evaluate_k_values(scaled_data, range(2, 9))
    k_scores.to_csv(K_SELECTION_SCORES_PATH, index=False, encoding="utf-8-sig")
    save_k_selection_figure(k_scores)

    fixed_k = 4
    kmeans = KMeans(n_clusters=fixed_k, random_state=42, n_init=30)
    original_labels = kmeans.fit_predict(scaled_data)
    silhouette = silhouette_score(scaled_data, original_labels)

    cluster_profile = pd.DataFrame(scaled_data, columns=FEATURE_COLUMNS).assign(cluster=original_labels).groupby("cluster").mean()
    profile_name_map = reorder_clusters(cluster_profile)
    cluster_names = [profile_name_map[int(label)] for label in original_labels]

    pca = PCA(n_components=2, random_state=42)
    pca_features = pca.fit_transform(scaled_data)
    save_pca_cluster_figure(pca_features, original_labels, cluster_names)

    dbscan, dbscan_labels, dbscan_summary, dbscan_info = search_dbscan_parameters(scaled_data)
    dbscan_summary.to_csv(DBSCAN_SUMMARY_PATH, index=False, encoding="utf-8-sig")

    clustered_df = df.copy()
    clustered_df["cluster"] = original_labels
    clustered_df["cluster_name"] = cluster_names
    clustered_df["pca_1"] = pca_features[:, 0]
    clustered_df["pca_2"] = pca_features[:, 1]
    clustered_df.to_csv(CLUSTERED_DATA_PATH, index=False, encoding="utf-8-sig")

    explained_variance_ratio = pca.explained_variance_ratio_.tolist()
    write_training_log(
        sample_count=len(df),
        silhouette=float(silhouette),
        explained_variance_ratio=explained_variance_ratio,
        k_scores=k_scores,
        dbscan_info=dbscan_info,
        dbscan_summary=dbscan_summary,
        fixed_k=fixed_k,
    )

    with open(KMEANS_MODEL_PATH, "wb") as file:
        pickle.dump(
            {
                "model": kmeans,
                "scaler": scaler,
                "pca": pca,
                "feature_columns": FEATURE_COLUMNS,
                "profile_name_map": profile_name_map,
                "best_k": fixed_k,
                "silhouette_score": float(silhouette),
                "k_selection_scores": k_scores.to_dict(orient="records"),
                "dbscan_model": dbscan,
                "dbscan_info": dbscan_info,
            },
            file,
        )

    return {
        "best_k": fixed_k,
        "silhouette_score": float(silhouette),
        "explained_variance_ratio": explained_variance_ratio,
    }


def main() -> None:
    results = run_clustering()
    print(f"Final K used: {results['best_k']}")
    print(f"K-Means silhouette score: {results['silhouette_score']:.4f}")
    print(f"PCA explained variance ratio: {results['explained_variance_ratio']}")


if __name__ == "__main__":
    main()
