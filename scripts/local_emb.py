from pathlib import Path

import numpy as np
import pandas as pd
from topic_des_vs_single_tag_vs_multi_tag import (
    multi_topical_articles,
    single_topical_articles,
    topic_des_as_articles,
)

from poprox_recommender.paths import project_root


def export_topic_angle_stats_to_excel(
    single_topic_angle_stats: dict,
    multi_topic_angle_stats: dict,
    single_vs_multi_topic_angle_stats: dict,
    save_path: str,
):
    all_topics = sorted(
        set(single_topic_angle_stats) | set(multi_topic_angle_stats) | set(single_vs_multi_topic_angle_stats)
    )

    rows = []
    for topic in all_topics:
        row = {
            "Topic": topic,
            "Description vs Single": single_topic_angle_stats.get(topic, {}).get("mean_angle", float("nan")),
            "Description vs Multi": multi_topic_angle_stats.get(topic, {}).get("mean_angle", float("nan")),
            "Single vs Multi": single_vs_multi_topic_angle_stats.get(topic, {}).get("mean_angle", float("nan")),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)


def angle_between_vectors(a: np.ndarray, b: np.ndarray) -> float:
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    cos_sim = np.dot(a, b)
    return cos_sim


def angle_calculator(single_topic_angle_stats, topic, embeddings_1, embeddings_2):
    if embeddings_1 and embeddings_2:
        single_angles = []
        for desc_emb in embeddings_1:
            for art_emb in embeddings_2:
                s_angle = angle_between_vectors(desc_emb, art_emb)
                single_angles.append(s_angle)

        single_topic_angle_stats[topic] = {
            "mean_angle": np.mean(single_angles),
            "min_angle": np.min(single_angles),
            "max_angle": np.max(single_angles),
            "count": len(single_angles),
        }


if __name__ == "__main__":
    data = project_root() / "data"

    single_topic_angle_stats = {}
    multi_topic_angle_stats = {}
    single_vs_multi_topic_angle_stats = {}

    for topic in topic_des_as_articles.keys():
        desc_embeddings = [np.array(a["embedding"]) for a in topic_des_as_articles[topic]]
        single_article_embeddings = [np.array(a["embedding"]) for a in single_topical_articles.get(topic, [])]
        multi_article_embeddings = [np.array(a["embedding"]) for a in multi_topical_articles.get(topic, [])]

        angle_calculator(single_topic_angle_stats, topic, desc_embeddings, single_article_embeddings)
        angle_calculator(multi_topic_angle_stats, topic, desc_embeddings, multi_article_embeddings)
        angle_calculator(single_vs_multi_topic_angle_stats, topic, single_article_embeddings, multi_article_embeddings)

    export_topic_angle_stats_to_excel(
        single_topic_angle_stats,
        multi_topic_angle_stats,
        single_vs_multi_topic_angle_stats,
        save_path=Path(data / "Test" / "topic_angle_comparison_5_sim.csv"),
    )
