import json

import numpy as np
import pandas as pd
import torch as th
from tqdm import tqdm

from poprox_recommender.paths import model_file_path, project_root


def sim_score(m_emb, a_emb):
    m_emb = th.as_tensor(m_emb, dtype=th.float32)
    a_emb = th.as_tensor(a_emb, dtype=th.float32)
    m_emb = m_emb / m_emb.norm()
    a_emb = a_emb / a_emb.norm()
    cos_sim = th.dot(m_emb, a_emb)
    return cos_sim.item()


def get_neighbor_from_AP(AP_emb, m_emb):
    single_article_neighbor_dict = {}
    for a_article in AP_emb.itertuples():
        a_id = a_article.article_id
        a_emb = a_article.embedding
        single_article_neighbor_dict[a_id] = sim_score(m_emb, a_emb)
    return single_article_neighbor_dict


def k_nearest_neighbors(threshold, all_neighbors):
    return sorted(all_neighbors.items(), key=lambda x: x[1], reverse=True)[:threshold]


def assigning_neighbors_topics(top_neighbors, ap_emb):
    neighbor_ids = [aid for aid, _ in top_neighbors]
    ap_meta = ap_emb.loc[ap_emb["article_id"].isin(neighbor_ids), ["article_id", "headline", "topic_name"]].copy()
    score_map = dict(top_neighbors)
    ap_meta["score"] = ap_meta["article_id"].map(score_map)
    ap_meta = ap_meta.sort_values("score", ascending=False, kind="mergesort")
    topics_union = {t for lst in ap_meta["topic_name"] for t in lst}

    return topics_union


# data read
data = project_root() / "models" / "precalculated_model"
ap_emb = pd.read_parquet(data / "ap_emb.parquet")
mind_emb = pd.read_parquet(data / "mind_emb.parquet")

threshold = 5


mind_article_top_neighbors = {}
mind_article_topics = {}

for m_article in mind_emb.itertuples():
    m_id = m_article.article_id
    m_headline = m_article.headline
    m_emb = m_article.embedding

    all_neighbors = get_neighbor_from_AP(ap_emb, m_emb)

    top_neighbors = k_nearest_neighbors(threshold, all_neighbors)
    mind_article_top_neighbors[m_id] = top_neighbors

    topics_union = assigning_neighbors_topics(top_neighbors, ap_emb)
    mind_article_topics[m_id] = [m_headline, topics_union]

breakpoint()
