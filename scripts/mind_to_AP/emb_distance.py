import numpy as np
import pandas as pd
import torch as th

from poprox_recommender.paths import project_root

device = th.device("cuda" if th.cuda.is_available() else "cpu")


def clustering_neighbor_from_AP(similarity, neighbor_count, M_ids, A_ids_arr):
    scores, idxs = th.topk(similarity, k=neighbor_count, dim=1, largest=True, sorted=True)
    scores = scores.cpu().numpy()  # score of top nearest AP neighbors
    idxs = idxs.cpu().numpy()  # id of top nearest AP neighbors
    rows = []
    for i, m_id in enumerate(M_ids):
        for ap_idx, sc in zip(idxs[i], scores[i]):
            rows.append((m_id, A_ids_arr[ap_idx], float(sc)))
    return pd.DataFrame(rows, columns=["mind_id", "ap_id", "score"])


data_dir = project_root() / "models" / "precalculated_model"
ap_emb = pd.read_parquet(data_dir / "ap_emb.parquet")
mind_emb = pd.read_parquet(data_dir / "mind_emb.parquet")

A_ids = ap_emb["article_id"].to_numpy()
A_topics = ap_emb[["article_id", "topic_name"]].rename(columns={"article_id": "ap_id"})  # type: ignore
A_mat = th.stack([th.tensor(x, dtype=th.float32, device=device) for x in ap_emb["embedding"]], dim=0)
A_ids_arr = np.asarray(A_ids)  # keeping track of AP ids according collumn


M_ids = mind_emb["article_id"].to_numpy()
M_heads = mind_emb["headline"].to_numpy()
M_mat = th.stack([th.tensor(x, dtype=th.float32, device=device) for x in mind_emb["embedding"]], dim=0)


similarity = M_mat @ A_mat.T  # matrix multiplication


neighbor_count = 5
mind_neighbors_df = clustering_neighbor_from_AP(similarity, neighbor_count, M_ids, A_ids_arr)


mind_topics = mind_neighbors_df.merge(A_topics, on="ap_id", how="left")
mind_topics = mind_topics.explode("topic_name", ignore_index=True)  # to use group by operation
mind_topic_sets = mind_topics.groupby("mind_id")["topic_name"].agg(lambda s: set(t for t in s.dropna()))
mind_topics_rows = [
    (m_id, head, ";".join(sorted(list(mind_topic_sets.get(m_id, set()))))) for m_id, head in zip(M_ids, M_heads)
]
mind_topics_df = pd.DataFrame(mind_topics_rows, columns=["mind_id", "headline", "topics"])


mind_neighbors_df.to_csv("mind_article_top_neighbors.csv", index=False)
mind_topics_df.to_csv("mind_article_topics.csv", index=False)
