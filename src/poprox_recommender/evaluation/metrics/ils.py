import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from poprox_concepts.domain import CandidateSet


def intralist_similarity(final_recs: CandidateSet, k: int):
    if not hasattr(final_recs, "embeddings") or final_recs.embeddings is None:
        return np.nan
    top_k_embeddings = final_recs.embeddings[:k]
    n = len(top_k_embeddings)

    if n <= 1:
        return 1.0
    similarity_matrix = cosine_similarity(top_k_embeddings.cpu().numpy())
    intralist_similarity_score = np.sum(np.triu(similarity_matrix, 1)) / (n * (n - 1) / 2)

    return float(intralist_similarity_score)
