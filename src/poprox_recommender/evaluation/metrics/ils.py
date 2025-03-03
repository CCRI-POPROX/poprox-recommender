import numpy as np  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity

from poprox_concepts.domain import ArticleSet


def intralist_similarity(final_recs: ArticleSet, k: int):
    similarity_matrix = cosine_similarity(final_recs.embeddings)  # figure the cached embeddings out
    n = len(final_recs.articles)
    if n <= 1:
        return 1.0
    intralist_similarity_score = np.sum(np.triu(similarity_matrix, 1)) / (n * (n - 1) / 2)

    return intralist_similarity_score
