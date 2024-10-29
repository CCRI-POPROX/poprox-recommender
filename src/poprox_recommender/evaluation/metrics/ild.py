import numpy as np

from poprox_concepts.domain import ArticleSet
from poprox_recommender.pytorch.datachecks import assert_tensor_size
from poprox_recommender.topics import extract_general_topics

ILD_TOPIC = 0
ILD_EMBEDDING = 1


def intra_list_distance(recs_list: ArticleSet, sim_type: int = ILD_TOPIC, k: int = 10) -> float:
    """
    Computes a pair-wise intra-list distance metric:
    For all pairs <i1, i2> in L, average (1 - sim(i1, i2)).
    """
    summands = []
    for i1 in range(0, k + 1):
        for i2 in range(0, k + 1):
            if i1 != i2:
                art1 = recs_list.articles[i1]
                art2 = recs_list.articles[i2]
                topics_a1 = extract_general_topics(art1)
                topics_a2 = extract_general_topics(art2)
                sim = 0
                if sim_type == ILD_TOPIC:
                    sim = topic_sim(topics_a1, topics_a2)
                elif sim_type == ILD_EMBEDDING:
                    sim = embedding_sim(recs_list, k=10)
                    sim = np.nan
                summands.append(1 - sim)

    ild = np.mean(summands, dtype=np.float64)

    return float(ild)


def topic_sim(topics1: set[str], topics2: set[str]):
    """
    Jaccard similarity for binary valued features
    J(a, b) = size(intersection(a, b)) / size(union(a, b))
    """
    inter = topics1.intersection(topics2)
    union = topics1.union(topics2)
    jaccard = len(inter) / len(union)

    return jaccard


def embedding_sim(recs_list: ArticleSet, k: int = 10):
    """
    Cosine similarity for embedding vectors
    """
    similarity_matrix = compute_similarity_matrix(recs_list.embeddings)

    return np.nan


def compute_similarity_matrix(recs_list):
    """
    From MMR implementation
    """
    num_values = len(recs_list)
    # M is (n, k), where n = # articles & k = embed. dim.
    # M M^T is (n, n) matrix of pairwise dot products
    similarity_matrix = recs_list @ recs_list.T
    assert_tensor_size(similarity_matrix, num_values, num_values, label="sim-matrix", prefix=False)
    return similarity_matrix
