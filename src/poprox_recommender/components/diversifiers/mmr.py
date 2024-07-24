from typing import Any

import numpy as np
from tqdm import tqdm

from poprox_concepts import ArticleSet, InterestProfile
from poprox_recommender.components.rankers import Ranker
from poprox_recommender.pytorch.decorators import torch_inference


class MMRDiversifier(Ranker):
    def __init__(self, algo_params: dict[str, Any], num_slots=10):
        self.validate_algo_params(algo_params, ["theta"])

        self.theta = float(algo_params.get("theta", 0.8))
        self.num_slots = num_slots

    @torch_inference
    def __call__(self, candidate_articles: ArticleSet, interest_profile: InterestProfile) -> ArticleSet:
        similarity_matrix = compute_similarity_matrix(candidate_articles.embeddings)

        article_indices = mmr_diversification(
            candidate_articles.scores, similarity_matrix, theta=self.theta, topk=self.num_slots
        )

        return ArticleSet(articles=[candidate_articles.articles[int(idx)] for idx in article_indices])


def compute_similarity_matrix(todays_article_vectors):
    num_values = len(todays_article_vectors)
    similarity_matrix = np.zeros((num_values, num_values))
    for i, value1 in tqdm(enumerate(todays_article_vectors), total=num_values):
        value1 = value1.detach().cpu()
        for j, value2 in enumerate(todays_article_vectors):
            if i <= j:
                value2 = value2.detach().cpu()
                similarity_matrix[i, j] = similarity_matrix[j, i] = np.dot(value1, value2)
    return similarity_matrix


def mmr_diversification(rewards, similarity_matrix, theta: float, topk: int):
    # MR_i = \theta * reward_i - (1 - \theta)*max_{j \in S} sim(i, j) # S us
    # R is all candidates (not selected yet)

    S = []  # final recommendation (topk index)
    # first recommended item
    S.append(rewards.argmax())

    for k in range(topk - 1):
        candidate = None  # next item
        best_MR = float("-inf")

        for i, reward_i in enumerate(rewards):  # iterate R for next item
            if i in S:
                continue
            max_sim = float("-inf")

            for j in S:
                sim = similarity_matrix[i][j]
                if sim > max_sim:
                    max_sim = sim

            mr_i = theta * reward_i - (1 - theta) * max_sim
            if mr_i > best_MR:
                best_MR = mr_i
                candidate = i

        if candidate is not None:
            S.append(candidate)
    return S
