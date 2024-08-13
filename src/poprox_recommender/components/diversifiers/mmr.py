import torch

from poprox_concepts import ArticleSet, InterestProfile
from poprox_recommender.lkpipeline import Component
from poprox_recommender.pytorch.decorators import torch_inference


class MMRDiversifier(Component):
    def __init__(self, theta: float = 0.8, num_slots: int = 10):
        self.theta = theta
        self.num_slots = num_slots

    @torch_inference
    def __call__(self, candidate_articles: ArticleSet, interest_profile: InterestProfile) -> ArticleSet:
        if candidate_articles.scores is None:
            return candidate_articles

        similarity_matrix = compute_similarity_matrix(candidate_articles.embeddings)

        article_indices = mmr_diversification(
            candidate_articles.scores, similarity_matrix, theta=self.theta, topk=self.num_slots
        )

        return ArticleSet(articles=[candidate_articles.articles[int(idx)] for idx in article_indices])


def compute_similarity_matrix(todays_article_vectors):
    num_values = len(todays_article_vectors)
    # M is (n, k), where n = # articles & k = embed. dim.
    # M M^T is (n, n) matrix of pairwise dot products
    similarity_matrix = todays_article_vectors @ todays_article_vectors.T
    assert similarity_matrix.size == (num_values, num_values)
    return similarity_matrix


def mmr_diversification(rewards, similarity_matrix, theta: float, topk: int):
    # MR_i = \theta * reward_i - (1 - \theta)*max_{j \in S} sim(i, j) # S us
    # R is all candidates (not selected yet)

    S = []  # final recommendation (topk index)
    # first recommended item
    S.append(rewards.argmax())

    for k in range(topk - 1):
        # find the best combo of reward and max sim to existing item
        # first, let's pare the matrix: candidates on rows, selected items on cols
        M = similarity_matrix[:, S]

        # for each target item, we want to find the *max* simialrity to an existing.
        # we do this by taking the min of each row.
        scores = torch.max(M, axis=1)
        assert len(scores) == len(rewards)

        # now, we want to compute θ*r - (1-θ)*s. let's do that *in-place* using
        # this scores vector. To start, multiply by θ-1 (-(1-θ)):
        scores *= theta - 1

        # with this, we can theta * rewards in-place:
        scores.add_(rewards, theta)

        # now, we're looking for the *max* score in this list. we can do this
        # in two steps. step 1: clear the items we already have:
        scores[S] = -torch.inf
        # step 2: find the largest value
        S.append(torch.argmax(scores))

    return S
