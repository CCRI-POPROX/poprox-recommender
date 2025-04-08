from copy import copy

import torch
from lenskit.pipeline import Component

from poprox_concepts import CandidateSet, InterestProfile
from poprox_recommender.pytorch.decorators import torch_inference


class ArticleScorer(Component):
    config: None

    @torch_inference
    def __call__(self, candidate_articles: CandidateSet, interest_profile: InterestProfile) -> CandidateSet:
        candidate_embeddings = candidate_articles.embeddings
        user_embedding = interest_profile.embedding

        scored_articles = copy(candidate_articles)
        if user_embedding is not None:
            pred = torch.matmul(candidate_embeddings, user_embedding.t()).squeeze()
            scored_articles.scores = pred.cpu().detach().numpy()
        else:
            scored_articles.scores = None

        return scored_articles
