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

        if user_embedding is not None:
            pred = torch.matmul(candidate_embeddings, user_embedding.t()).squeeze()
            candidate_articles.scores = pred.cpu().detach().numpy()
        else:
            candidate_articles.scores = None

        return candidate_articles
