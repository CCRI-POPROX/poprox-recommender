#TODO CHANGE
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

        with_scores = candidate_articles.model_copy()

        #aggregation piece
        #change from one user embeddings to 32, which gets 32 scores for each article. and then aggregate (average them)
        #need to create mind interest profile (list)
        if user_embedding is not None:
            pred = torch.matmul(candidate_embeddings, user_embedding.t()).squeeze()
            with_scores.scores = pred.cpu().detach().numpy()
        else:
            with_scores.scores = None

        return with_scores
