import torch

from poprox_concepts import ArticleSet, InterestProfile
from poprox_recommender.pytorch.decorators import torch_inference


class ArticleScorer:
    @torch_inference
    def __call__(self, candidate_articles: ArticleSet, interest_profile: InterestProfile) -> ArticleSet:
        candidate_embeddings = candidate_articles.embeddings
        user_embedding = interest_profile.embedding

        if user_embedding is not None:
            pred = torch.matmul(candidate_embeddings, user_embedding.t()).squeeze()
            candidate_articles.scores = pred.cpu().detach().numpy()
        else:
            candidate_articles.scores = None

        return candidate_articles
