from poprox_concepts import ArticleSet, InterestProfile
from poprox_recommender.pytorch.decorators import torch_inference


class ArticleScorer:
    def __init__(self, model):
        self.model = model

    @torch_inference
    def __call__(self, candidate_articles: ArticleSet, interest_profile: InterestProfile) -> ArticleSet:
        candidate_embeddings = candidate_articles.embeddings
        user_embedding = interest_profile.embedding

        pred = self.model.get_prediction(candidate_embeddings, user_embedding.squeeze())
        candidate_articles.scores = pred.cpu().detach().numpy()

        return candidate_articles
