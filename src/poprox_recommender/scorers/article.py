from poprox_concepts import ArticleSet, InterestProfile


class ArticleScorer:
    def __init__(self, model):
        self.model = model

    def __call__(self, candidate_articles: ArticleSet, interest_profile: InterestProfile) -> ArticleSet:
        candidate_embeddings = candidate_articles.embeddings
        user_embedding = interest_profile.embedding

        pred = self.model.get_prediction(candidate_embeddings, user_embedding.squeeze())
        return candidate_articles.model_copy(update={"scores": pred.cpu().detach().numpy()})
