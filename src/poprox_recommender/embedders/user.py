import torch as th

from poprox_concepts import ArticleSet, ClickHistory, InterestProfile
from poprox_recommender.torch.decorators import torch_inference


class UserEmbedder:
    def __init__(self, model, device, max_clicks_per_user: int = 50):
        self.model = model
        self.device = device
        self.max_clicks = max_clicks_per_user

    @torch_inference
    def __call__(self, clicked_articles: ArticleSet, interest_profile: InterestProfile) -> InterestProfile:
        embedding_lookup = {}
        for article, article_vector in zip(clicked_articles.articles, clicked_articles.embeddings, strict=True):
            if article.article_id not in embedding_lookup:
                embedding_lookup[article.article_id] = article_vector

        embedding_lookup["PADDED_NEWS"] = th.zeros(list(embedding_lookup.values())[0].size(), device=self.device)

        interest_profile.embedding = build_user_embedding(
            interest_profile.click_history,
            embedding_lookup,
            self.model,
            self.device,
            self.max_clicks,
        )

        return interest_profile


# Compute a vector for each user
def build_user_embedding(click_history: ClickHistory, article_embeddings, model, device, max_clicks_per_user):
    article_ids = list(dict.fromkeys(click_history.article_ids))[
        -max_clicks_per_user:
    ]  # deduplicate while maintaining order

    padded_positions = max_clicks_per_user - len(article_ids)
    assert padded_positions >= 0

    article_ids = ["PADDED_NEWS"] * padded_positions + article_ids
    default = article_embeddings["PADDED_NEWS"]
    clicked_article_embeddings = [
        article_embeddings.get(clicked_article, default).to(device) for clicked_article in article_ids
    ]
    clicked_news_vector = (
        th.stack(
            clicked_article_embeddings,
            dim=0,
        )
        .unsqueeze(0)
        .to(device)
    )

    return model.get_user_vector(clicked_news_vector)
