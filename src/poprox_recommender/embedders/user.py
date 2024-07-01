import torch as th

from poprox_concepts import ClickHistory, InterestProfile


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


class UserEmbedder:
    def __init__(self, model, device, max_clicks_per_user: int = 50):
        self.model = model
        self.device = device
        self.max_clicks = max_clicks_per_user

    def __call__(self, interest_profile: InterestProfile, clicked_article_embeddings: dict[str, th.Tensor]):
        user_embedding = build_user_embedding(
            interest_profile.click_history,
            clicked_article_embeddings,
            self.model,
            self.device,
            self.max_clicks,
        )

        return user_embedding
