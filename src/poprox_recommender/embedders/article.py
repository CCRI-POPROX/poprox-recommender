import torch as th

from poprox_concepts import Article


class ArticleEmbedder:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, articles: list[Article]):
        article_features = transform_article_features(articles, self.tokenizer)
        article_lookup, article_tensor = build_article_embeddings(article_features, self.model, self.device)

        return article_lookup, article_tensor


def transform_article_features(articles: list[Article], tokenizer) -> dict[str, list]:
    tokenized_titles = {}
    for article in articles:
        tokenized_titles[article.article_id] = tokenizer.encode(
            article.title, padding="max_length", max_length=30, truncation=True
        )

    return tokenized_titles


# Compute a vector for each news story
def build_article_embeddings(article_features, model, device) -> tuple[dict[str, th.Tensor], th.Tensor]:
    articles = {
        "id": list(article_features.keys()),
        "title": th.tensor(list(article_features.values())),
    }
    article_embeddings = {}
    article_vectors = model.get_news_vector(articles["title"].to(device))
    for article_id, article_vector in zip(articles["id"], article_vectors, strict=False):
        if article_id not in article_embeddings:
            article_embeddings[article_id] = article_vector

    article_embeddings["PADDED_NEWS"] = th.zeros(list(article_embeddings.values())[0].size(), device=device)
    return article_embeddings, article_vectors
