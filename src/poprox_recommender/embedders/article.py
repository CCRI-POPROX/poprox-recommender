import torch as th

from poprox_concepts import Article


class ArticleEmbedder:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, articles: list[Article]) -> th.Tensor:
        tokenized_titles = {}
        for article in articles:
            tokenized_titles[article.article_id] = self.tokenizer.encode(
                article.title, padding="max_length", max_length=30, truncation=True
            )

        articles = {
            "id": list(tokenized_titles.keys()),
            "title": th.tensor(list(tokenized_titles.values())),
        }

        return self.model.get_news_vector(articles["title"].to(self.device))
