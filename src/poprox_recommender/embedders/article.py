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

        title_tensor = th.tensor(list(tokenized_titles.values())).to(self.device)
        if len(title_tensor.shape) == 1:
            title_tensor = title_tensor.unsqueeze(dim=0)

        return self.model.get_news_vector(title_tensor)
