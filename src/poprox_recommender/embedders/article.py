import torch as th

from poprox_concepts import ArticleSet


class ArticleEmbedder:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, article_set: ArticleSet) -> ArticleSet:
        tokenized_titles = {}
        for article in article_set.articles:
            tokenized_titles[article.article_id] = self.tokenizer.encode(
                article.title, padding="max_length", max_length=30, truncation=True
            )

        title_tensor = th.tensor(list(tokenized_titles.values())).to(self.device)
        if len(title_tensor.shape) == 1:
            title_tensor = title_tensor.unsqueeze(dim=0)

        article_embeddings = self.model.get_news_vector(title_tensor)

        return article_set.model_copy(update={"embeddings": article_embeddings})
