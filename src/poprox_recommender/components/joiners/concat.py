from poprox_concepts import ArticleSet


class Concatenate:
    def __call__(self, candidates1: ArticleSet, candidates2: ArticleSet) -> ArticleSet:
        reverse_articles = (candidates1.articles + candidates2.articles)[::-1]
        articles = {article.article_id: article for article in reverse_articles}

        return ArticleSet(articles=list(articles.values())[::-1])
