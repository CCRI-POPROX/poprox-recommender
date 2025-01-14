from itertools import zip_longest

from lenskit.pipeline import Component

from poprox_concepts import ArticleSet


class Interleave(Component):
    def __call__(self, candidates1: ArticleSet, candidates2: ArticleSet) -> ArticleSet:
        articles = []
        for pair in zip_longest(candidates1.articles, candidates2.articles):
            for article in pair:
                if article is not None:
                    articles.append(article)

        return ArticleSet(articles=articles)
