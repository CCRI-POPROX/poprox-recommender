from poprox_concepts import ArticleSet


class Interleave:
    def __call__(self, candidates1: ArticleSet, candidates2: ArticleSet) -> ArticleSet:
        articles = [None] * (len(candidates1.articles) + len(candidates2.articles))
        articles[::2] = candidates1.articles
        articles[1::2] = candidates2.articles

        return ArticleSet(articles=articles)
