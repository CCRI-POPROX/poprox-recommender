from poprox_concepts import ArticleSet


class Fill:
    def __init__(self, num_slots):
        self.num_slots = num_slots

    def __call__(self, candidates1: ArticleSet, candidates2: ArticleSet) -> ArticleSet:
        articles = candidates1.articles + candidates2.articles
        return ArticleSet(articles=articles[: self.num_slots])
