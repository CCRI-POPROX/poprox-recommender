from poprox_concepts import ArticleSet


class Fill:
    def __init__(self, num_slots):
        self.num_slots = num_slots

    def __call__(self, candidates1: ArticleSet, candidates2: ArticleSet) -> ArticleSet:
        # Fill as many slots as you can from candidates1
        articles = candidates1.articles[: self.num_slots]

        # Fill out remaining slots from candidates2
        articles += candidates2.articles[: self.num_slots - len(articles)]

        return ArticleSet(articles=articles)
