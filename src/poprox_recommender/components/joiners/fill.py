from poprox_concepts import ArticleSet
from poprox_recommender.lkpipeline import Component
from poprox_recommender.lkpipeline.types import Lazy


class Fill(Component):
    def __init__(self, num_slots):
        self.num_slots = num_slots

    def __call__(self, candidates1: ArticleSet, candidates2: Lazy[ArticleSet]) -> ArticleSet:
        articles = candidates1.articles
        if len(articles) < self.num_slots:
            articles = articles + candidates2.get().articles

        return ArticleSet(articles=articles[: self.num_slots])
