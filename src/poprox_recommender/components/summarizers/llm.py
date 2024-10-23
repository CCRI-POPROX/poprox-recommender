import numpy as np

from poprox_concepts import ArticleSet
from poprox_recommender.lkpipeline import Component


class Summarizer(Component):
    def __init__(self):
        pass

    def __call__(self, candidates: ArticleSet) -> ArticleSet:
        summary = "This is an article summary"

        for article in candidates.articles:
            article.subhead = summary

        return candidates
