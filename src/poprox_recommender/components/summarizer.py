from transformers import pipeline

from poprox_concepts import ArticleSet
from poprox_recommender.lkpipeline import Component

summarizer = pipeline("summarization", model="Falconsai/text_summarization")


class Summarizer(Component):
    def __call__(self, article_set: ArticleSet):
        for article in article_set.articles:
            summary = summarizer(
                f"{article.headline} {article.subhead}", max_length=200, min_length=100, do_sample=False
            )
            article.subhead = summary[0]["summary_text"]

        return article_set
