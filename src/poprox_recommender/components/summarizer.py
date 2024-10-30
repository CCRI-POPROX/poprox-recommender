from transformers import pipeline

from poprox_concepts import ArticleSet

summarizer = pipeline("summarization", model="Falconsai/text_summarization")


class Summarizer:
    def __call__(self, article_set: ArticleSet):
        for article in article_set:
            article.subhead = summarizer(article, max_length=1000, min_length=300, do_sample=False)

        return article_set
