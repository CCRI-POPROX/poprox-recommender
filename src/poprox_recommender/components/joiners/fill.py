from lenskit.pipeline import Component
from lenskit.pipeline.types import Lazy

from poprox_concepts import ArticleSet


class Fill(Component):
    def __init__(self, num_slots: int, deduplicate: bool = True):
        self.num_slots = num_slots
        self.deduplicate = deduplicate

    def __call__(self, candidates1: ArticleSet, candidates2: Lazy[ArticleSet]) -> ArticleSet:
        articles = candidates1.articles

        if self.deduplicate:
            # Track the articles by their article_id
            existing_articles = {(article.article_id) for article in articles}

            # Add articles from candidates2 only if they are not duplicates
            if len(articles) < self.num_slots:
                new_articles = []
                for article in candidates2.get().articles:
                    # Check if the article is a duplicate based on article_id
                    if (article.article_id) not in existing_articles:
                        new_articles.append(article)
                        existing_articles.add((article.article_id))  # Avoid future duplicates
                    # Stop if we have enough articles
                    if len(articles) + len(new_articles) >= self.num_slots:
                        break

                articles = articles + new_articles
        else:
            articles = articles + candidates2.get().articles

        # Return the resulting ArticleSet, limiting the size to num_slots
        return ArticleSet(articles=articles[: self.num_slots])
