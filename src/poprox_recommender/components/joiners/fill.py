from lenskit.pipeline import Component
from lenskit.pipeline.types import Lazy
from pydantic import BaseModel

from poprox_concepts import CandidateSet
from poprox_concepts.domain import RecommendationList


class FillConfig(BaseModel):
    num_slots: int
    deduplicate: bool = True


class Fill(Component):
    def __call__(
        self, recs1: CandidateSet | RecommendationList, recs2: Lazy[CandidateSet | RecommendationList]
    ) -> RecommendationList:
        articles = recs1.articles

        if self.config.deduplicate:
            # Track the articles by their article_id
            existing_articles = {(article.article_id) for article in articles}

            # Add articles from candidates2 only if they are not duplicates
            if len(articles) < self.config.num_slots:
                new_articles = []
                for article in recs2.get().articles:
                    # Check if the article is a duplicate based on article_id
                    if (article.article_id) not in existing_articles:
                        new_articles.append(article)
                        existing_articles.add((article.article_id))  # Avoid future duplicates
                    # Stop if we have enough articles
                    if len(articles) + len(new_articles) >= self.config.num_slots:
                        break

                articles = articles + new_articles
        else:
            articles = articles + recs2.get().articles

        # Return the resulting RecommendationList, limiting the size to num_slots
        return RecommendationList(articles=articles[: self.config.num_slots])
