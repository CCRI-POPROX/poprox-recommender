from itertools import zip_longest

from lenskit.pipeline import Component
from lenskit.pipeline.types import Lazy
from pydantic import BaseModel

from poprox_concepts import CandidateSet
from poprox_concepts.domain import RecommendationList


class FillConfig(BaseModel):
    num_slots: int
    deduplicate: bool = True


class FillCandidates(Component):
    config: FillConfig

    def __call__(self, recs1: CandidateSet, recs2: Lazy[CandidateSet]) -> RecommendationList:
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
        return CandidateSet(articles=articles[: self.config.num_slots])


class FillRecs(Component):
    config: FillConfig

    def __call__(self, recs1: RecommendationList | CandidateSet, recs2: Lazy[RecommendationList]) -> RecommendationList:
        if isinstance(recs1, CandidateSet):
            recs1 = RecommendationList(articles=recs1.articles)
        articles = recs1.articles
        extras = recs1.extras

        if not extras:
            extras = [{} for _ in articles]

        if self.config.deduplicate:
            # Track the articles by their article_id
            existing_articles = {(article.article_id) for article in articles}

            # Add articles from candidates2 only if they are not duplicates
            if len(articles) < self.config.num_slots:
                new_articles = []
                new_extras = []
                recs2_content = recs2.get()
                for article, extra in zip_longest(recs2_content.articles, recs2_content.extras):
                    # Check if the article is a duplicate based on article_id
                    if (article.article_id) not in existing_articles:
                        new_articles.append(article)
                        new_extras.append(extra or {})
                        existing_articles.add((article.article_id))  # Avoid future duplicates
                    # Stop if we have enough articles
                    if len(articles) + len(new_articles) >= self.config.num_slots:
                        break

                articles = articles + new_articles
                extras = extras + new_extras
        else:
            recs2_content = recs2.get()
            articles = articles + recs2_content.articles
            extras = extras + recs2_content.extras

        # Return the resulting RecommendationList, limiting the size to num_slots
        return RecommendationList(articles=articles[: self.config.num_slots], extras=extras[: self.config.num_slots])
