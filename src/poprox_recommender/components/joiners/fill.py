from lenskit.pipeline import Component
from lenskit.pipeline.types import Lazy
from pydantic import BaseModel

from poprox_concepts.domain import ImpressedSection, Impression


class FillConfig(BaseModel):
    num_slots: int
    deduplicate: bool = True


class FillRecs(Component):
    config: FillConfig

    def __call__(self, recs1: ImpressedSection, recs2: Lazy[ImpressedSection]) -> ImpressedSection:
        combined: list[Impression] = []
        combined.extend(recs1.impressions)

        if self.config.deduplicate:
            # Track the articles by their article_id
            existing_articles = {(impression.article.article_id) for impression in recs1.impressions}

            # Add articles from candidates2 only if they are not duplicates
            if len(recs1.impressions) < self.config.num_slots:
                new_impressions: list[Impression] = []
                recs2_content = recs2.get()
                for impression in recs2_content.impressions:
                    # Check if the article is a duplicate based on article_id
                    if (impression.article.article_id) not in existing_articles:
                        new_impressions.append(impression)
                        existing_articles.add((impression.article.article_id))  # Avoid future duplicates
                    # Stop if we have enough articles
                    if len(recs1.impressions) + len(new_impressions) >= self.config.num_slots:
                        break
                combined.extend(new_impressions)
        else:
            recs2_content = recs2.get()

            combined.extend(recs2_content.impressions)

        # Return the resulting ImpressedSection, limiting the size to num_slots
        return ImpressedSection(impressions=combined[: self.config.num_slots])
