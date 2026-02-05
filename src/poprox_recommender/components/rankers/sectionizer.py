import logging
import random
from uuid import UUID

import numpy as np
from lenskit.pipeline import Component
from pydantic import BaseModel

from poprox_concepts.domain import Article, ArticlePackage, CandidateSet, ImpressedSection, InterestProfile
from poprox_recommender.components.filters import PackageFilter, PackageFilterConfig

logger = logging.getLogger(__name__)


class SectionizerConfig(BaseModel):
    top_news_entity_id: UUID
    max_top_news: int = 3
    max_topic_sections: int = 3
    max_articles_per_topic: int = 3
    max_misc_articles: int = 3


class Sectionizer(Component):
    config: SectionizerConfig

    def __call__(
        self,
        candidate_set: CandidateSet,
        article_packages: list[ArticlePackage],
        interest_profile: InterestProfile,
    ) -> list[ImpressedSection]:
        """
        Build newsletter sections from ranked articles and topic packages.
        """
        if not candidate_set.articles:
            logger.debug("No ranked articles available.")
            return []

        topic_entity_ids = get_top_topics(interest_profile, top_n=self.config.max_topic_sections)
        used_ids = set()
        sections = []

        # top news section
        entity_id = self.config.top_news_entity_id
        package = next((p for p in article_packages if p.seed and p.seed.entity_id == entity_id), None)
        if package:
            package_filter = PackageFilter(config=PackageFilterConfig(package_entity_id=entity_id))
            filtered = package_filter(candidate_set, [package])

            ranked_articles = select_from_candidates(filtered, self.config.max_top_news, list(used_ids))

            used_ids.update(a.article_id for a in ranked_articles)
            top_section = ImpressedSection.from_articles(
                ranked_articles, title=package.title, personalized=True, seed_entity_id=entity_id
            )

            if len(top_section.impressions) > 0:
                sections.append(top_section)

        # topic sections
        for topic_entity_id in topic_entity_ids:
            package = next((p for p in article_packages if p.seed and p.seed.entity_id == topic_entity_id), None)
            if package:
                package_filter = PackageFilter(config=PackageFilterConfig(package_entity_id=topic_entity_id))
                filtered = package_filter(candidate_set, [package])

                ranked_articles = select_from_candidates(filtered, self.config.max_top_news, list(used_ids))

                used_ids.update(a.article_id for a in ranked_articles)
                topic_section = ImpressedSection.from_articles(
                    ranked_articles, title=package.title, personalized=True, seed_entity_id=topic_entity_id
                )

                if len(topic_section.impressions) > 0:
                    sections.append(topic_section)

        # in other news / misc / for you section
        misc_section = self._make_misc_section(candidate_set, used_ids)
        if misc_section:
            sections.append(misc_section)

        logger.debug("Sectionizer created %d total sections", len(sections))
        return sections

    def _make_misc_section(self, candidate_set, used_ids):
        remaining = [a for a in candidate_set.articles if a.article_id not in used_ids]
        if not remaining:
            return None

        if hasattr(candidate_set, "scores") and candidate_set.scores is not None:
            # rank remaining by score
            articles_indices = [
                i for i, a in enumerate(candidate_set.articles) if a.article_id in [r.article_id for r in remaining]
            ]
            scores = np.array(candidate_set.scores)[articles_indices]
            sorted_indices = np.argsort(scores)[::-1][: self.config.max_misc_articles]
            misc_articles = [candidate_set.articles[articles_indices[int(i)]] for i in sorted_indices]
        else:
            misc_articles = remaining[: self.config.max_misc_articles]

        section = ImpressedSection.from_articles(misc_articles, title="In Other News", personalized=True)

        return section


def select_from_candidates(candidates: CandidateSet, num_articles: int, excluding: list[UUID] = None) -> list[Article]:
    excluding = excluding or []

    if hasattr(candidates, "scores") and candidates.scores is not None:
        # rank candidates by score if scores are available
        sorted_indices = np.argsort(np.array(candidates.scores))[::-1]
        ranked_articles = [
            candidates.articles[int(i)]
            for i in sorted_indices
            if candidates.articles[int(i)].article_id not in excluding
        ][:num_articles]
    else:
        # otherwise select from the top of the list of candidates preserving order
        ranked_articles = [a for a in candidates.articles if a.article_id not in excluding][:num_articles]

    return ranked_articles


def get_top_topics(interest_profile: InterestProfile, top_n: int) -> list[UUID]:
    topics = list(interest_profile.interests_by_type("topic"))
    random.shuffle(topics)
    topics_sorted = sorted(
        topics,
        key=lambda i: i.preference,
        reverse=True,
    )
    return [i.entity_id for i in topics_sorted[:top_n]]
