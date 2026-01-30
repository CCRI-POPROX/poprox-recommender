import hashlib
import logging
from datetime import date
from uuid import UUID

import numpy as np
from lenskit.pipeline import Component
from pydantic import BaseModel

from poprox_concepts.domain import ArticlePackage, CandidateSet, ImpressedSection, InterestProfile
from poprox_recommender.components.filters import PackageFilter, PackageFilterConfig

logger = logging.getLogger(__name__)


class SectionizerConfig(BaseModel):
    top_news_entity_id: UUID
    max_top_news: int = 3
    max_topic_sections: int = 3
    max_articles_per_topic: int = 3
    max_misc_articles: int = 3
    add_section_metadata: bool = True
    random_seed: int = 22


class Sectionizer(Component):
    config: SectionizerConfig

    def __call__(
        self,
        candidate_set: CandidateSet,
        article_packages: list[ArticlePackage],
        interest_profile: InterestProfile,
        today: date = date.today(),
    ) -> list[ImpressedSection]:
        """
        Build newsletter sections from ranked articles and topic packages.
        """
        if not candidate_set.articles:
            logger.debug("No ranked articles available.")
            return []

        seed = self.random_daily_seed(interest_profile.profile_id, today, self.config.random_seed)

        topic_entity_ids = get_top_topics(interest_profile, top_n=self.config.max_topic_sections, seed=seed)
        used_ids = set()
        sections = []

        # top news section
        top_section = self._make_section(
            candidate_set,
            article_packages,
            entity_id=self.config.top_news_entity_id,
            max_articles=self.config.max_top_news,
            used_ids=used_ids,
        )
        if top_section:
            sections.append(top_section)
        # topic sections
        for topic_entity_id in topic_entity_ids:
            section = self._make_section(
                candidate_set,
                article_packages,
                entity_id=topic_entity_id,
                max_articles=self.config.max_articles_per_topic,
                used_ids=used_ids,
            )
            if section:
                sections.append(section)

        # in other news / misc / for you section
        misc_section = self._make_misc_section(candidate_set, used_ids)
        if misc_section:
            sections.append(misc_section)

        logger.debug("Sectionizer created %d total sections", len(sections))
        return sections

    def random_daily_seed(self, profile_id, day, base_seed: int) -> int:
        seed_str = f"{profile_id}_{day.isoformat()}_{base_seed}"
        hash_obj = hashlib.sha256(seed_str.encode("utf-8"))
        hash_hex = hash_obj.hexdigest()
        return int(hash_hex, 16)

    def _make_section(self, candidate_set, packages, entity_id, max_articles, used_ids):
        package = next((p for p in packages if p.seed and p.seed.entity_id == entity_id), None)
        if not package:
            logger.debug("No package found for entity_id '%s'", entity_id)
            return None

        package_filter = PackageFilter(config=PackageFilterConfig(package_entity_id=entity_id))
        filtered = package_filter(candidate_set, [package])

        # rank filtered candidates
        if hasattr(candidate_set, "scores") and candidate_set.scores is not None:
            article_ids = [a.article_id for a in filtered.articles if a.article_id not in used_ids]
            article_indices = [i for i, a in enumerate(candidate_set.articles) if a.article_id in article_ids]

            scores = np.array(candidate_set.scores)[article_indices]
            sorted_indices = np.argsort(scores)[::-1][:max_articles]

            ranked_articles = [candidate_set.articles[article_indices[int(i)]] for i in sorted_indices]
        else:
            # preserve order if no scores
            ranked_articles = [a for a in filtered.articles if a.article_id not in used_ids][:max_articles]
        if not ranked_articles:
            return None

        used_ids.update(a.article_id for a in ranked_articles)
        section = ImpressedSection.from_articles(ranked_articles)

        if self.config.add_section_metadata:
            section.title = package.title
            section.personalized = True
            section.seed_entity_id = entity_id

        return section

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

        section = ImpressedSection.from_articles(misc_articles)

        if self.config.add_section_metadata:
            section.title = "In Other News"
            section.personalized = True

        return section


def get_top_topics(interest_profile: InterestProfile, top_n: int, seed: int) -> list[UUID]:
    topics = list(interest_profile.interests_by_type("topic"))
    rng = np.random.default_rng(seed)
    topics_sorted = sorted(
        topics,
        key=lambda t: (t.preference, rng.random()),
        reverse=True,
    )
    return [i.entity_id for i in topics_sorted[:top_n]]
