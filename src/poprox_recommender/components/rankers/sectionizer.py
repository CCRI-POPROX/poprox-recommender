import hashlib
import logging
from datetime import date

import numpy as np
from lenskit.pipeline import Component
from pydantic import BaseModel

from poprox_concepts.domain import ArticlePackage, CandidateSet, ImpressedSection, InterestProfile
from poprox_recommender.components.filters.topic import TopicFilter
from poprox_recommender.components.sections.base import select_from_candidates, select_mentioning
from poprox_recommender.components.sections.top_news import PersonalizedTopNews, PersonalizedTopNewsConfig

logger = logging.getLogger(__name__)


class TopicalSectionsConfig(BaseModel):
    max_topic_sections: int = 3
    max_articles_per_topic: int = 3
    random_seed: int = 22


class TopicalSections(Component):
    config: TopicalSectionsConfig

    def __call__(
        self,
        candidate_set: CandidateSet,
        article_packages: list[ArticlePackage],
        interest_profile: InterestProfile,
        sections: list[ImpressedSection] | None = None,
        today: date | None = None,
    ) -> list[ImpressedSection]:
        sections = sections or []

        if today is None:
            today = date.today()

        seed = self.random_daily_seed(interest_profile.profile_id, today, self.config.random_seed)

        used_ids = set(impression.article.article_id for section in sections for impression in section.impressions)

        topical_interests = list(interest_profile.interests_by_type("topic"))
        rng = np.random.default_rng(seed)
        rng.shuffle(topical_interests)
        sorted_interests = sorted(
            topical_interests,
            key=lambda i: i.preference,
            reverse=True,
        )

        topical_sections: list[ImpressedSection] = []
        for interest in sorted_interests:
            package = next((p for p in article_packages if p.seed and p.seed.entity_id == interest.entity_id), None)
            if package:
                displayed_title = package.title.replace("Top ", "").replace(" Stories", "")

                filtered = select_mentioning(candidate_set, [package.seed] if package.seed else [])
                logger.info(f"Creating {package.seed.name} section from {len(filtered.articles)} topical candidates")
                ranked_articles = select_from_candidates(filtered, self.config.max_articles_per_topic, used_ids)

                topic_section = ImpressedSection.from_articles(
                    ranked_articles, title=displayed_title, personalized=True, seed_entity_id=interest.entity_id
                )

                if len(topic_section.impressions) >= self.config.max_articles_per_topic:
                    used_ids.update(a.article_id for a in ranked_articles)
                    topical_sections.append(topic_section)
                else:
                    logger.info(
                        f"Discarding partial {package.seed.name} section with {len(topic_section.impressions)} articles"
                    )

                if len(topical_sections) >= self.config.max_topic_sections:
                    break

        sections.extend(topical_sections)

        return sections

    def random_daily_seed(self, profile_id, day, base_seed: int) -> int:
        seed_str = f"{profile_id}_{day.isoformat()}_{base_seed}"
        hash_obj = hashlib.sha256(seed_str.encode("utf-8"))
        hash_hex = hash_obj.hexdigest()
        return int(hash_hex, 16)


class InOtherNewsConfig(BaseModel):
    max_articles: int = 3


class InOtherNews(Component):
    config: InOtherNewsConfig

    def __call__(
        self,
        candidate_set: CandidateSet,
        article_packages: list[ArticlePackage],
        interest_profile: InterestProfile,
        sections: list[ImpressedSection] | None = None,
    ) -> list[ImpressedSection]:
        sections = sections or []

        topic_filter = TopicFilter()

        used_ids = set(impression.article.article_id for section in sections for impression in section.impressions)
        topic_seeds = [
            package.seed
            for package in article_packages
            for section in sections
            if section.seed_entity_id == package.seed.entity_id
        ]

        used_topic_articles = select_mentioning(candidate_set, topic_seeds)
        for article in used_topic_articles.articles:
            used_ids.add(article.article_id)

        topic_filtered = topic_filter(candidate_set, interest_profile)
        logger.info(f"Creating Other News section from {len(topic_filtered.articles)} filtered candidates")
        ranked_articles = select_from_candidates(topic_filtered, self.config.max_articles, used_ids)
        if len(ranked_articles) < self.config.max_articles:
            logger.info(f"Falling back to full pool of {len(candidate_set.articles)} candidates")
            ranked_articles = select_from_candidates(candidate_set, self.config.max_articles, used_ids)

        misc_section = ImpressedSection.from_articles(ranked_articles, title="In Other News", personalized=True)

        if len(misc_section.impressions) > 0:
            sections.append(misc_section)

        return sections


class SectionizerConfig(BaseModel):
    max_top_news: int = 3
    max_topic_sections: int = 3
    max_articles_per_topic: int = 3
    max_misc_articles: int = 3
    random_seed: int = 22


class Sectionizer(Component):
    config: SectionizerConfig

    def __call__(
        self,
        candidate_set: CandidateSet,
        article_packages: list[ArticlePackage],
        interest_profile: InterestProfile,
        today: date | None = None,
    ) -> list[ImpressedSection]:
        """
        Build newsletter sections from ranked articles and topic packages.
        """
        if not candidate_set.articles:
            logger.debug("No ranked articles available.")
            return []

        newsletter_sections = []

        top_news_config = PersonalizedTopNewsConfig(max_articles=self.config.max_top_news)
        newsletter_sections = PersonalizedTopNews(top_news_config).__call__(
            candidate_set, article_packages, interest_profile, newsletter_sections
        )

        topical_config = TopicalSectionsConfig(
            max_topic_sections=self.config.max_topic_sections,
            max_articles_per_topic=self.config.max_articles_per_topic,
            random_seed=self.config.random_seed,
        )
        newsletter_sections = TopicalSections(topical_config).__call__(
            candidate_set, article_packages, interest_profile, newsletter_sections
        )

        other_news_config = InOtherNewsConfig(max_articles=self.config.max_misc_articles)
        newsletter_sections = InOtherNews(other_news_config).__call__(
            candidate_set,
            article_packages,
            interest_profile,
            newsletter_sections,
        )

        logger.debug("Sectionizer created %d total sections", len(newsletter_sections))
        return newsletter_sections
