import logging
from datetime import date

from lenskit.pipeline import Component
from pydantic import BaseModel

from poprox_concepts.domain import ArticlePackage, CandidateSet, ImpressedSection, InterestProfile
from poprox_recommender.components.sections.other_news import InOtherNews, InOtherNewsConfig
from poprox_recommender.components.sections.top_news import PersonalizedTopNews, PersonalizedTopNewsConfig
from poprox_recommender.components.sections.topical import TopicalSections, TopicalSectionsConfig

logger = logging.getLogger(__name__)


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
