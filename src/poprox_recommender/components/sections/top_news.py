import logging
from datetime import date

from lenskit.pipeline import Component
from pydantic import BaseModel

from poprox_concepts.domain import ArticlePackage, CandidateSet, ImpressedSection, InterestProfile
from poprox_recommender.components.filters.topic import TopicFilter
from poprox_recommender.components.sections.base import select_from_candidates

logger = logging.getLogger(__name__)


class PersonalizedTopNewsConfig(BaseModel):
    max_articles: int = 3


class PersonalizedTopNews(Component):
    config: PersonalizedTopNewsConfig

    def __call__(
        self,
        candidate_set: CandidateSet,
        article_packages: list[ArticlePackage],
        interest_profile: InterestProfile,
        sections: list[ImpressedSection] | None = None,
        today: date | None = None,
    ) -> list[ImpressedSection]:
        sections = sections or []

        used_ids = set(impression.article.article_id for section in sections for impression in section.impressions)

        topic_filter = TopicFilter()

        # top news section
        top_articles = select_from_packages(candidate_set, article_packages)
        filtered_top = topic_filter(top_articles, interest_profile)

        logger.info(f"Creating Top Stories section from {len(filtered_top.articles)} filtered candidates")
        ranked_articles = select_from_candidates(filtered_top, self.config.max_articles, used_ids)

        if len(ranked_articles) < self.config.max_articles:
            logger.info(f"Falling back to full pool of {len(candidate_set.articles)} top candidates")
            ranked_articles = select_from_candidates(top_articles, self.config.max_articles, used_ids)

        top_section = ImpressedSection.from_articles(ranked_articles, title="Your Top Stories", personalized=True)

        if len(top_section.impressions) > 0:
            used_ids.update(a.article_id for a in ranked_articles)
            sections.append(top_section)

        return sections


def select_from_packages(candidate_articles: CandidateSet, packages: list[ArticlePackage]) -> CandidateSet:
    article_index_lookup = {article.article_id: i for i, article in enumerate(candidate_articles.articles)}
    selected_articles = []
    selected_indices = []

    package_article_ids = set(article_id for package in packages for article_id in package.article_ids)

    for article_id in package_article_ids:
        if article_id in article_index_lookup:
            idx = article_index_lookup[article_id]
            selected_articles.append(candidate_articles.articles[idx])
            selected_indices.append(idx)

    logger.debug(
        "PackageFilter selected %d of %d candidate articles using %s packages",
        len(selected_articles),
        len(candidate_articles.articles),
        ", ".join([p.title for p in packages]),
    )

    filtered = CandidateSet(articles=selected_articles)
    scores = getattr(candidate_articles, "scores", None)
    if scores is not None:
        filtered.scores = [scores[i] for i in selected_indices]
    else:
        filtered.scores = None

    return filtered
