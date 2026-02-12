import hashlib
import logging
from datetime import date
from uuid import UUID

import numpy as np
from lenskit.pipeline import Component
from pydantic import BaseModel

from poprox_concepts.domain import Article, ArticlePackage, CandidateSet, Entity, ImpressedSection, InterestProfile

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

        if today is None:
            today = date.today()

        seed = self.random_daily_seed(interest_profile.profile_id, today, self.config.random_seed)

        used_ids = set()
        sections = []

        # top news section
        filtered = select_from_packages(candidate_set, article_packages)
        ranked_articles = select_from_candidates(filtered, self.config.max_top_news, used_ids)

        used_ids.update(a.article_id for a in ranked_articles)
        top_section = ImpressedSection.from_articles(ranked_articles, title="Your Top Stories", personalized=True)

        if len(top_section.impressions) > 0:
            sections.append(top_section)

        # topic sections
        topical_interests = list(interest_profile.interests_by_type("topic"))
        rng = np.random.default_rng(seed)
        rng.shuffle(topical_interests)
        sorted_interests = sorted(
            topical_interests,
            key=lambda i: i.preference,
            reverse=True,
        )

        topical_sections = []
        for interest in sorted_interests:
            package = next((p for p in article_packages if p.seed and p.seed.entity_id == interest.entity_id), None)
            if package:
                displayed_title = package.title.replace("Top ", "").replace(" Stories", "")
                displayed_title = f"{displayed_title} For You"

                filtered = select_mentioning(candidate_set, [package.seed] if package.seed else [])
                ranked_articles = select_from_candidates(filtered, self.config.max_articles_per_topic, used_ids)

                used_ids.update(a.article_id for a in ranked_articles)
                topic_section = ImpressedSection.from_articles(
                    ranked_articles, title=displayed_title, personalized=True, seed_entity_id=interest.entity_id
                )

                if len(topic_section.impressions) >= self.config.max_articles_per_topic:
                    topical_sections.append(topic_section)

                if len(topical_sections) >= self.config.max_topic_sections:
                    break

        sections.extend(topical_sections)

        # in other news / misc / for you section
        remaining = [a for a in candidate_set.articles if a.article_id not in used_ids]

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

        misc_section = ImpressedSection.from_articles(misc_articles, title="In Other News", personalized=True)

        if len(misc_section.impressions) > 0:
            sections.append(misc_section)

        logger.debug("Sectionizer created %d total sections", len(sections))
        return sections

    def random_daily_seed(self, profile_id, day, base_seed: int) -> int:
        seed_str = f"{profile_id}_{day.isoformat()}_{base_seed}"
        hash_obj = hashlib.sha256(seed_str.encode("utf-8"))
        hash_hex = hash_obj.hexdigest()
        return int(hash_hex, 16)


def select_mentioning(candidate: CandidateSet, entities: list[Entity]):
    entity_ids = set(e.entity_id for e in entities)

    kept_articles = []
    kept_scores = []
    for idx, article in enumerate(candidate.articles):
        mentioned = set(m.entity.entity_id for m in article.mentions if m.relevance and m.relevance >= 76)
        if len(entity_ids.intersection(mentioned)) > 0:
            kept_articles.append(article)
            if hasattr(candidate, "scores") and candidate.scores is not None:
                kept_scores.append(candidate.scores[idx])

    filtered = CandidateSet(articles=kept_articles)
    if kept_scores:
        filtered.scores = np.array(kept_scores)
    else:
        filtered.scores = None

    return filtered


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


def select_from_candidates(candidates: CandidateSet, num_articles: int, excluding: set[UUID] = None) -> list[Article]:
    excluding = excluding or set()

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
