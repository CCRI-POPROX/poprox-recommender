import logging

from lenskit.pipeline import Component

from poprox_concepts import CandidateSet, InterestProfile

logger = logging.getLogger(__name__)


class TopicFilter(Component):
    config: None

    def __call__(self, candidates: CandidateSet, interest_profile: InterestProfile) -> CandidateSet:
        # Preference values from onboarding are 1-indexed, where 1 means "absolutely no interest."
        # We might want to normalize them to 0-indexed somewhere upstream, but in the mean time
        # this is one of the simpler ways to filter out topics people aren't interested in from
        # their early newsletters
        interests = {interest.entity_name: interest.preference for interest in interest_profile.onboarding_topics}

        very_high = {key for key, value in interests.items() if value == 5}
        high = {key for key, value in interests.items() if value == 4}
        low = {key for key, value in interests.items() if value == 2}
        very_low = {key for key, value in interests.items() if value == 1}

        topical_articles = []
        for article in candidates.articles:
            article_topics = {mention.entity.name for mention in article.mentions}

            # Articles with very high interest topics are included in the candidate set
            if overlap(article_topics, very_high):
                topical_articles.append(article)
                continue

            # Remaining articles with a very low interest are excluded from the candidate set
            if overlap(article_topics, very_low):
                continue

            # In the middle, exclude articles with more low than high interest topics
            if overlap(article_topics, high) < overlap(article_topics, low):
                continue

            # If none of the above apply, default to including the article
            topical_articles.append(article)

        logger.debug(
            "topic filter accepted %d of %d articles for user %s",
            len(topical_articles),
            len(candidates.articles),
            interest_profile.profile_id,
        )

        return CandidateSet(articles=topical_articles)


def overlap(a: set, b: set):
    return len(a.intersection(b))
