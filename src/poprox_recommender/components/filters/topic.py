import logging

from lenskit.pipeline import Component

from poprox_concepts import ArticleSet, InterestProfile

logger = logging.getLogger(__name__)


class TopicFilter(Component):
    def __call__(self, candidate: ArticleSet, interest_profile: InterestProfile) -> ArticleSet:
        # Preference values from onboarding are 1-indexed, where 1 means "absolutely no interest."
        # We might want to normalize them to 0-indexed somewhere upstream, but in the mean time
        # this is one of the simpler ways to filter out topics people aren't interested in from
        # their early newsletters
        profile_topics = {
            interest.entity_name for interest in interest_profile.onboarding_topics if interest.preference > 1
        }

        if len(profile_topics) == 0:
            return candidate

        topical_articles = []
        for article in candidate.articles:
            article_topics = {mention.entity.name for mention in article.mentions}
            if len(profile_topics.intersection(article_topics)) > 0:
                topical_articles.append(article)

        logger.debug(
            "filter accepted %d of %d articles for user %s",
            len(topical_articles),
            len(candidate.articles),
            interest_profile.profile_id,
        )
        return ArticleSet(articles=topical_articles)
