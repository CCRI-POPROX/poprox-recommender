# pyright: basic

from lenskit.pipeline import PipelineBuilder

from poprox_concepts import CandidateSet, InterestProfile
from poprox_recommender.components.diversifiers.topic_calibration import TopicCalibrator
from poprox_recommender.components.embedders import NRMSArticleEmbedder
from poprox_recommender.components.embedders.article import NRMSArticleEmbedderConfig
from poprox_recommender.components.embedders.article_topic_relvs import NRMSArticleTopicEmbedder
from poprox_recommender.components.embedders.user import NRMSUserEmbedder, NRMSUserEmbedderConfig
from poprox_recommender.components.embedders.user_topic_prefs import (
    PreLearnedCandidateArticleUserTopicEmbedder,
    PreLearnedHybridUserTopicEmbedder,
    PreLearnedStaticDefinitionUserTopicEmbedder,
    UserTopicEmbedderConfig,
)
from poprox_recommender.components.filters.topic import TopicFilter
from poprox_recommender.components.joiners.score import ScoreFusion
from poprox_recommender.components.rankers.topk import TopkRanker
from poprox_recommender.components.scorers.article import ArticleScorer
from poprox_recommender.paths import model_file_path


##TODO:
# allow weigths for the scores (1/-1)
def oracle_topic_scorer(config, candidate_articles, interest_profile):
    with_scores = candidate_articles.model_copy()
    user_topics = {t.entity_name for t in interest_profile.onboarding_topics if t.preference == 5}

    scores = []
    for article in candidate_articles.articles:
        article_topics = {m.entity.name for m in article.mentions}
        score = 1.0 if (user_topics & article_topics) else 0.0
        scores.append(score)

    with_scores.scores = scores

    return with_scores


def configure(builder: PipelineBuilder, num_slots: int, device: str):
    i_candidates = builder.create_input("candidate", CandidateSet)
    i_profile = builder.create_input("profile", InterestProfile)

    n_scorer = builder.add_component(
        "scorer",
        oracle_topic_scorer,
        candidate_articles=i_candidates,
        interest_profile=i_profile,
    )

    builder.add_component(
        "recommender",
        TopkRanker,
        {"num_slots": num_slots},
        candidate_articles=n_scorer,
    )
