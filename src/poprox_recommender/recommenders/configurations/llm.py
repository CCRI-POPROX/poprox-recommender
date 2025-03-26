# src/poprox_recommender/recommenders/configurations/llm.py
from lenskit.pipeline import PipelineBuilder

from poprox_concepts import CandidateSet, InterestProfile
from poprox_recommender.components.filters.topic import TopicFilter
from poprox_recommender.components.joiners.fill import FillRecs
from poprox_recommender.components.rankers.topk import TopkRanker
from poprox_recommender.components.samplers.uniform import UniformSampler

# Import our new LLM scorer
from poprox_recommender.components.scorers.llm_scorer import LLMScorer, LLMScorerConfig


def configure(builder: PipelineBuilder, num_slots: int, device: str):
    """Configure the LLM-based recommender pipeline."""

    # Define pipeline inputs
    i_candidates = builder.create_input("candidate", CandidateSet)
    i_clicked = builder.create_input("clicked", CandidateSet)
    i_profile = builder.create_input("profile", InterestProfile)

    # Configure the LLM scorer
    llm_config = LLMScorerConfig(
        model="gpt-4o",
        temperature=0.7,
        # The API key should be set as an environment variable
        # or provided during deployment
    )

    # Score candidates using LLM
    n_scorer = builder.add_component(
        "scorer",
        LLMScorer,
        llm_config,
        candidate_articles=i_candidates,
        clicked_articles=i_clicked,
        interest_profile=i_profile,
    )

    # Rank top articles based on LLM scores
    n_ranker = builder.add_component("ranker", TopkRanker, {"num_slots": num_slots}, candidate_articles=n_scorer)

    # Fallback: sample from user topic interests if LLM scoring fails
    n_topic_filter = builder.add_component(
        "topic-filter", TopicFilter, candidate=i_candidates, interest_profile=i_profile
    )

    n_sampler = builder.add_component("sampler", UniformSampler, candidates1=n_topic_filter, candidates2=i_candidates)

    # Combine primary ranker and fallback
    builder.add_component("recommender", FillRecs, {"num_slots": num_slots}, recs1=n_ranker, recs2=n_sampler)
