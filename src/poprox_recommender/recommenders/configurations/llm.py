# src/poprox_recommender/recommenders/configurations/llm.py
from lenskit.pipeline import PipelineBuilder

from poprox_concepts import CandidateSet, InterestProfile
from poprox_recommender.components.filters.topic import TopicFilter
from poprox_recommender.components.joiners.fill import FillRecs
from poprox_recommender.components.rankers.topk import TopkRanker
from poprox_recommender.components.samplers.uniform import UniformSampler

# Import your custom LLM scorer
from poprox_recommender.components.scorers.llm import LLMScorer


def configure(builder: PipelineBuilder, num_slots: int, device: str):
    # Define pipeline inputs
    i_candidates = builder.create_input("candidate", CandidateSet)
    i_clicked = builder.create_input("clicked", CandidateSet)
    i_profile = builder.create_input("profile", InterestProfile)

    # Score using LLM
    n_scorer = builder.add_component(
        "scorer", LLMScorer, candidate_articles=i_candidates, clicked_articles=i_clicked, interest_profile=i_profile
    )

    # Rank top articles based on scores
    n_ranker = builder.add_component("ranker", TopkRanker, {"num_slots": num_slots}, candidate_articles=n_scorer)

    # Fallback: sample from user topic interests
    n_topic_filter = builder.add_component(
        "topic-filter", TopicFilter, candidate=i_candidates, interest_profile=i_profile
    )
    n_sampler = builder.add_component("sampler", UniformSampler, candidates1=n_topic_filter, candidates2=i_candidates)

    # Combine primary ranker and fallback
    builder.add_component("recommender", FillRecs, {"num_slots": num_slots}, recs1=n_ranker, recs2=n_sampler)
