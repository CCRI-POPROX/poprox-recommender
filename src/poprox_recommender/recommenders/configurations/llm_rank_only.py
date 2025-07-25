# pyright: basic

from lenskit.pipeline import PipelineBuilder

from poprox_concepts import CandidateSet, InterestProfile
from poprox_recommender.components.rankers.openai_ranker import LLMRanker, LLMRankerConfig


def configure(builder: PipelineBuilder, num_slots: int, device: str):
    # Define inputs
    i_candidates = builder.create_input("candidate", CandidateSet)
    i_clicked = builder.create_input("clicked", CandidateSet)
    i_profile = builder.create_input("profile", InterestProfile)

    # LLM-based ranking only (no rewriting)
    rank_cfg = LLMRankerConfig(num_slots=num_slots)
    builder.add_component(
        "recommender",
        LLMRanker,
        rank_cfg,
        candidate_articles=i_candidates,
        interest_profile=i_profile,
        articles_clicked=i_clicked,
    )