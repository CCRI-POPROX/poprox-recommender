# pyright: basic

from lenskit.pipeline import Component, PipelineBuilder

from poprox_concepts import CandidateSet, InterestProfile
from poprox_concepts.domain import RecommendationList
from poprox_recommender.components.rankers.openai_ranker import LLMRanker, LLMRankerConfig


class LLMRankOnlyWrapper(Component):
    """Wrapper component that extracts just the RecommendationList from LLMRanker's tuple output."""

    def __call__(self, ranker_output: tuple[RecommendationList, str, str, dict, dict]) -> RecommendationList:
        recommendations, _, _, _, _ = ranker_output
        return recommendations


def configure(builder: PipelineBuilder, num_slots: int, device: str):
    # Define inputs
    i_candidates = builder.create_input("candidate", CandidateSet)
    i_clicked = builder.create_input("clicked", CandidateSet)
    i_profile = builder.create_input("profile", InterestProfile)

    # LLM-based ranking only (no rewriting)
    rank_cfg = LLMRankerConfig(num_slots=num_slots)
    ranker_output = builder.add_component(
        "ranker",
        LLMRanker,
        rank_cfg,
        candidate_articles=i_candidates,
        interest_profile=i_profile,
        articles_clicked=i_clicked,
    )
    
    # Wrapper to extract just the RecommendationList
    builder.add_component(
        "recommender",
        LLMRankOnlyWrapper,
        ranker_output=ranker_output,
    )
