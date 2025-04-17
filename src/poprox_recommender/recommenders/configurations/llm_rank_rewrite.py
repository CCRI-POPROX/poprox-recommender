# pyright: basic
from lenskit.pipeline import PipelineBuilder

from poprox_concepts import CandidateSet, InterestProfile
from poprox_recommender.components.rankers.openai_ranker import LLMRanker, LLMRankerConfig
from poprox_recommender.components.rewriters.openai_rewriter import LLMRewriter, LLMRewriterConfig


def configure(builder: PipelineBuilder, num_slots: int, device: str):
    # Define inputs
    i_candidates = builder.create_input("candidate", CandidateSet)
    i_profile = builder.create_input("profile", InterestProfile)

    # LLM-based ranking
    rank_cfg = LLMRankerConfig(num_slots=num_slots)
    ranked = builder.add_component(
        "llm-ranker", LLMRanker, rank_cfg, candidate_articles=i_candidates, interest_profile=i_profile
    )

    # LLM-based rewriting
    rewrite_cfg = LLMRewriterConfig()
    builder.add_component("llm-rewriter", LLMRewriter, rewrite_cfg, recommendations=ranked, interest_profile=i_profile)
