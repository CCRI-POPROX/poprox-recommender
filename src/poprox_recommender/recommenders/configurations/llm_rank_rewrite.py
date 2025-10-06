# pyright: basic

from lenskit.pipeline import PipelineBuilder

from poprox_concepts import CandidateSet, InterestProfile

# Remove RecommenderInfo import
from poprox_recommender.components.rankers.openai_ranker import LLMRanker, LLMRankerConfig
from poprox_recommender.components.rewriters.openai_rewriter import LLMRewriter, LLMRewriterConfig


def configure(builder: PipelineBuilder, num_slots: int, device: str):
    # Define inputs
    i_candidates = builder.create_input("candidate", CandidateSet)
    i_clicked = builder.create_input("clicked", CandidateSet)
    i_profile = builder.create_input("profile", InterestProfile)

    # LLM-based ranking
    rank_cfg = LLMRankerConfig(num_slots=num_slots)
    ranker_output = builder.add_component(
        "ranker",
        LLMRanker,
        rank_cfg,
        candidate_articles=i_candidates,
        interest_profile=i_profile,
        articles_clicked=i_clicked,
    )

    # LLM-based rewriting
    rewrite_cfg = LLMRewriterConfig(pipeline_name="llm_rank_rewrite")
    builder.add_component(
        "recommender",
        LLMRewriter,
        rewrite_cfg,
        ranker_output=ranker_output,
    )

    # Remove RecommenderInfo usage
    # Get the git SHA from environment or use package version as fallback
    # git_sha = os.environ.get("GIT_SHA", version("poprox-recommender"))
    # builder.add_metadata(RecommenderInfo(name="llm-rank-rewrite", version=git_sha))
