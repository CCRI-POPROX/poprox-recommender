# pyright: basic

import logging
from lenskit.pipeline import PipelineBuilder

from poprox_concepts import CandidateSet, InterestProfile
from poprox_recommender.components.embedders.article import NRMSArticleEmbedder, NRMSArticleEmbedderConfig
from poprox_recommender.components.embedders.user_persona import UserPersonaEmbedder, UserPersonaConfig
from poprox_recommender.components.scorers.persona_scorer import PersonaScorer
from poprox_recommender.components.rankers.topk import TopkRanker
from poprox_recommender.paths import model_file_path

logger = logging.getLogger(__name__)


def configure(builder: PipelineBuilder, num_slots: int, device: str):
    """
    Configure persona-based news recommendation pipeline.
    Compatible with existing POPROX pipeline structure.
    
    Args:
        builder: Pipeline builder
        num_slots: Number of recommendations to return
        device: Device to run on ('cpu' or 'cuda')
    """
    # Define pipeline inputs (extended for persona analysis)
    i_candidates = builder.create_input("candidate", CandidateSet)
    i_clicked = builder.create_input("clicked", CandidateSet)
    i_profile = builder.create_input("profile", InterestProfile)
    # New input for historical newsletter data
    i_historical = builder.create_input("historical_newsletters", CandidateSet, required=False)

    # Embed candidate articles using NRMS embedder (compatible with existing models)
    ae_config = NRMSArticleEmbedderConfig(
        model_path=model_file_path("nrms-mind/news_encoder.safetensors"), 
        device=device
    )
    e_candidates = builder.add_component(
        "candidate-embedder", 
        NRMSArticleEmbedder, 
        ae_config, 
        article_set=i_candidates
    )
    
    e_clicked = builder.add_component(
        "history-embedder",
        NRMSArticleEmbedder,
        ae_config,
        article_set=i_clicked
    )

    # Generate user persona from click history
    persona_config = UserPersonaConfig(
        model_path=model_file_path("nrms-mind/user_encoder.safetensors"),
        device=device,
        llm_api_key="",  # Will be loaded from environment
        max_history_length=50,
        persona_dimensions=128
    )
    
    # Embed historical newsletter articles if provided
    e_historical = None
    if i_historical is not None:
        e_historical = builder.add_component(
            "historical-embedder",
            NRMSArticleEmbedder,
            ae_config,
            article_set=i_historical
        )

    # Generate user persona from engagement and disengagement patterns
    persona_inputs = {
        "candidate_articles": e_candidates,
        "clicked_articles": e_clicked,
        "interest_profile": i_profile
    }
    if e_historical is not None:
        persona_inputs["historical_newsletters"] = e_historical
    
    e_persona = builder.add_component(
        "persona-embedder",
        UserPersonaEmbedder,
        persona_config,
        **persona_inputs
    )

    # Score articles based on persona similarity
    persona_scores = builder.add_component(
        "persona-scorer",
        PersonaScorer,
        {"alpha": 0.7, "beta": 0.3},
        candidate_articles=e_candidates,
        user_persona=e_persona,
        interest_profile=i_profile
    )

    # Rank and select top articles
    builder.add_component(
        "recommender", 
        TopkRanker, 
        {"num_slots": num_slots}, 
        candidate_articles=persona_scores
    )