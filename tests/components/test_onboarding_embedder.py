from copy import deepcopy
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pytest
import torch as th

from poprox_concepts.domain import AccountInterest, Article, CandidateSet, Click, InterestProfile
from poprox_recommender.components.embedders import NRMSUserEmbedder
from poprox_recommender.components.embedders.user_topic_prefs import TOPIC_ARTICLES, UserOnboardingEmbedder
from poprox_recommender.paths import model_file_path


def test_embed_user():
    try:
        plain_nrms_embedder = NRMSUserEmbedder(
            model_path=model_file_path("nrms-mind/user_encoder.safetensors"), device="cpu"
        )
        topic_aware_embedder = UserOnboardingEmbedder(
            model_path=model_file_path("nrms-mind/user_encoder.safetensors"), device="cpu"
        )
    except FileNotFoundError:
        pytest.xfail("NRMS model not found, so test failure is expected")

    interests = [
        AccountInterest(
            entity_id=topic_article.article_id, entity_name=topic_article.external_id, preference=2, frequency=None
        )
        for topic_article in TOPIC_ARTICLES
    ]

    article_id = uuid4()

    clicked = CandidateSet(
        articles=[
            Article(
                article_id=article_id,
                headline="",
                subhead=None,
                url=None,
                preview_image_id=None,
                published_at=datetime.now(timezone.utc),  # Set current time for simplicity
                mentions=[],
                source="",
                external_id="",
                raw_data={},
            )
        ],
        embeddings=th.rand(1, 768),
    )

    profile = InterestProfile(click_history=[Click(article_id=article_id)], onboarding_topics=interests)

    initial_clicks = len(profile.click_history)

    enriched_profile = deepcopy(profile)

    enriched_profile = topic_aware_embedder(
        clicked_articles=clicked, candidate_articles=clicked, interest_profile=enriched_profile
    )

    assert len(enriched_profile.click_history) > initial_clicks
    assert len(enriched_profile.click_history) == len(TOPIC_ARTICLES)

    plain_profile = deepcopy(profile)

    plain_profile = plain_nrms_embedder(clicked_articles=clicked, interest_profile=plain_profile)

    assert not np.allclose(enriched_profile.embedding, plain_profile.embedding)
