from datetime import datetime, timezone
from uuid import uuid4

import torch as th

from poprox_concepts.domain import AccountInterest, Article, ArticleSet, Click, InterestProfile
from poprox_recommender.components.embedders.topic_wise_user import TopicUserEmbedder, topic_articles
from poprox_recommender.paths import model_file_path


def test_embed_user():
    embedder = TopicUserEmbedder(model_file_path("nrms-mind/user_encoder.safetensors"), "cpu")

    interests = [
        AccountInterest(
            entity_id=topic_article.article_id, entity_name=topic_article.external_id, preference=1, frequency=None
        )
        for topic_article in topic_articles
    ]

    clicked = ArticleSet(
        articles=[
            Article(
                article_id=uuid4(),
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

    profile = InterestProfile(click_history=[Click(article_id=uuid4())], onboarding_topics=interests)
    topics = ArticleSet(articles=topic_articles)

    initial_clicks = len(profile.click_history)

    enriched_profile = embedder(clicked=clicked, interest_profile=profile, topics=topics)

    assert len(enriched_profile.click_history) > initial_clicks
    assert len(enriched_profile.click_history) == 15
