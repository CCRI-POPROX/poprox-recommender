from datetime import datetime, timezone
from uuid import uuid4

import torch as th
from safetensors.torch import save_file

from poprox_concepts import Article, CandidateSet
from poprox_recommender.components.embedders import NRMSArticleEmbedder
from poprox_recommender.components.topical_description_llm import TOPIC_DESCRIPTIONS
from poprox_recommender.paths import model_file_path

TOPIC_ARTICLES = [
    Article(
        article_id=uuid4(),
        headline=description,
        subhead=None,
        url=None,
        preview_image_id=None,
        published_at=datetime.now(timezone.utc),  # Set current time for simplicity
        mentions=[],
        source="topic",
        external_id=topic,
        raw_data={},
    )
    for topic, description in TOPIC_DESCRIPTIONS.items()
]

# Compute embeddings for the topics
article_embedder = NRMSArticleEmbedder(model_path=model_file_path("nrms-mind/news_encoder.safetensors"), device="cpu")
topic_article_set = article_embedder(CandidateSet(articles=TOPIC_ARTICLES))

topic_embeddings_by_name: dict[str, th.Tensor] = {
    (article.external_id or ""): embedding for article, embedding in zip(TOPIC_ARTICLES, topic_article_set.embeddings)
}

# breakpoint()

# Write them to a safetensors file
save_file(topic_embeddings_by_name, "topic_embeddings_def_llm.safetensors")
