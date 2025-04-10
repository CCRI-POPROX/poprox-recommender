import logging
from collections import defaultdict

from lenskit.pipeline import Component
from pydantic import BaseModel

from poprox_concepts.domain import CandidateSet

logger = logging.getLogger(__name__)


class ScoreFusionConfig(BaseModel):
    combiner: str = "sum"


class ScoreFusion(Component):
    config: ScoreFusionConfig

    def __call__(self, candidates1: CandidateSet, candidates2: CandidateSet) -> CandidateSet:
        combined_score = defaultdict(float)
        combined_article = {}
        combined_embedding = {}

        for article, score, embeddings in zip(candidates1.articles, candidates1.scores, candidates1.embeddings):
            article_id = article.article_id
            combined_score[article_id] += score
            combined_article[article_id] = article
            combined_embedding[article_id] = embeddings

        for article, score, embeddings in zip(candidates2.articles, candidates2.scores, candidates2.embeddings):
            article_id = article.article_id
            combined_score[article_id] += score
            combined_article[article_id] = article
            combined_embedding[article_id] = embeddings

        if self.config.combiner == "avg":
            denominator = 2
        else:
            denominator = 1

        merged_scores = []
        merged_articles = []
        merged_embeddings = []

        for key, score in combined_score.items():
            merged_articles.append(combined_article[key])
            merged_scores.append(score / denominator)
            merged_embeddings.append(combined_embedding[key])

        logger.info(f"Fused {denominator} candidate sets...")
        return CandidateSet(articles=merged_articles, scores=merged_scores, embeddings=merged_embeddings)
