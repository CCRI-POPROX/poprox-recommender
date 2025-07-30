import logging
import os
from typing import Any, NamedTuple
from uuid import UUID

import numpy as np
import pandas as pd
from lenskit.data import ItemList
from lenskit.metrics import call_metric
from lenskit.metrics.ranking import NDCG, RecipRank

from poprox_concepts import Article, CandidateSet
from poprox_recommender.data.eval import EvalData
from poprox_recommender.evaluation.metrics.ils import intralist_similarity
from poprox_recommender.evaluation.metrics.lip import least_item_promoted
from poprox_recommender.evaluation.metrics.rbe import rank_bias_entropy
from poprox_recommender.evaluation.metrics.rbo import rank_biased_overlap

# Global cache for embeddings
_embeddings_cache = None


def load_embeddings_cache():
    """Load embeddings from parquet file"""
    global _embeddings_cache
    if _embeddings_cache is None:
        try:
            possible_paths = [
                "outputs/mind-subset/nrms_topic_scores/embeddings.parquet",  # This one works
                "outputs/mind-subset/nrms_topic_mmr/embeddings.parquet",
                "outputs/mind-subset/embeddings.parquet",
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    df = pd.read_parquet(path)
                    _embeddings_cache = {row["article_id"]: np.array(row["embedding"]) for _, row in df.iterrows()}
                    logging.getLogger(__name__).info(f"Loaded {len(_embeddings_cache)} embeddings from {path}")
                    break
            else:
                logging.getLogger(__name__).warning("No embeddings file found, ILS will be NaN")
                _embeddings_cache = {}
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to load embeddings: {e}")
            _embeddings_cache = {}
    return _embeddings_cache


__all__ = [
    "rank_biased_overlap",
    "ProfileRecs",
    "measure_profile_recs",
    "least_item_promoted",
    "rank_bias_entropy",
    "intralist_similarity",
]

logger = logging.getLogger(__name__)


class ProfileRecs(NamedTuple):
    """
    A user profile's recommendations (possibly from multiple algorithms and stages)
    """

    profile_id: UUID
    recs: pd.DataFrame
    truth: pd.DataFrame


def convert_df_to_article_set(rec_df, eval_data=None):
    articles = []
    embeddings_cache = load_embeddings_cache()

    for _, row in rec_df.iterrows():
        if eval_data is not None:
            article = eval_data.lookup_article(uuid=UUID(row["item_id"]))
            articles.append(article)
        else:
            articles.append(Article(article_id=row["item_id"], headline=""))

    embeddings = []
    # Determine embedding size dynamically from cache or use default
    embedding_size = 768  # Default for DistilBERT
    if embeddings_cache:
        # Get the size from the first available embedding
        for emb in embeddings_cache.values():
            if emb is not None and len(emb) > 0:
                embedding_size = len(emb)
                break

    for article in articles:
        emb = embeddings_cache.get(str(article.article_id))
        if emb is not None:
            embeddings.append(emb)
        else:
            embeddings.append(np.zeros(embedding_size))

    if embeddings:
        embeddings_array = np.stack(embeddings)
    else:
        embeddings_array = None

    return CandidateSet(articles=articles, embeddings=embeddings_array)


def measure_profile_recs(profile: ProfileRecs, eval_data: EvalData | None = None) -> dict[str, Any]:
    """
    Measure a single user profile's recommendations.  Returns the profile ID and
    a dictionary of evaluation metrics.
    """
    profile_id, recs, truth = profile
    truth.index = truth.index.astype(str)

    truth = truth.reset_index()

    truth = truth[truth["rating"] > 0]
    truth = ItemList.from_df(truth)

    final_rec_df = recs[recs["stage"] == "final"]
    final_rec = ItemList.from_df(final_rec_df)

    single_rr = call_metric(RecipRank, final_rec, truth)
    single_ndcg5 = call_metric(NDCG, final_rec, truth, k=5)
    single_ndcg10 = call_metric(NDCG, final_rec, truth, k=10)

    ranked_rec_df = recs[recs["stage"] == "ranked"]
    ranked = convert_df_to_article_set(ranked_rec_df, eval_data)

    reranked_rec_df = recs[recs["stage"] == "reranked"]
    reranked = convert_df_to_article_set(reranked_rec_df, eval_data)

    if ranked and reranked:
        single_rbo5 = rank_biased_overlap(ranked, reranked, k=5)
        single_rbo10 = rank_biased_overlap(ranked, reranked, k=10)
        lip = least_item_promoted(ranked, reranked, k=10)
    else:
        single_rbo5 = None
        single_rbo10 = None
        lip = None

    rbe = rank_bias_entropy(reranked, k=10, d=0.5, eval_data=eval_data)
    ils = intralist_similarity(reranked, k=10)

    logger.debug(
        "profile %s: NDCG@5=%0.3f, NDCG@10=%0.3f, RR=%0.3f, RBO@5=%0.3f, RBO@10=%0.3f",
        " LIP=%0.3f, RBE=%0.3f",
        profile_id,
        single_ndcg5,
        single_ndcg10,
        single_rr,
        single_rbo5 or -1.0,
        single_rbo10 or -1.0,
        lip,
        rbe,
        ils,
    )

    return {
        "profile_id": profile_id,
        # FIXME: this is some hard-coded knowledge of our rec pipeline, but this
        # whole function should be revised for generality when we want to support
        # other pipelines.
        "personalized": len(ranked.articles) > 0,
        "NDCG@5": single_ndcg5,
        "NDCG@10": single_ndcg10,
        "RR": single_rr,
        "RBO@5": single_rbo5,
        "RBO@10": single_rbo10,
        "rank_based_entropy": rbe,
        "least_item_promoted": lip,
        "intralist_similarity": ils,
    }
