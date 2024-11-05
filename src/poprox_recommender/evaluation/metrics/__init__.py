import logging
from typing import Any, NamedTuple
from uuid import UUID

import pandas as pd
from lenskit.metrics import topn

from poprox_concepts import Article, ArticleSet
from poprox_recommender.evaluation.metrics.rbo import rank_biased_overlap

__all__ = ["rank_biased_overlap", "UserRecs", "measure_user_recs"]

logger = logging.getLogger(__name__)

METRIC_COLUMNS = [
    "user_id",
    "recommender",
    "personalized",
    "NDCG@5",
    "NDCG@10",
    "MRR",
    "RBO@5",
    "RBO@10",
]


class UserRecs(NamedTuple):
    """
    A user's recommendations (possibly from multiple algorithms and stages)
    """

    user_id: UUID
    personalized: bool
    recs: pd.DataFrame
    truth: pd.DataFrame


def convert_df_to_article_set(rec_df):
    articles = []
    for _, row in rec_df.iterrows():
        articles.append(Article(article_id=row["item"], headline=""))
    return ArticleSet(articles=articles)


def measure_user_recs(user: UserRecs) -> list[list[Any]]:
    """
    Measure a single user's recommendations.  Returns the user ID and
    a data frame of evaluation metrics.
    """
    user_id, personalized, all_recs, truth = user
    truth.index = truth.index.astype(str)

    results = []

    for name, recs in all_recs.groupby("recommender", observed=True):
        final_rec_df = recs[recs["stage"] == "final"]
        single_rr = topn.recip_rank(final_rec_df, truth[truth["rating"] > 0])
        single_ndcg5 = topn.ndcg(final_rec_df, truth, k=5)
        single_ndcg10 = topn.ndcg(final_rec_df, truth, k=10)

        ranked_rec_df = recs[recs["stage"] == "ranked"]
        ranked = convert_df_to_article_set(ranked_rec_df)

        reranked_rec_df = recs[recs["stage"] == "reranked"]
        reranked = convert_df_to_article_set(reranked_rec_df)

        if ranked and reranked:
            single_rbo5 = rank_biased_overlap(ranked, reranked, k=5)
            single_rbo10 = rank_biased_overlap(ranked, reranked, k=10)
        else:
            single_rbo5 = None
            single_rbo10 = None

        logger.debug(
            "user %s rec %s: NDCG@5=%0.3f, NDCG@10=%0.3f, RR=%0.3f, RBO@5=%0.3f, RBO@10=%0.3f",
            user_id,
            name,
            single_ndcg5,
            single_ndcg10,
            single_rr,
            single_rbo5 or -1.0,
            single_rbo10 or -1.0,
        )

        # KEEP IN SYNC with the metric columns above
        results.append(
            [
                user_id,
                name,
                personalized,
                single_ndcg5,
                single_ndcg10,
                single_rr,
                single_rbo5,
                single_rbo10,
            ]
        )

    return results
