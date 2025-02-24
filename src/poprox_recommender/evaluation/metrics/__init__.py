import logging
from typing import Any, NamedTuple
from uuid import UUID

import pandas as pd
from lenskit.data import ItemList
from lenskit.metrics import call_metric
from lenskit.metrics.ranking import NDCG, RecipRank

from poprox_concepts import Article, ArticleSet
from poprox_recommender.evaluation.metrics.entropy import rank_bias_entropy
from poprox_recommender.evaluation.metrics.gini import gini_coeff
from poprox_recommender.evaluation.metrics.kcoverage import k_coverage_score
from poprox_recommender.evaluation.metrics.lip import least_item_promoted
from poprox_recommender.evaluation.metrics.rbo import rank_biased_overlap

__all__ = ["rank_biased_overlap", "ProfileRecs", "measure_profile_recs"]

logger = logging.getLogger(__name__)


class ProfileRecs(NamedTuple):
    """
    A user profile's recommendations (possibly from multiple algorithms and stages)
    """

    profile_id: UUID
    recs: pd.DataFrame
    truth: pd.DataFrame


def convert_df_to_article_set(rec_df):
    articles = []
    for _, row in rec_df.iterrows():
        articles.append(Article(article_id=row["item_id"], headline=""))
    return ArticleSet(articles=articles)


def measure_profile_recs(profile: ProfileRecs) -> list[dict[str, Any]]:
    """
    Measure a single user profile's recommendations.  Returns the profile ID and
    an ItemList of evaluation metrics.
    """
    profile_id, all_recs, truth = profile
    truth.index = truth.index.astype(str)

    truth = truth.reset_index()

    truth = truth[truth["rating"] > 0]
    truth = ItemList.from_df(truth)

    results = []

    for name, recs in all_recs.groupby("recommender", observed=True):
        final_rec_df = recs[recs["stage"] == "final"]
        final_rec = ItemList.from_df(final_rec_df)

        single_rr = call_metric(RecipRank, final_rec, truth)
        single_ndcg5 = call_metric(NDCG, final_rec, truth, k=5)
        single_ndcg10 = call_metric(NDCG, final_rec, truth, k=10)

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

        rbe = rank_bias_entropy(final_rec, k=10, d=0.5)
        gini = gini_coeff(final_rec)
        k_coverage = k_coverage_score(ranked, reranked, k=1)
        lip = least_item_promoted(ranked, reranked, k=10)

        logger.debug(
            "profile %s rec %s: "
            "NDCG@5=%0.3f, NDCG@10=%0.3f, RR=%0.3f, RBO@5=%0.3f, RBO@10=%0.3f, RBE=%0.3f,"
            "Gini=%0.3f, KCoverage=%0.3f, LIP=%0.3f",
            profile_id,
            name,
            single_ndcg5,
            single_ndcg10,
            single_rr,
            single_rbo5 or -1.0,
            single_rbo10 or -1.0,
            rbe,
            gini,
            k_coverage,
            lip,
        )

        results.append(
            {
                "profile_id": profile_id,
                "recommender": name,
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
                "gini_index": gini,
                "k_coverage": k_coverage,
                "least_item_promoted": lip,
            }
        )

    return results
