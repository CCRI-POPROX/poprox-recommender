import logging
from typing import Any, NamedTuple
from uuid import UUID

import pandas as pd
from lenskit.data import ItemList
from lenskit.metrics import call_metric
from lenskit.metrics.ranking import NDCG, RecipRank

from poprox_concepts import Article, CandidateSet
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
    return CandidateSet(articles=articles)


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

    for (name, theta_topic, theta_loc, similarity_threshold), recs in all_recs.groupby(
        ["recommender", "theta_topic", "theta_locality", "similarity_threshold"], observed=True
    ):
        final_rec_df = recs[recs["stage"] == "final"]
        final_rec = ItemList.from_df(final_rec_df)

        single_rr = call_metric(RecipRank, final_rec, truth)
        single_ndcg5 = call_metric(NDCG, final_rec, truth, k=5)
        single_ndcg10 = call_metric(NDCG, final_rec, truth, k=10)

        ranked_rec_df = recs[recs["stage"] == "ranked"]
        ranked = convert_df_to_article_set(ranked_rec_df)

        reranked_rec_df = recs[recs["stage"] == "reranked"]
        reranked = convert_df_to_article_set(reranked_rec_df)

        generator_rec_df = recs[recs["stage"] == "generator"]

        # Locality tuning metrcis
        if str(name).startswith("locality_cali"):
            # newsletter metrics (grab first because they are the same between articles)
            # logger.error(reranked_rec_df.head())
            k1_topic = reranked_rec_df["k1_topic"].iloc[0] if not reranked_rec_df["k1_topic"].empty else None
            k1_loc = reranked_rec_df["k1_locality"].iloc[0] if not reranked_rec_df["k1_locality"].empty else None
            is_inside_locality_threshold = (
                reranked_rec_df["is_inside_locality_threshold"].iloc[0]
                if not reranked_rec_df["is_inside_locality_threshold"].empty
                else None
            )
            # individual rec metrics
            num_treatment = reranked_rec_df["treatment"].sum()
        else:
            k1_topic = None
            k1_loc = None
            is_inside_locality_threshold = None
            event_level_prompt_ratio = None
            num_treatment = None

        if name == "locality_cali_context":
            event_level_prompt_ratio = (
                generator_rec_df["prompt_level_ratio"].iloc[0]
                if not generator_rec_df["prompt_level_ratio"].empty
                else None
            )
            rouge1 = generator_rec_df["rouge1"].iloc[0] if not generator_rec_df["rouge1"].empty else None
            rouge2 = generator_rec_df["rouge2"].iloc[0] if not generator_rec_df["rouge2"].empty else None
            rougeL = generator_rec_df["rougeL"].iloc[0] if not generator_rec_df["rougeL"].empty else None
            prompt_level = (
                generator_rec_df["prompt_level"].iloc[0] if not generator_rec_df["prompt_level"].empty else None
            )
        else:
            event_level_prompt_ratio = None
            rouge1 = None
            rouge2 = None
            rougeL = None
            prompt_level = None

        if ranked and reranked:
            single_rbo5 = rank_biased_overlap(ranked, reranked, k=5)
            single_rbo10 = rank_biased_overlap(ranked, reranked, k=10)
        else:
            single_rbo5 = None
            single_rbo10 = None

        logger.debug(
            "profile %s rec %s: NDCG@5=%0.3f, NDCG@10=%0.3f, RR=%0.3f, RBO@5=%0.3f, RBO@10=%0.3f",
            profile_id,
            name,
            single_ndcg5,
            single_ndcg10,
            single_rr,
            single_rbo5 or -1.0,
            single_rbo10 or -1.0,
        )

        results.append(
            {
                "profile_id": profile_id,
                "recommender": name,
                "theta_topic": theta_topic,
                "theta_loc": theta_loc,
                "similarity_threshold": similarity_threshold,
                # FIXME: this is some hard-coded knowledge of our rec pipeline, but this
                # whole function should be revised for generality when we want to support
                # other pipelines.
                "personalized": len(ranked.articles) > 0,
                "NDCG@5": single_ndcg5,
                "NDCG@10": single_ndcg10,
                "RR": single_rr,
                "RBO@5": single_rbo5,
                "RBO@10": single_rbo10,
                "KL_TOPIC": k1_topic,
                "KL_LOC": k1_loc,
                "event_level_prompt_ratio": event_level_prompt_ratio,
                "inside_loc_threshold": is_inside_locality_threshold,
                "num_treatment": num_treatment,
                "rouge1": rouge1,
                "rouge2": rouge2,
                "rougeL": rougeL,
                "prompt_level": prompt_level,
            }
        )

    return results
