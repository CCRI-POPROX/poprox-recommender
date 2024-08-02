import csv
import gzip
import json
import logging

import numpy as np
import pandas as pd
from lenskit.metrics import topn
from line_profiler import LineProfiler
from tqdm import tqdm

from poprox_concepts.api.recommendations import RecommendationRequest
from poprox_concepts.domain import Article, ArticleSet
from poprox_recommender.data.mind import MindData
from poprox_recommender.evaluation.metrics import rank_biased_overlap
from poprox_recommender.paths import project_root

logger = logging.getLogger("poprox_recommender.measurement_offline")


def convert_df_to_article_set(rec_df):
    articles = []
    for _, row in rec_df.iterrows():
        articles.append(Article(article_id=row["item"], title=""))
    return ArticleSet(articles=articles)


def compute_rec_metric(recs_df: pd.DataFrame, request: RecommendationRequest):
    # Make sure you run generate.py first to get the recommendations data file
    recs = recs_df.loc[str(request.interest_profile.profile_id)]
    truth = mind_data.user_truth(request.interest_profile.profile_id)

    personalized = 1 if request.interest_profile.click_history.article_ids else 0

    single_rr = topn.recip_rank(recs[recs["stage"] == "final"], truth[truth["rating"] > 0])
    single_ndcg5 = topn.ndcg(recs[recs["stage"] == "final"], truth, k=5)
    single_ndcg10 = topn.ndcg(recs[recs["stage"] == "final"], truth, k=10)

    ranked = convert_df_to_article_set(recs[recs["stage"] == "ranked"])
    reranked = convert_df_to_article_set(recs[recs["stage"] == "reranked"])

    if ranked and reranked:
        single_rbo5 = rank_biased_overlap(ranked, reranked, k=5)
        single_rbo10 = rank_biased_overlap(ranked, reranked, k=10)
    else:
        single_rbo5 = None
        single_rbo10 = None

    return single_ndcg5, single_ndcg10, single_rr, single_rbo5, single_rbo10, personalized


if __name__ == "__main__":
    mind_data = MindData()

    recs_fn = project_root() / "outputs" / "mind-val-recommendations.parquet"
    recs_df = pd.read_parquet(recs_fn)

    logger.info("measuring recommendations")
    user_out_fn = project_root() / "outputs" / "user-metrics-test.csv.gz"
    user_out_fn.parent.mkdir(exist_ok=True, parents=True)
    user_out = gzip.open(user_out_fn, "wt")
    user_csv = csv.writer(user_out)
    user_csv.writerow(["user_id", "personalized", "NDCG@5", "NDCG@10", "RecipRank", "RBO@5", "RBO@10"])

    ngood = 0
    nbad = 0
    ndcg5 = []
    ndcg10 = []
    recip_rank = []
    rbo5 = []
    rbo10 = []

    recs_df.set_index("user", inplace=True)
    for request in tqdm(mind_data.iter_users(), total=mind_data.n_users, desc="evaluate"):
        lp = LineProfiler()
        lp_wrapper = lp(compute_rec_metric)
        lp.print_stats()
        single_ndcg5, single_ndcg10, single_rr, single_rbo5, single_rbo10, personalized = compute_rec_metric(
            recs_df, request
        )
        user_csv.writerow(
            [
                request.interest_profile.profile_id,
                personalized,
                single_ndcg5,
                single_ndcg10,
                single_rr,
                single_rbo5,
                single_rbo10,
            ]
        )
        # recommendations {account id (uuid): LIST[Article]}
        logger.debug(
            "user %s: NDCG@5=%0.3f, NDCG@10=%0.3f, RR=%0.3f, RBO@5=%0.3f, RBO@10=%0.3f",
            request.interest_profile.profile_id,
            single_ndcg5,
            single_ndcg10,
            single_rr,
            single_rbo5 or -1.0,
            single_rbo10 or -1.0,
        )

        ndcg5.append(single_ndcg5)
        ndcg10.append(single_ndcg10)
        recip_rank.append(single_rr)
        rbo5.append(single_rbo5 or np.nan)
        rbo10.append(single_rbo10 or np.nan)
        ngood += 1

    user_out.close()

    logger.info("recommended for %d users", ngood)
    if nbad:
        logger.error("recommendation FAILED for %d users", nbad)
    agg_metrics = {
        "NDCG@5": np.mean(ndcg5),
        "NDCG@10": np.mean(ndcg10),
        "MRR": np.mean(recip_rank),
        "RBO@5": np.nanmean(rbo5),
        "RBO@10": np.nanmean(rbo10),
    }
    out_fn = project_root() / "outputs" / "metrics-test.json"
    out_fn.parent.mkdir(exist_ok=True, parents=True)
    out_fn.write_text(json.dumps(agg_metrics) + "\n")
    logger.info("Mean NDCG@5: %.3f", np.mean(ndcg5))
    logger.info("Mean NDCG@10: %.3f", np.mean(ndcg10))
    logger.info("Mean RR: %.3f", np.mean(recip_rank))
    logger.info("Mean RBO@5: %.3f", np.mean(rbo5))
    logger.info("Mean RBO@10: %.3f", np.mean(rbo10))
