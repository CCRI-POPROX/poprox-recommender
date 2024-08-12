"""
Generate evaluations for offline test data.

For an evaluation run NAME, it reads outputs/NAME-recommendation.parquet and
produces OUTPUTS/name-user-eval-metrics.csv.gz and OUTPUTS/name-metrics.json.

Usage:
    poprox_recommender.evaluation.evaluate [options] <name>

Options:
    -v, --verbose       enable verbose diagnostic logs
    --log-file=FILE     write log messages to FILE
    <name>              the name of the evaluation to run [default: mind-val]
"""

# pyright: basic
import csv
import gzip
import json
import logging
from multiprocessing import Pool
from typing import Iterator, NamedTuple
from uuid import UUID

import numpy as np
import pandas as pd
from docopt import docopt
from lenskit.metrics import topn
from tqdm import tqdm

from poprox_concepts.api.recommendations import RecommendationRequest
from poprox_concepts.domain import Article, ArticleSet
from poprox_recommender.config import available_cpu_parallelism
from poprox_recommender.data.mind import MindData
from poprox_recommender.evaluation.metrics import rank_biased_overlap
from poprox_recommender.logging_config import setup_logging
from poprox_recommender.paths import project_root

logger = logging.getLogger("poprox_recommender.evaluation.evaluate")


class UserRecs(NamedTuple):
    request: RecommendationRequest
    recs: pd.DataFrame
    truth: pd.DataFrame


def convert_df_to_article_set(rec_df):
    articles = []
    for _, row in rec_df.iterrows():
        articles.append(Article(article_id=row["item"], title=""))
    return ArticleSet(articles=articles)


def compute_rec_metric(user: UserRecs):
    # Make sure you run generate.py first to get the recommendations data file
    # Convert truth index from UUID to string
    request, recs, truth = user
    truth.index = truth.index.astype(str)

    personalized = 1 if request.interest_profile.click_history.article_ids else 0

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

    return (
        request.interest_profile.profile_id,
        single_ndcg5,
        single_ndcg10,
        single_rr,
        single_rbo5,
        single_rbo10,
        personalized,
    )


def rec_users(mind_data: MindData, user_recs: dict[UUID, pd.DataFrame]) -> Iterator[UserRecs]:
    """
    Iterate over rec users, yielding each request with its recommendations and
    truth.  This supports parallel computation of the final metrics.
    """
    for request in tqdm(mind_data.iter_users(), total=mind_data.n_users, desc="evaluate"):
        user_id = request.interest_profile.profile_id
        assert user_id is not None
        recs = user_recs[user_id]
        truth = mind_data.user_truth(user_id)
        assert truth is not None
        yield UserRecs(request, recs, truth)


def main():
    options = docopt(__doc__)  # type: ignore
    setup_logging(verbose=options["--verbose"], log_file=options["--log-file"])

    global mind_data
    mind_data = MindData()

    eval_name = options["<name>"]
    logger.info("measuring evaluation %s", eval_name)
    recs_fn = project_root() / "outputs" / f"{eval_name}-recommendations.parquet"
    logger.info("loading recommendations from %s", recs_fn)
    recs_df = pd.read_parquet(recs_fn)
    user_recs = dict((UUID(str(u)), df) for (u, df) in recs_df.groupby("user"))
    del recs_df
    logger.info("loaded recommendations for %d users", len(user_recs))

    logger.info("measuring recommendations")
    user_out_fn = project_root() / "outputs" / f"{eval_name}-user-metrics.csv.gz"
    logger.info("writing user results to %s", user_out_fn)
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

    n_workers = available_cpu_parallelism(4)
    logger.info("running with %d workers", n_workers)
    with Pool(n_workers) as pool:
        for metrics in tqdm(
            pool.imap_unordered(compute_rec_metric, rec_users(mind_data, user_recs)),
            total=mind_data.n_users,
            desc="evaluate",
        ):
            user_id, single_ndcg5, single_ndcg10, single_rr, single_rbo5, single_rbo10, personalized = metrics
            user_csv.writerow(
                [
                    user_id,
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
                user_id,
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
    out_fn = project_root() / "outputs" / f"{eval_name}-metrics.json"
    logger.info("saving evaluation to %s", out_fn)
    out_fn.parent.mkdir(exist_ok=True, parents=True)
    out_fn.write_text(json.dumps(agg_metrics) + "\n")
    logger.info("Mean NDCG@5: %.3f", np.mean(ndcg5))
    logger.info("Mean NDCG@10: %.3f", np.mean(ndcg10))
    logger.info("Mean RR: %.3f", np.mean(recip_rank))
    logger.info("Mean RBO@5: %.3f", np.nanmean(rbo5))
    logger.info("Mean RBO@10: %.3f", np.nanmean(rbo10))


if __name__ == "__main__":
    main()
