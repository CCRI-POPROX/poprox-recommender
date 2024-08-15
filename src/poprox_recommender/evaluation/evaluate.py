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
import logging
from concurrent.futures import ProcessPoolExecutor
from typing import Iterator, NamedTuple
from uuid import UUID

import pandas as pd
from docopt import docopt
from lenskit.metrics import topn
from progress_api import make_progress

from poprox_concepts.domain import Article, ArticleSet
from poprox_recommender.config import available_cpu_parallelism
from poprox_recommender.data.mind import MindData
from poprox_recommender.evaluation.metrics import rank_biased_overlap
from poprox_recommender.logging_config import setup_logging
from poprox_recommender.paths import project_root

logger = logging.getLogger("poprox_recommender.evaluation.evaluate")


class UserRecs(NamedTuple):
    user_id: UUID
    personalized: bool
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

        results.append(
            {
                "recommender": name,
                "NDCG@5": single_ndcg5,
                "NDCG@10": single_ndcg10,
                "MRR": single_rr,
                "RBO@5": single_rbo5,
                "RBO@10": single_rbo10,
                "personalized": personalized,
            }
        )

    return user_id, pd.DataFrame.from_records(results).set_index("recommender")


def rec_users(mind_data: MindData, user_recs: dict[UUID, pd.DataFrame]) -> Iterator[UserRecs]:
    """
    Iterate over rec users, yielding each request with its recommendations and
    truth.  This supports parallel computation of the final metrics.
    """
    for request in mind_data.iter_users():
        user_id = request.interest_profile.profile_id
        assert user_id is not None
        recs = user_recs[user_id]
        truth = mind_data.user_truth(user_id)
        assert truth is not None
        yield UserRecs(user_id, bool(request.interest_profile.click_history), recs, truth)


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
    results: dict[UUID, pd.DataFrame] = {}

    n_workers = available_cpu_parallelism(4)
    logger.info("running with %d workers", n_workers)
    with ProcessPoolExecutor(n_workers) as pool, make_progress(logger, "evaluate", total=mind_data.n_users) as pb:
        for result in pool.map(compute_rec_metric, rec_users(mind_data, user_recs)):
            user_id, metrics = result
            results[user_id] = metrics
            pb.update()

    logger.info("measured recs for %d users", len(results))
    metrics = pd.concat(results, names=["user_id"]).reset_index()

    user_out_fn = project_root() / "outputs" / f"{eval_name}-user-metrics.csv.gz"
    logger.info("writing user results to %s", user_out_fn)
    metrics.to_csv(user_out_fn, index=False)

    agg_metrics = metrics.drop(columns=["user_id", "personalized"]).groupby("recommender").mean()

    logger.info("aggregate metrics:\n%s", agg_metrics)

    out_fn = project_root() / "outputs" / f"{eval_name}-metrics.csv"
    logger.info("saving evaluation to %s", out_fn)
    agg_metrics.to_csv(out_fn)


if __name__ == "__main__":
    main()
