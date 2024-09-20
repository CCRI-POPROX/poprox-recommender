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
from typing import Iterator
from uuid import UUID

import ipyparallel as ipp
import pandas as pd
from docopt import docopt
from progress_api import make_progress

from poprox_recommender.config import available_cpu_parallelism
from poprox_recommender.data.mind import MindData
from poprox_recommender.evaluation.metrics import UserRecs, measure_user_recs
from poprox_recommender.logging_config import setup_logging
from poprox_recommender.paths import project_root

logger = logging.getLogger("poprox_recommender.evaluation.evaluate")


def rec_users(mind_data: MindData, user_recs: dict[UUID, pd.DataFrame]) -> Iterator[UserRecs]:
    """
    Iterate over rec users, yielding each request with its recommendations and
    truth.  This supports parallel computation of the final metrics.
    """
    for request in mind_data.iter_users():
        user_id = request.interest_profile.profile_id
        assert user_id is not None
        recs = user_recs[user_id].copy(deep=True)
        truth = mind_data.user_truth(user_id)
        assert truth is not None
        yield UserRecs(user_id, bool(request.interest_profile.click_history), recs, truth)


def user_eval_results(
    mind_data: MindData, user_recs: dict[UUID, pd.DataFrame], n_procs: int
) -> Iterator[tuple[UUID, pd.DataFrame]]:
    if n_procs > 1:
        logger.info("starting parallel measurement with %d workers", n_procs)
        with ipp.Cluster(n=n_procs) as client:
            lb = client.load_balanced_view()
            yield from lb.imap(measure_user_recs, rec_users(mind_data, user_recs), ordered=False)
    else:
        for user in rec_users(mind_data, user_recs):
            yield measure_user_recs(user)


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

    n_procs = available_cpu_parallelism(4)
    with make_progress(logger, "evaluate", total=mind_data.n_users, unit="users") as pb:
        for user_id, metrics in user_eval_results(mind_data, user_recs, n_procs):
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
