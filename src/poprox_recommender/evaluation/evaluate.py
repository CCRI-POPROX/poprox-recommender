"""
Generate evaluations for offline test data.

For an evaluation run NAME, it reads outputs/NAME-recommendation.parquet and
produces OUTPUTS/name-user-eval-metrics.csv.gz and OUTPUTS/name-metrics.json.

Usage:
    poprox_recommender.evaluation.evaluate [options] <name>

Options:
    -v, --verbose       enable verbose diagnostic logs
    --log-file=FILE     write log messages to FILE
    -M DATA, --mind-data=DATA
            read MIND test data DATA [default: MINDsmall_dev]
    -P DATA, --poprox-data=DATA
            read POPROX test data DATA
    <name>              the name of the evaluation to measure
"""

# pyright: basic
import logging
from typing import Any, Iterator
from uuid import UUID

import ipyparallel as ipp
import pandas as pd
from docopt import docopt
from progress_api import make_progress

# from poprox_recommender.data.mind import MindData
from poprox_recommender.config import available_cpu_parallelism
from poprox_recommender.data.eval import EvalData
from poprox_recommender.data.mind import MindData
from poprox_recommender.data.poprox import PoproxData
from poprox_recommender.evaluation.metrics import UserRecs, measure_user_recs
from poprox_recommender.logging_config import setup_logging
from poprox_recommender.paths import project_root

logger = logging.getLogger("poprox_recommender.evaluation.evaluate")


def rec_users(eval_data: EvalData, user_recs: pd.DataFrame) -> Iterator[UserRecs]:
    """
    Iterate over rec users, yielding each recommendation list with its truth and
    whether the user is personalized.  This supports parallel computation of the
    final metrics.
    """
    for user_id, recs in user_recs.groupby("user"):
        user_id = UUID(str(user_id))
        truth = eval_data.profile_truth(user_id)
        assert truth is not None
        yield UserRecs(user_id, recs.copy(), truth)


def user_eval_results(eval_data: EvalData, user_recs: pd.DataFrame, n_procs: int) -> Iterator[list[dict[str, Any]]]:
    if n_procs > 1:
        logger.info("starting parallel measurement with %d workers", n_procs)
        with ipp.Cluster(n=n_procs) as client:
            lb = client.load_balanced_view()
            yield from lb.imap(
                measure_user_recs, rec_users(eval_data, user_recs), ordered=False, max_outstanding=n_procs * 10
            )
    else:
        for user in rec_users(eval_data, user_recs):
            yield measure_user_recs(user)


def main():
    options = docopt(__doc__)  # type: ignore
    setup_logging(verbose=options["--verbose"], log_file=options["--log-file"])

    global eval_data

    if options["--poprox-data"]:
        eval_data = PoproxData(options["--poprox-data"])
    else:
        eval_data = MindData(options["--mind-data"])

    eval_name = options["<name>"]
    logger.info("measuring evaluation %s", eval_name)
    recs_fn = project_root() / "outputs" / eval_name / "recommendations"
    logger.info("loading recommendations from %s", recs_fn)
    recs_df = pd.read_parquet(recs_fn)
    n_users = recs_df["user"].nunique()
    logger.info("loaded recommendations for %d users", n_users)

    logger.info("measuring recommendations")

    n_procs = available_cpu_parallelism(4)
    records = []
    with (
        make_progress(logger, "evaluate", total=n_users, unit="users") as pb,
    ):
        for user_rows in user_eval_results(eval_data, recs_df, n_procs):
            records += user_rows
            pb.update()

    metrics = pd.DataFrame.from_records(records)
    logger.info("measured recs for %d users", metrics["user_id"].nunique())

    user_out_fn = project_root() / "outputs" / eval_name / "user-metrics.csv.gz"
    logger.info("saving per-user metrics to %s", user_out_fn)
    metrics.to_csv(user_out_fn)

    agg_metrics = metrics.drop(columns=["user_id", "personalized"]).groupby("recommender").mean()
    # reciprocal rank means to MRR
    agg_metrics = agg_metrics.rename(columns={"RR": "MRR"})

    logger.info("aggregate metrics:\n%s", agg_metrics)

    out_fn = project_root() / "outputs" / eval_name / "metrics.csv"
    logger.info("saving evaluation to %s", out_fn)
    agg_metrics.to_csv(out_fn)


if __name__ == "__main__":
    main()
