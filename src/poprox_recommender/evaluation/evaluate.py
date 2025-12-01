#!/usr/bin/env python3
"""
Generate evaluations for offline test data.

For an evaluation EVAL and PIPELINE, this script reads
outputs/DATA/PIPELINE/recommendations.parquet and produces
ouptuts/DATA/PIPELINE/recommendation-metrics.csv.gz and ouptuts/DATA/PIPELINE/metrics.json.

Usage:
    poprox_recommender.evaluation.evaluate [options] EVAL PIPELINE

Options:
    -v, --verbose
            enable verbose diagnostic logs
    --log-file=FILE
            write log messages to FILE
    -M DATA, --mind-data=DATA
            read MIND test data DATA [default: MINDsmall_dev]
    -P DATA, --poprox-data=DATA
            read POPROX test data DATA
    EVAL    the name of the evaluation to measure
    PIPELINE
            the name of the pipeline to measure
"""

# pyright: basic
import logging
import os
from collections.abc import Sequence
from itertools import batched
from typing import Any, Iterator
from uuid import UUID

import lenskit
import pandas as pd
import ray
from docopt import docopt
from lenskit.logging import LoggingConfig, item_progress
from lenskit.parallel import get_parallel_config
from lenskit.parallel.ray import TaskLimiter, init_cluster

from poprox_recommender.data.eval import EvalData
from poprox_recommender.data.mind import MindData
from poprox_recommender.data.poprox import PoproxData
from poprox_recommender.evaluation.metrics import RecsWithTruth, measure_rec_metrics
from poprox_recommender.paths import project_root

logger = logging.getLogger(__name__)


def recs_with_truth(eval_data: EvalData, recs_df: pd.DataFrame) -> Iterator[RecsWithTruth]:
    """
    Iterate over recommendations, yielding each recommendation list with its
    truth.  This supports parallel computation of the final metrics.
    """
    for slate_id, recs in recs_df.groupby("slate_id"):
        slate_id = UUID(str(slate_id))
        truth = eval_data.slate_truth(slate_id)
        assert truth is not None
        # if len(truth) > 0:
        #     yield RecsWithTruth(recommendation_id, recs.copy(), truth)
        # else:
        #     logger.warning("request %s has no truth", recommendation_id)

        # To include all the recs with or without clicks (for non accuracy metrics)
        yield RecsWithTruth(slate_id, recs.copy(), truth)


def recommendation_eval_results(eval_data: EvalData, recs_df: pd.DataFrame) -> Iterator[dict[str, Any]]:
    pc = get_parallel_config()
    rwts = recs_with_truth(eval_data, recs_df)
    if pc.processes > 1:
        logger.info("starting parallel measurement with up to %d tasks", pc.processes)
        init_cluster(global_logging=True)

        eval_data_ref = ray.put(eval_data)
        limit = TaskLimiter(pc.processes)

        for bres in limit.imap(
            lambda batch: measure_batch.remote(batch, eval_data_ref), batched(rwts, 100), ordered=False
        ):
            assert isinstance(bres, list)
            yield from bres

    else:
        for rwt in recs_with_truth(eval_data, recs_df):
            yield measure_rec_metrics(rwt, eval_data)


def main():
    options = docopt(__doc__)  # type: ignore
    log_cfg = LoggingConfig()
    if options["--verbose"] or os.environ.get("RUNNER_DEBUG", 0):
        log_cfg.set_verbose(True)
    if options["--log-file"]:
        log_cfg.set_log_file(options["--log-file"])
    log_cfg.apply()
    lenskit.configure(cfg_dir=project_root())

    global eval_data

    if options["--poprox-data"]:
        eval_data = PoproxData(options["--poprox-data"])
    else:
        eval_data = MindData(options["--mind-data"])

    eval_name = options["EVAL"]
    pipe_name = options["PIPELINE"]
    logger.info("measuring evaluation %s for %s", eval_name, pipe_name)
    recs_fn = project_root() / "outputs" / eval_name / pipe_name / "recommendations.parquet"
    logger.info("loading recommendations from %s", recs_fn)
    recs_df = pd.read_parquet(recs_fn)
    n_recommendations = recs_df["slate_id"].nunique()
    logger.info("loaded recommendations for %d recommendations", n_recommendations)

    logger.info("measuring recommendations")

    metric_records = []
    with (
        item_progress("evaluate", total=n_recommendations) as pb,
    ):
        for metric_row in recommendation_eval_results(eval_data, recs_df):
            metric_records.append(metric_row)
            pb.update()

    metrics = pd.DataFrame.from_records(metric_records)
    print(metrics)
    logger.info("measured metrics for %d recommendations", metrics["recommendation_id"].nunique())

    metrics_out_fn = project_root() / "outputs" / eval_name / pipe_name / "recommendation-metrics.csv.gz"
    logger.info("saving per-recommendation metrics to %s", metrics_out_fn)
    metrics.to_csv(metrics_out_fn)

    agg_metrics = metrics.drop(columns=["recommendation_id", "personalized"]).mean()
    # reciprocal rank mean to MRR
    agg_metrics = agg_metrics.rename(index={"RR": "MRR"})

    logger.info("aggregate metrics:\n%s", agg_metrics)

    out_fn = project_root() / "outputs" / eval_name / pipe_name / "metrics.json"
    logger.info("saving evaluation to %s", out_fn)
    with open(out_fn, "wt") as jsf:
        print(agg_metrics.to_json(), file=jsf)


@ray.remote(num_cpus=1)
def measure_batch(rwts: Sequence[RecsWithTruth], eval_data) -> list[dict[str, Any]]:
    """
    Measure a batch of recommendations.
    """
    return [measure_rec_metrics(rwt, eval_data) for rwt in rwts]


if __name__ == "__main__":
    main()
