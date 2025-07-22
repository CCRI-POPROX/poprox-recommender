#!/usr/bin/env python3
"""
Generate evaluations for offline test data.

For an evaluation EVAL and PIPELINE, this script reads
outputs/DATA/PIPELINE/recommendations.parquet and produces
ouptuts/DATA/PIPELINE/profile-metrics.csv.gz and ouptuts/DATA/PIPELINE/metrics.json.

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
from itertools import batched
from typing import Any, Iterator
from uuid import UUID

import pandas as pd
import ray
from docopt import docopt
from lenskit.logging import LoggingConfig, item_progress
from lenskit.parallel import get_parallel_config
from lenskit.parallel.ray import init_cluster

from poprox_recommender.data.eval import EvalData
from poprox_recommender.data.mind import MindData
from poprox_recommender.data.poprox import PoproxData
from poprox_recommender.evaluation.metrics import ProfileRecs, measure_profile_recs
from poprox_recommender.paths import project_root

logger = logging.getLogger(__name__)


def rec_profiles(eval_data: EvalData, profile_recs: pd.DataFrame) -> Iterator[ProfileRecs]:
    """
    Iterate over rec profiles, yielding each recommendation list with its truth and
    whether the profile is personalized.  This supports parallel computation of the
    final metrics.
    """
    for profile_id, recs in profile_recs.groupby("profile_id"):
        profile_id = UUID(str(profile_id))
        truth = eval_data.profile_truth(profile_id)
        assert truth is not None
        if len(truth) > 0:
            yield ProfileRecs(profile_id, recs.copy(), truth)


def profile_eval_results(eval_data: EvalData, profile_recs: pd.DataFrame) -> Iterator[dict[str, Any]]:
    pc = get_parallel_config()
    profiles = rec_profiles(eval_data, profile_recs)
    if pc.processes > 1:
        logger.info("starting parallel measurement with %d workers", pc.processes)
        init_cluster(global_logging=True)

        eval_data_ref = ray.put(eval_data)

        # use the batch backpressure mechanism
        # https://docs.ray.io/en/latest/ray-core/patterns/limit-pending-tasks.html
        result_refs = []
        for batch in batched(profiles, 100):
            if len(result_refs) > pc.processes:
                # wait for a result, and return it
                ready_refs, result_refs = ray.wait(result_refs, num_returns=1)
                for rr in ready_refs:
                    yield from ray.get(rr)

            result_refs.append(measure_batch.remote(batch, eval_data_ref))

        # yield remaining items
        while result_refs:
            ready_refs, result_refs = ray.wait(result_refs, num_returns=1)
            for rr in ready_refs:
                yield from ray.get(rr)

    else:
        for profile in rec_profiles(eval_data, profile_recs):
            yield measure_profile_recs(profile, eval_data)


def main():
    options = docopt(__doc__)  # type: ignore
    log_cfg = LoggingConfig()
    if options["--verbose"] or os.environ.get("RUNNER_DEBUG", 0):
        log_cfg.set_verbose(True)
    if options["--log-file"]:
        log_cfg.set_log_file(options["--log-file"])
    log_cfg.apply()

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
    n_profiles = recs_df["profile_id"].nunique()
    logger.info("loaded recommendations for %d profiles", n_profiles)

    logger.info("measuring recommendations")

    records = []
    with (
        item_progress("evaluate", total=n_profiles) as pb,
    ):
        for profile_rows in profile_eval_results(eval_data, recs_df):
            records.append(profile_rows)
            pb.update()

    metrics = pd.DataFrame.from_records(records)
    logger.info("measured recs for %d profiles", metrics["profile_id"].nunique())

    profile_out_fn = project_root() / "outputs" / eval_name / pipe_name / "profile-metrics.csv.gz"
    logger.info("saving per-profile metrics to %s", profile_out_fn)
    metrics.to_csv(profile_out_fn)

    agg_metrics = metrics.drop(columns=["profile_id", "personalized"]).mean()
    # reciprocal rank mean to MRR
    agg_metrics = agg_metrics.rename(index={"RR": "MRR"})

    logger.info("aggregate metrics:\n%s", agg_metrics)

    out_fn = project_root() / "outputs" / eval_name / pipe_name / "metrics.json"
    logger.info("saving evaluation to %s", out_fn)
    with open(out_fn, "wt") as jsf:
        print(agg_metrics.to_json(), file=jsf)


@ray.remote(num_cpus=1)
def measure_batch(profiles: list[ProfileRecs], eval_data_ref) -> list[dict[str, Any]]:
    """
    Measure a batch of profile recommendations.
    """
    eval_data = eval_data_ref
    return [measure_profile_recs(profile, eval_data) for profile in profiles]


if __name__ == "__main__":
    main()
