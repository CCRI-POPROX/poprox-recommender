"""
Compute metrics for offline test data.

For an evaluation EVAL and PIPELINE, this script reads
outputs/DATA/PIPELINE/recommendations.parquet and produces
ouptuts/DATA/PIPELINE/recommendation-metrics.csv.gz and ouptuts/DATA/PIPELINE/metrics.json.

When --section-eval is given it additionally reads
outputs/DATA/PIPELINE/recommendations.ndjson.zst (which preserves the section
structure written by JSONRecommendationWriter) and produces
outputs/DATA/PIPELINE/section-recommendation-metrics.csv.gz and
outputs/DATA/PIPELINE/section-metrics.json.

Usage:
    poprox_recommender.evaluation.measure [options] EVAL PIPELINE

Options:
    -v, --verbose
            enable verbose diagnostic logs
    --log-file=FILE
            write log messages to FILE
    -M DATA, --mind-data=DATA
            read MIND test data DATA [default: MINDsmall_dev]
    -P DATA, --poprox-data=DATA
            read POPROX test data DATA
    --section-eval
            run section-based evaluation (requires recommendations.ndjson.zst)
    EVAL    the name of the evaluation to measure
    PIPELINE
            the name of the pipeline to measure
"""

# pyright: basic
import json
import logging
import os
import re
from uuid import UUID

import lenskit
import numpy as np
import pandas as pd
import zstandard
from docopt import docopt
from humanize import metric
from lenskit.logging import LoggingConfig, item_progress
from rich import print
from rich.pretty import pprint

from poprox_concepts.domain import ImpressedSection
from poprox_recommender.data.mind import MindData
from poprox_recommender.data.poprox import PoproxData
from poprox_recommender.evaluation.measure.batch_measure import recommendation_eval_results
from poprox_recommender.evaluation.options import load_eval_options
from poprox_recommender.evaluation.section_metrics import measure_section_rec_metrics
from poprox_recommender.paths import project_root

logger = logging.getLogger("poprox_recommender.evaluation.measure")
EFF_METRIC_NAMES = re.compile(r"^(NDCG|M?RR|RBP)(@|$)?")


def main():
    cli_opts = docopt(__doc__)  # type: ignore
    log_cfg = LoggingConfig()
    if cli_opts["--verbose"] or os.environ.get("RUNNER_DEBUG", 0):
        log_cfg.set_verbose(True)
    if cli_opts["--log-file"]:
        log_cfg.set_log_file(cli_opts["--log-file"])
    log_cfg.apply()
    lenskit.configure(cfg_dir=project_root())

    global eval_data

    if cli_opts["--poprox-data"]:
        eval_data = PoproxData(cli_opts["--poprox-data"])
    else:
        eval_data = MindData(cli_opts["--mind-data"])

    eval_name = cli_opts["EVAL"]
    pipe_name = cli_opts["PIPELINE"]

    out_dir = project_root() / "outputs" / eval_name / pipe_name
    options = load_eval_options(out_dir)

    logger.info("measuring evaluation %s for %s", eval_name, pipe_name)
    recs_fn = out_dir / "recommendations.ndjson.zst"

    logger.info("loading recommendations from %s", recs_fn)
    metric_records = []
    with item_progress("evaluate") as pb:
        for metric_row in recommendation_eval_results(eval_data, recs_fn):
            metric_records.append(metric_row)
            pb.update()

    metrics = pd.DataFrame.from_records(metric_records)

    print("[bold]Preview of Metrics[/bold]")
    print(metrics.set_index("slate_id"))
    logger.info("measured metrics for %d recommendations", metrics["slate_id"].nunique())

    metrics_out_fn = out_dir / "recommendation-metrics.csv.gz"
    logger.info("saving per-recommendation metrics to %s", metrics_out_fn)
    metrics.to_csv(metrics_out_fn)

    agg_metrics = metrics.drop(columns=["slate_id", "personalized"]).mean()
    # reciprocal rank mean to MRR
    agg_metrics = agg_metrics.rename(index={"RR": "MRR"})

    # issue one warning about lots of missing truth
    num_bad = np.sum(metrics["num_truth"] == 0)
    frac_bad = num_bad / len(metrics)
    if frac_bad >= 0.1:
        logging.warning(
            "%.1f%% (%s/%s) of eval slates have no truth",
            frac_bad * 100,
            metric(num_bad),
            metric(len(metrics)),
        )
    else:
        logging.info(
            "%.1f%% (%s/%s) of eval slates have no truth",
            frac_bad * 100,
            metric(num_bad),
            metric(len(metrics)),
        )

    out_fn = out_dir / "metrics.json"
    logger.info("saving evaluation to %s", out_fn)
    with open(out_fn, "wt") as jsf:
        print(agg_metrics.to_json(indent=2), file=jsf)

    # set up to print summary metrics, filtering if needed
    printable = agg_metrics.to_dict()
    if not options.evaluate.print_effectiveness:
        printable = {k: v for (k, v) in printable.items() if not EFF_METRIC_NAMES.match(k)}

    print("[bold]Summary of Metrics:[/bold]", end=" ")
    pprint(printable, expand_all=True)

    if cli_opts["--section-eval"]:
        _run_section_eval(out_dir, eval_data, options)


def _run_section_eval(out_dir, eval_data, options):
    json_fn = out_dir / "recommendations.ndjson.zst"
    if not json_fn.exists():
        logger.error(
            "section eval requires %s, which does not exist; " "re-run generation with a pipeline that writes NDJSON",
            json_fn,
        )
        return

    logger.info("running section-based evaluation from %s", json_fn)
    section_metric_records = []

    with zstandard.open(json_fn, "rt") as jf:
        for line in jf:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            slate_id = UUID(data["slate_id"])

            final = data["results"]["final"]
            if isinstance(final, list):
                sections = [ImpressedSection.model_validate(s) for s in final]
            else:
                sections = [ImpressedSection.model_validate(final)]

            truth = eval_data.slate_truth(slate_id)
            if truth is None:
                logger.debug("slate %s has no truth, skipping section eval", slate_id)
                continue

            row = measure_section_rec_metrics(slate_id, sections, truth, eval_data)
            section_metric_records.append(row)

    if not section_metric_records:
        logger.warning("no section metrics produced; check that truth data aligns with the pipeline output")
        return

    section_metrics = pd.DataFrame.from_records(section_metric_records)
    logger.info("measured section metrics for %d slates", section_metrics["slate_id"].nunique())

    section_metrics_out_fn = out_dir / "section-recommendation-metrics.csv.gz"
    logger.info("saving per-slate section metrics to %s", section_metrics_out_fn)
    section_metrics.to_csv(section_metrics_out_fn)

    agg_section_metrics = section_metrics.drop(columns=["slate_id"]).mean(numeric_only=True)

    section_out_fn = out_dir / "section-metrics.json"
    logger.info("saving section evaluation summary to %s", section_out_fn)
    with open(section_out_fn, "wt") as jsf:
        print(agg_section_metrics.to_json(indent=2), file=jsf)

    print("[bold]Section Metrics Summary:[/bold]", end=" ")
    pprint(agg_section_metrics.to_dict(), expand_all=True)


if __name__ == "__main__":
    main()
