"""
Collect offline metrics from different pipelines for a run.

For an evaluation EVAL, this script collects metrics from
outputs/EVAL/*/metrics.json, and outputs them to outputs/EVAL-metrics.csv.

Usage:
    poprox_recommender.evaluation.collect [options] EVAL

Options:
    -v, --verbose
            enable verbose diagnostic logs
    --log-file=FILE
            write log messages to FILE
    EVAL    the name of the evaluation to aggregate
"""

import json
import os
from pathlib import Path

import pandas as pd
from docopt import docopt
from lenskit.logging import LoggingConfig, get_logger

logger = get_logger("poprox_recommender.evaluation.collect")


def main():
    options = docopt(__doc__ or "")
    log_cfg = LoggingConfig()
    if options["--verbose"] or os.environ.get("RUNNER_DEBUG", 0):
        log_cfg.set_verbose(True)
    if options["--log-file"]:
        log_cfg.set_log_file(options["--log-file"])
    log_cfg.apply()

    name = options["EVAL"]
    assert isinstance(name, str)
    path = Path("outputs") / name

    results = {}
    for mf in path.glob("*/metrics.json"):
        pipe = mf.parent.name
        logger.info("reading pipeline metrics", pipeline=pipe, path=mf)
        metrics = json.loads(mf.read_text())
        results[pipe] = metrics

    rdf = pd.DataFrame.from_dict(results)
    rdf.index.rename("pipeline", inplace=True)
    csv_out = path.parent / f"{name}-metrics.csv"
    logger.info("saving metrics for %d pipelines", len(results), file=str(csv_out))
    rdf.to_csv(csv_out, index=True)


if __name__ == "__main__":
    main()
