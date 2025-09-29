"""
Collect offline metrics from different pipelines for a run.

For an evaluation EVAL, this script collects metrics from
outputs/EVAL/*/metrics.json, and outputs them to outputs/EVAL-metrics.csv.
It likewise aggregates the profile metrics.

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
from lenskit.logging import LoggingConfig, Task, get_logger

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

    agg_results = {}
    for mf in sorted(path.glob("*/metrics.json"), key=lambda p: p.as_posix()):
        pipe = mf.parent.name
        logger.info("reading pipeline metrics", pipeline=pipe, path=mf)
        metrics = json.loads(mf.read_text())
        agg_results[pipe] = metrics

    rdf = pd.DataFrame.from_dict(agg_results, "index")
    rdf.index.name = "pipeline"
    csv_out = path.parent / f"{name}-metrics.csv"
    logger.info("saving metrics for %d pipelines", len(agg_results), file=str(csv_out))
    rdf.to_csv(csv_out, index=True)

    gen_stats = {}
    for tf in sorted(path.glob("*/generate-task.json")):
        task = Task.model_validate_json(tf.read_text())
        gen_stats[tf.parent.name] = {
            "Machine": task.machine,
            "Stage": "generate",
            "WallTime": task.duration,
            "CPUTime": task.total_cpu(),
            "Power": task.system_power,
            "CPUPower": task.cpu_power,
            "GPUPower": task.gpu_power,
        }

    tdf = pd.DataFrame.from_dict(gen_stats, "index")
    tdf.index.name = "pipeline"
    csv_out = path.parent / f"{name}-tasks.csv"
    logger.info("saving task statistics to %s", csv_out)
    tdf.to_csv(csv_out, index=True)

    prof_results = {}
    for mf in sorted(path.glob("*/profile-metrics.csv.gz"), key=lambda p: p.as_posix()):
        pipe = mf.parent.name
        logger.info("reading pipeline profile metrics", pipeline=pipe, path=mf)
        metrics = pd.read_csv(mf).set_index("profile_id")
        prof_results[pipe] = metrics

    prof_df = pd.concat(prof_results, names=["pipeline"])
    csv_out = path.parent / f"{name}-profile-metrics.csv.gz"
    logger.info("saving profile metrics for %d pipelines", len(prof_results), file=str(csv_out))
    prof_df.to_csv(csv_out, index=True)


if __name__ == "__main__":
    main()
