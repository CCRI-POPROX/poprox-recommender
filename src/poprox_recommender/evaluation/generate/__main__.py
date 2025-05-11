"""
Generate recommendations for offline test data.

Usage:
    poprox_recommender.evaluation.generate [options]

Options:
    -v, --verbose
            enable verbose diagnostic logs
    --log-file=FILE
            write log messages to FILE
    -o PATH, --output-path=PATH
            write output to PATH [default: outputs/]
    -M DATA, --mind-data=DATA
            read MIND test data DATA [default: MINDsmall_dev]
    -P DATA, --poprox-data=DATA
            read POPROX test data DATA
    -j N, --jobs=N
            use N parallel jobs
    --subset=N
            test only on the first N test profiles
    --start_date=START_DATE
            regenerate newsletters on and after START_DATE in the form mm/dd/yyyy
    --end_date=END_DATE
            regenerate newsleters before END_DATE in the form mm/dd/yyyy
    --click_threshold=N
            test only profiles with N clicks from start_date to end_date
    --topic_thetas=TUPLE
            test all theta values in TUPLE in the form (start, end)
    --locality_thetas=TUPLE
            test all theta values in TUPLE in the form (start, end)
    --similarity_thresholds=TUPLE
            test all threshold values in TUPLE in the form (start, end)
    --pipelines=<pipelines>...
            list of pipeline names (separated by spaces)
"""

import ast
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
from docopt import docopt
from lenskit.logging import LoggingConfig

# from poprox_recommender.config import available_cpu_parallelism
from poprox_recommender.data.mind import MindData
from poprox_recommender.data.poprox import PoproxData
from poprox_recommender.evaluation.generate.outputs import RecOutputs
from poprox_recommender.evaluation.generate.worker import generate_profile_recs
from poprox_recommender.rusage import pretty_time

logger = logging.getLogger("poprox_recommender.evaluation.generate")


def generate_main():
    """
    For offline evaluation, set theta in mmr_diversity = 1
    """
    options = docopt(__doc__)  # type: ignore
    log_cfg = LoggingConfig()
    # turn on verbose logging when GitHub actions run in debug mode
    if options["--verbose"] or os.environ.get("RUNNER_DEBUG", 0):
        log_cfg.set_verbose(True)
    if options["--log-file"]:
        log_cfg.set_log_file(options["--log-file"])
    log_cfg.apply()

    out_path = Path(options["--output-path"])
    outputs = RecOutputs(out_path)

    n_profiles = options["--subset"]
    if n_profiles is not None:
        n_profiles = int(n_profiles)

    n_jobs = options["--jobs"]
    if n_jobs is not None:
        n_jobs = int(n_jobs)
        if n_jobs <= 0:
            logger.warning("--jobs must be positive, using single job")
            n_jobs = 1
    else:
        # n_jobs = available_cpu_parallelism(4)
        n_jobs = 1

    # parse start and end dates
    start_date = None
    end_date = None
    if options["--start_date"]:
        start_date = datetime.strptime(options["--start_date"], "%m/%d/%Y")
    if options["--end_date"]:
        end_date = datetime.strptime(options["--end_date"], "%m/%d/%Y")

    # Ok if None
    topic_thetas = options["--topic_thetas"]
    locality_thetas = options["--locality_thetas"]
    similarity_thresholds = options["--similarity_thresholds"]

    topic_thetas = ast.literal_eval(topic_thetas) if topic_thetas else None
    locality_thetas = ast.literal_eval(locality_thetas) if locality_thetas else None
    similarity_thresholds = ast.literal_eval(similarity_thresholds) if similarity_thresholds else None

    if (topic_thetas or locality_thetas) and similarity_thresholds:
        raise ValueError(
            "You cannot set 'similarity_thresholds' when 'topic_thetas' or 'locality_thetas' is provided. "
        )

    # subset pipelines
    if options["--poprox-data"]:
        dataset = PoproxData(options["--poprox-data"], start_date, end_date)
    elif options["--mind-data"]:
        dataset = MindData(options["--mind-data"])

    pipelines = None
    if options["--pipelines"]:
        pipelines = options["--pipelines"]
        if isinstance(pipelines, str):
            pipelines = [pipelines]
        logger.info("generating pipelines: %s", pipelines)

    worker_usage = generate_profile_recs(
        dataset, outputs, n_profiles, n_jobs, topic_thetas, locality_thetas, similarity_thresholds, pipelines=pipelines
    )

    logger.info("de-duplicating embeddings")
    emb_df = pd.read_parquet(outputs.emb_temp_dir)
    n = len(emb_df)
    emb_df = emb_df.drop_duplicates(subset="article_id")
    logger.info("keeping %d of %d total embeddings", len(emb_df), n)
    emb_df.to_parquet(out_path / "embeddings.parquet", compression="zstd")
    logger.debug("removing temporary embedding files")
    shutil.rmtree(outputs.emb_temp_dir)

    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        cpu = usage.ru_stime + usage.ru_utime
        logger.info("parent process used %s CPU time", pretty_time(cpu))
        if worker_usage:
            child_tot = sum(u.ru_stime + u.ru_utime for u in worker_usage)
            logger.info("worker processes used a combined %s CPU time", pretty_time(child_tot))
            logger.info("total CPU time: %s", pretty_time(cpu + child_tot))
    except ImportError:
        logger.warning("resource usage only available on Unix")


if __name__ == "__main__":
    generate_main()
