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
    -j N, --jobs=N
            use N parallel jobs
    --subset=N
            test only on the first N test users
"""

import logging
import shutil
from pathlib import Path

import pandas as pd
from docopt import docopt

from poprox_recommender.config import available_cpu_parallelism
from poprox_recommender.evaluation.generate.outputs import RecOutputs
from poprox_recommender.evaluation.generate.worker import generate_user_recs
from poprox_recommender.logging_config import setup_logging
from poprox_recommender.rusage import pretty_time

logger = logging.getLogger("poprox_recommender.evaluation.generate")


def generate_main():
    """
    For offline evaluation, set theta in mmr_diversity = 1
    """
    options = docopt(__doc__)  # type: ignore
    setup_logging(verbose=options["--verbose"], log_file=options["--log-file"])

    out_path = Path(options["--output-path"])
    outputs = RecOutputs(out_path)

    n_users = options["--subset"]
    if n_users is not None:
        n_users = int(n_users)

    n_jobs = options["--jobs"]
    if n_jobs is not None:
        n_jobs = int(n_jobs)
        if n_jobs <= 0:
            logger.warning("--jobs must be positive, using single job")
            n_jobs = 1
    else:
        n_jobs = available_cpu_parallelism(4)

    worker_usage = generate_user_recs(options["--mind-data"], outputs, n_users, n_jobs)

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
