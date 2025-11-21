"""
Generate recommendations for offline test data.

Usage:
    poprox_recommender.evaluation.generate [options] PIPELINE

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
    --subset=N
            test only on the first N test requests
    PIPELINE
            The name of the pipeline to generate from
"""

import logging
import os
import sys
from pathlib import Path

import lenskit
from docopt import docopt
from lenskit.logging import LoggingConfig

from poprox_recommender.data.mind import MindData
from poprox_recommender.data.poprox import PoproxData
from poprox_recommender.evaluation.generate.outputs import RecOutputs
from poprox_recommender.evaluation.generate.worker import generate_recs_for_requests
from poprox_recommender.paths import project_root

logger = logging.getLogger("poprox_recommender.evaluation.generate")


def generate_main():
    options = docopt(__doc__)  # type: ignore
    log_cfg = LoggingConfig()
    # turn on verbose logging when GitHub actions run in debug mode
    if options["--verbose"] or os.environ.get("RUNNER_DEBUG", 0):
        log_cfg.set_verbose(True)
    if options["--log-file"]:
        log_cfg.set_log_file(options["--log-file"])
    log_cfg.apply()
    lenskit.configure(cfg_dir=project_root())

    out_path = Path(options["--output-path"])
    out_path.mkdir(exist_ok=True, parents=True)
    outputs = RecOutputs(out_path)

    n_requests = options["--subset"]
    if n_requests is not None:
        n_requests = int(n_requests)

    if options["--poprox-data"]:
        dataset = PoproxData(options["--poprox-data"])
    elif options["--mind-data"]:
        dataset = MindData(options["--mind-data"])
    else:
        logger.error("must specify a data source")
        sys.exit(2)

    pipe_name = options["PIPELINE"]

    logger.info("preparing to generate for pipeline %s", pipe_name)
    generate_recs_for_requests(dataset, outputs, pipe_name, n_requests)


if __name__ == "__main__":
    generate_main()
