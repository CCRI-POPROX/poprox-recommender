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
            test only on the first N test profiles
    PIPELINE
            The name of the pipeline to generate from
"""

import logging
import os
from pathlib import Path

from docopt import docopt
from lenskit.logging import LoggingConfig

from poprox_recommender.data.mind import MindData
from poprox_recommender.data.poprox import PoproxData
from poprox_recommender.evaluation.generate.outputs import RecOutputs
from poprox_recommender.evaluation.generate.worker import generate_profile_recs

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

    if options["--poprox-data"]:
        dataset = PoproxData(options["--poprox-data"])
    elif options["--mind-data"]:
        dataset = MindData(options["--mind-data"])

    pipe_name = options["PIPELINE"]

    generate_profile_recs(dataset, outputs, pipe_name, n_profiles)


if __name__ == "__main__":
    generate_main()
