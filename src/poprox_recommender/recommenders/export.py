"""
Export POPROX recommender configurations to JSON.

Usage:
    poprox_recommender.recommenders.export [-v] [-o FILE] PIPELINE

Options:
    -v, --verbose
        Enable verbose logging.
    -o FILE, --output=FILE
        Write output to FILE.
    PIPELINE
        Name of the recommender pipeline to export.
"""

import lenskit
from docopt import docopt
from lenskit.logging import LoggingConfig, get_logger, stdout_console

from poprox_recommender.paths import project_root

from .load import get_pipeline

logger = get_logger("poprox_recommender.recommenders.export")


def main():
    cli_opts = docopt(__doc__)  # type: ignore
    stdout = stdout_console()
    log_cfg = LoggingConfig()
    if cli_opts["--verbose"]:
        log_cfg.set_verbose(True)
    log_cfg.apply()
    lenskit.configure(cfg_dir=project_root())

    name = cli_opts["PIPELINE"]
    logger.info("Loading pipeline %s", name)

    pipe = get_pipeline(name)
    cfg = pipe.config

    if fn := cli_opts["--output"]:
        logger.info("saving pipeline configuration to %s", fn)
        with open(fn, "wt") as jsf:
            jsf.write(cfg.model_dump_json(indent=2))
            jsf.write("\n")

    else:
        stdout.print_json(cfg.model_dump_json())


if __name__ == "__main__":
    main()
