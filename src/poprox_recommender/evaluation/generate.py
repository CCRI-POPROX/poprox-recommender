"""
Generate recommendations for offline test data.

Usage:
    poprox_recommender.evaluation.generate [options]

Options:
    -v, --verbose
            enable verbose diagnostic logs
    --log-file=FILE
            write log messages to FILE
    -o FILE, --output=FILE
            write output to FILE [default: outputs/recommendations.duckdb]
    -M DATA, --mind-data=DATA
            read MIND test data DATA [default: MINDsmall_dev]
    --subset=N
            test only on the first N test users
"""

# pyright: basic
from __future__ import annotations

import itertools as it
import logging
import logging.config

from docopt import docopt
from lenskit.util import Stopwatch
from progress_api import make_progress

from poprox_concepts.domain import ArticleSet
from poprox_recommender.config import default_device
from poprox_recommender.data.mind import TEST_REC_COUNT, MindData
from poprox_recommender.evaluation.recdb import RecListWriter
from poprox_recommender.logging_config import setup_logging
from poprox_recommender.recommenders import recommendation_pipelines

logger = logging.getLogger("poprox_recommender.evaluation.evaluate")

# long-term TODO:
# - support other MIND data (test?)
# - support our data

STAGES = ["final", "ranked", "reranked"]


def generate_user_recs(dataset: str, out: RecListWriter, n_users: int | None = None):
    mind_data = MindData(dataset)

    pipelines = recommendation_pipelines(device=default_device())

    logger.info("generating recommendations")

    user_iter = mind_data.iter_users()
    if n_users is None:
        n_users = mind_data.n_users
        logger.info("recommending for all %d users", n_users)
    else:
        logger.info("running on subset of %d users", n_users)
        user_iter = it.islice(user_iter, n_users)

    timer = Stopwatch()
    with make_progress(logger, "recommend", total=n_users) as pb:
        for request in user_iter:  # one by one
            logger.debug("recommending for user %s", request.interest_profile.profile_id)
            if request.num_recs != TEST_REC_COUNT:
                logger.warn(
                    "request for %s had unexpected recommendation count %d",
                    request.interest_profile.profile_id,
                    request.num_recs,
                )
            inputs = {
                "candidate": ArticleSet(articles=request.todays_articles),
                "clicked": ArticleSet(articles=request.past_articles),
                "profile": request.interest_profile,
            }
            for name, pipe in pipelines.items():
                try:
                    outputs = pipe.run_all(**inputs)
                except Exception as e:
                    logger.error("error recommending for user %s: %s", request.interest_profile.profile_id, e)
                    raise e
                out.store_results(name, request, outputs)

            pb.update()

    timer.stop()
    logger.info("finished recommending in %s", timer)


if __name__ == "__main__":
    """
    For offline evaluation, set theta in mmr_diversity = 1
    """
    options = docopt(__doc__)  # type: ignore
    setup_logging(verbose=options["--verbose"], log_file=options["--log-file"])

    n_users = options["--subset"]
    if n_users is not None:
        n_users = int(n_users)

    out_fn = options["--output"]
    logger.info("storing recommendations in %s", out_fn)
    with RecListWriter(out_fn) as out:
        generate_user_recs(options["--mind-data"], out, n_users)
