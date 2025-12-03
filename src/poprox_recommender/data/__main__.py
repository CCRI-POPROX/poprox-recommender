"""
Query POPROX evaluation data.

Usage:
    poprox_recommender.data (-P NAME | -M NAME) --dump-request=SLATE

Options:
    -P NAME, --poprox-data=NAME
        Load from POPROX data set NAME.
    -M NAME, --mind-data=NAME
        Load from MIND datasaet NAME.
    --dump-request=SLATE
        Dump the request named by SLATE. SLATE can either be a UUID or an
        integer; integers count evaluation slates in iteration order.
"""

# pyright: basic
from __future__ import annotations

import re
import sys
from itertools import islice
from uuid import UUID

from lenskit.logging import basic_logging, get_logger

from .mind import MindData
from .poprox import PoproxData

logger = get_logger(__name__)


def main():
    from docopt import docopt

    basic_logging()
    opts = docopt(__doc__ or "")

    if poprox := opts["--poprox-data"]:
        data = PoproxData(poprox)
    elif mind := opts["--mind-data"]:
        data = MindData(mind)

    if slate := opts["--dump-request"]:
        if re.match(r"^\d+$", slate):
            logger.info("finding slate number %s", slate)
            siter = islice(data.iter_slate_ids(), int(slate), None)
            slate = next(siter)
        else:
            slate = UUID(slate)
        logger.info("fetching requst for slate %s", slate)
        req = data.lookup_request(slate)
        print(req.model_dump_json(indent=2))
    else:
        logger.error("no action specified")
        sys.exit(2)


if __name__ == "__main__":
    main()
