# PyTest runs this file before even attempting to import any of the test cases,
# so in addition to dynamic PyTest configuration, we can also put code to set
# up import paths and things like that here.
import os
import sys
from pathlib import Path

import structlog

test_dir = Path(__file__).parent
root_dir = test_dir.parent.resolve()
src_dir = root_dir / "src"

try:
    import poprox_recommender  # noqa: F401
except ImportError:
    # tweak up paths so pytest can work without installing the recommender
    sys.path.insert(0, os.fspath(src_dir))
    # put in `PYTHONPATH` too so that serverless can find our python code
    os.environ["PYTHONPATH"] = os.fspath(src_dir)

# set up structlog to dump to standard logging
# TODO: enable JSON logs
structlog.configure(
    [
        structlog.processors.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.MaybeTimeStamper(),
        structlog.processors.KeyValueRenderer(key_order=["event", "timestamp"]),
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
)
