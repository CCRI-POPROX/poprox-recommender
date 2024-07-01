import os
import sys
from pathlib import Path

test_dir = Path(__file__).parent
root_dir = test_dir.parent.resolve()
src_dir = root_dir / "src"

try:
    import poprox_recommender  # noqa: F401
except ImportError:
    # tweak up paths so pytest can work without installing the recommender
    sys.path.insert(0, os.fspath(src_dir))
    os.environ["PYTHONPATH"] = os.fspath(src_dir)
