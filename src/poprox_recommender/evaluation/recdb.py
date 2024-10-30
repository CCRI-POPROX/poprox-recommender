"""
Interface to database storing recommendation results.
"""

from os import PathLike
from pathlib import Path

from poprox_concepts.api.recommendations import RecommendationRequest
from poprox_recommender.pipeline import PipelineState

RLW_TRANSACTION_SIZE = 1000


class RecListWriter:
    """
    Writer that stores recommendation lists in a database.

    Writers can be used as context managers (in a ``with`` block), and
    automatically finish the write and close the database when the block exits
    successfully.
    """

    path: Path
    _current_batch_size: int = 0

    def __init__(self, path: PathLike[str]):
        self.path = Path(path)

    def store_results(self, name: str, request: RecommendationRequest, pipeline_state: PipelineState) -> None:
        pass

    def _commit(self):
        pass

    def finish(self):
        "Finish writing.  The database remains open."
        pass

    def close(self):
        "Close the database."
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type:
            pass
        else:
            self.finish()

        self.close()
