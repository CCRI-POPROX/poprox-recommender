"""
Load recommendation outputs for measurement.
"""

# pyright: strict
from __future__ import annotations

from collections.abc import Iterable
from logging import getLogger
from os import fspath
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from uuid import UUID

import duckdb
import pandas as pd

logger = getLogger(__name__)


class RecLoader:
    """
    This class loads recommendation runs for measurement.  It works with a
    read-only DuckDB connection, like the eval data loaders, and can be
    efficiently sent to Ray workers.

    This class works as a context manager, closing the DB connection (and
    cleaning up the temp directory, if appropriate) at exit.
    """

    path: Path
    db: duckdb.DuckDBPyConnection
    _dir: TemporaryDirectory[str] | None = None

    def __init__(self, path: Path):
        """
        Open a results database at a path.
        """
        self.path = path
        logger.debug("opening results database %s", path)
        self.db = duckdb.connect(path, read_only=True)

    @classmethod
    def from_parquet(cls, path: Path):
        """
        Create a results database for recommendations in a Parquet file.
        """
        td = TemporaryDirectory(prefix="poprox-eval.")
        try:
            logger.debug("initializing recommendation DB in temp dir %s", td)
            db_path = Path(td.name) / "recommendations.db"
            db = duckdb.connect(db_path)
            db.execute(
                """
                CREATE TABLE recs (
                    slate_id UUID NOT NULL,
                    stage VARCHAR NOT NULL,
                    item_id UUID NOT NULL,
                    rank INTEGER NOT NULL,
                )
                """
            )
            db.execute(
                f"""
                INSERT INTO recs
                SELECT slate_id, stage, item_id, rank
                FROM '{fspath(path)}'
                ORDER BY slate_id, stage, rank
                """
            )
            db.read_parquet(fspath(path)).insert_into("recs")
            db.execute("CREATE INDEX rec_slate_idx ON recs (slate_id)")
            db.close()
            loader = cls(db_path)
            loader._dir = td
        except Exception as e:
            # manually clean up exception because we're going to hand it off
            td.cleanup()
            raise e

        return loader

    def count_slates(self) -> int:
        self.db.execute("SELECT COUNT(DISTINCT slate_id) FROM recs")
        (n,) = self.db.fetchone() or [0]
        return n

    def iter_slate_ids(self) -> Iterable[UUID]:
        with self.db.cursor() as cur:
            cur.execute("SELECT DISTINCT(slate_id) FROM recs")
            while row := cur.fetchone():
                (slate,) = row
                assert isinstance(slate, UUID)
                yield slate

    def slate_recs(self, slate_id: UUID) -> pd.DataFrame:
        self.db.execute(
            """
            SELECT slate_id::VARCHAR AS slate_id, stage, item_id::VARCHAR AS item_id, rank
            FROM recs
            WHERE slate_id = ?
            ORDER BY stage, rank
            """,
            [slate_id],
        )
        return self.db.fetch_df()

    def __reduce__(self):
        # we want to unpickle by just calling the constructor on the path
        return (RecLoader, (self.path,))

    def __enter__(self) -> RecLoader:
        return self

    def __exit__(self, *args: Any):
        self.db.close()
        if self._dir is not None:
            logger.debug("cleaning up temporary directory %s", self._dir)
            self._dir.cleanup()
