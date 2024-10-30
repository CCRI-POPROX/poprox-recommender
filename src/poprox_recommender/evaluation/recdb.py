"""
Interface to database storing recommendation results.
"""

from logging import getLogger
from os import PathLike
from pathlib import Path
from uuid import UUID

import duckdb

from poprox_concepts import ArticleSet
from poprox_concepts.api.recommendations import RecommendationRequest
from poprox_recommender.lkpipeline import PipelineState

RLW_TRANSACTION_SIZE = 1000
SCHEMA_DIR = Path(__file__).parent

logger = getLogger(__name__)


class RecListWriter:
    """
    Writer that stores recommendation lists in a database.

    Writers can be used as context managers (in a ``with`` block), and
    automatically finish the write and close the database when the block exits
    successfully.
    """

    path: Path
    db: duckdb.DuckDBPyConnection
    _current_batch_size: int = 0

    def __init__(self, path: PathLike[str]):
        self.path = Path(path)
        self.db = duckdb.connect(self.path)
        self._init_db()

    def store_results(self, name: str, request: RecommendationRequest, pipeline_state: PipelineState) -> None:
        if self._current_batch_size == 0:
            logger.debug("beginning database transaction")
            self.db.begin()

        user = request.interest_profile.profile_id
        assert user is not None, "no user ID specified"

        recs = pipeline_state["recommender"]
        assert isinstance(recs, ArticleSet)
        self._store_list(name, user, "final", recs)

        ranked = pipeline_state.get("ranker", None)
        if ranked is not None:
            assert isinstance(ranked, ArticleSet)
            self._store_list(name, user, "ranked", ranked)

        reranked = pipeline_state["reranker"]
        if reranked is not None:
            assert isinstance(reranked, ArticleSet)
            self._store_list(name, user, "ranked", reranked)

        self._current_batch_size += 1
        if self._current_batch_size >= RLW_TRANSACTION_SIZE:
            logger.debug("commiting result batch")
            self.db.commit()
            self._current_batch_size = 0

    def _init_db(self):
        "Initialize the database schema."
        sql_file = SCHEMA_DIR / "rec-schema.sql"

        logger.info("applying schema script %s", sql_file)
        sql = sql_file.read_text()
        self.db.execute(sql)

    def _store_list(self, name: str, user: UUID, stage: str, recs: ArticleSet):
        self.db.execute(
            """
            INSERT INTO rec_list_meta (recommender, user_id, stage)
            VALUES (?, ?, ?)
            RETURNING rl_id
            """,
            [name, user, stage],
        )
        row = self.db.fetchone()
        assert row is not None, "failed to fetch insert result"
        (rl_id,) = row

        scores = getattr(recs, "scores", None)
        self.db.executemany(
            """
            INSERT INTO rec_list_articles (rl_id, rank, article_id, score)
            VALUES (?, ?, ?, ?)
            """,
            [
                [rl_id, i + 1, a.article_id, scores[i] if scores is not None else None]
                for (i, a) in enumerate(recs.articles)
            ],
        )

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
