"""
Interface to database storing recommendation results.
"""

import os
from logging import getLogger
from os import PathLike
from pathlib import Path
from typing import NamedTuple
from uuid import UUID

import duckdb
import numpy as np
import pandas as pd

from poprox_concepts import ArticleSet
from poprox_concepts.api.recommendations import RecommendationRequest
from poprox_recommender.lkpipeline import PipelineState

RLW_TRANSACTION_SIZE = 2500
SCHEMA_DIR = Path(__file__).parent

logger = getLogger(__name__)


class RecListMeta(NamedTuple):
    rl_id: int
    recommender: str
    user_id: UUID
    stage: str


class RecListWriter:
    """
    Writer that stores recommendation lists in a database.

    Writers can be used as context managers (in a ``with`` block), and
    automatically finish the write and close the database when the block exits
    successfully.
    """

    path: Path
    db: duckdb.DuckDBPyConnection
    batch_size: int = RLW_TRANSACTION_SIZE
    _current_batch_meta: list[RecListMeta]
    _current_batch_articles: list[pd.DataFrame]
    _last_rl_id: int = 0

    def __init__(self, path: PathLike[str]):
        self.path = Path(path)
        self.db = duckdb.connect(self.path)
        self._init_db()

        if "POPROX_BATCH_SIZE" in os.environ:
            self.batch_size = int(os.environ["POPROX_BATCH_SIZE"])

        self._current_batch_meta = []
        self._current_batch_articles = []

    def store_results(self, name: str, request: RecommendationRequest, pipeline_state: PipelineState) -> None:
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

        if len(self._current_batch_meta) >= self.batch_size:
            self._save_batch()

    def _init_db(self):
        "Initialize the database schema."
        sql_file = SCHEMA_DIR / "rec-schema.sql"

        logger.info("applying schema script %s", sql_file)
        sql = sql_file.read_text()
        self.db.execute(sql)

    def _store_list(self, name: str, user: UUID, stage: str, recs: ArticleSet):
        self._last_rl_id += 1
        rl_id = self._last_rl_id

        self._current_batch_meta.append(RecListMeta(rl_id, name, user, stage))

        if recs.articles:
            rows = pd.DataFrame(
                {
                    "rl_id": rl_id,
                    "rank": np.arange(1, len(recs.articles) + 1),
                    "article_id": [a.article_id for a in recs.articles],
                }
            )
            scores = getattr(recs, "scores", None)
            if scores is not None:
                rows["score"] = scores
            self._current_batch_articles.append(rows)
        else:
            logger.debug("user %s has empty list for stage %s", user, stage)

    def _save_batch(self):
        if not self._current_batch_meta:
            return

        meta_df = pd.DataFrame.from_records(self._current_batch_meta)
        article_df = pd.concat(self._current_batch_articles, ignore_index=True)
        aq_cols = ", ".join(article_df.columns)
        a_query = f"INSERT INTO rec_list_articles ({aq_cols}) SELECT * FROM article_df"
        logger.debug("saving batch of %d lists", len(meta_df))

        self.db.begin()
        try:
            self.db.from_df(meta_df).insert_into("rec_list_meta")
            self.db.execute(a_query)
            self.db.commit()
        except Exception as e:
            self.db.rollback()
            raise e

        self._current_batch_meta = []
        self._current_batch_articles = []

    def finish(self):
        "Finish writing.  The database remains open."
        self._save_batch()

        sql_file = SCHEMA_DIR / "rec-index.sql"

        logger.info("applying index script %s", sql_file)
        sql = sql_file.read_text()
        self.db.execute(sql)

    def close(self):
        "Close the database."
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type:
            pass
        else:
            self.finish()

        self.close()
