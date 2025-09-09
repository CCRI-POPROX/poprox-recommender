"""
Support for loading MIND_ data for evaluation.

.. _MIND: https://msnews.github.io/
"""

# pyright: basic
from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator, Literal
from uuid import NAMESPACE_URL, UUID, uuid5

import duckdb
import pandas as pd
from duckdb import DuckDBPyConnection

from poprox_concepts import Article, Click, Entity, InterestProfile, Mention
from poprox_concepts.api.recommendations import RecommendationRequest
from poprox_recommender.data.eval import EvalData
from poprox_recommender.paths import project_root

logger = logging.getLogger(__name__)
TEST_REC_COUNT = 10
NAMESPACE_ARTICLE = uuid5(NAMESPACE_URL, "https://data.poprox.io/mind/article/")
NAMESPACE_IMPRESSION = uuid5(NAMESPACE_URL, "https://data.poprox.io/mind/impression/")


class MindData(EvalData):
    """
    News and behavior data loaded from MIND data.
    """

    name: str
    "Name of the MIND dataset."
    path: Path
    "Path to the MIND database."
    duck: DuckDBPyConnection
    "DuckDB connection."

    _article_count: int
    _impression_count: int

    def __init__(self, archive: str = "MINDsmall_dev"):
        self.name = archive
        self.path = project_root() / "data" / f"{archive}.db"

        self.duck = duckdb.connect(self.path, read_only=True)

        # pre-fetch counts
        self.duck.execute("SELECT COUNT(*) FROM impressions")
        (n,) = self.duck.fetchone() or (0,)
        self._impression_count = n

        self.duck.execute("SELECT COUNT(*) FROM articles")
        (n,) = self.duck.fetchone() or (0,)
        self._article_count = n

    def news_uuid_for_id(self, id: str) -> UUID:
        return uuid5(NAMESPACE_ARTICLE, id)

    def behavior_uuid_for_id(self, id: str) -> UUID:
        return uuid5(NAMESPACE_IMPRESSION, id)

    @property
    def n_profiles(self) -> int:
        return self._impression_count

    @property
    def n_articles(self) -> int:
        return self._article_count

    def profile_truth(self, user: UUID) -> pd.DataFrame | None:
        """
        Look up the ground-truth data for a particular user profile,
        in LensKit format with item UUIDs for item IDs.
        """

        self.duck.execute(
            """
            SELECT article_uuid AS item_id,
                'N' || article_id AS mind_item_id,
                CAST(clicked AS INT2) AS rating
            FROM impressions
            JOIN impression_articles USING (imp_id)
            JOIN articles USING (article_id)
            WHERE imp_uuid = ?
            """,
            [user],
        )
        truth = self.duck.fetch_df()
        return truth.set_index("item_id")

    def iter_profiles(self) -> Generator[RecommendationRequest]:
        # collect the with larger impression sets
        logger.info("querying for test articles")

        # we use 2 queries: an outer query to list the impression IDs, and inner
        # queries to get the articles and article data.  outer query is in a cloned
        # connection so that they don't interfere with each other.
        with self.duck.cursor() as clone:
            clone.execute("SELECT imp_id, imp_uuid FROM impressions")

            # loop over the results and yield the recommendations
            logger.info("iterating test articles")
            while row := clone.fetchone():
                imp_id, imp_uuid = row

                # get the historical articles and click list
                past = self.lookup_articles(imp_id, relation="history")
                clicks = [Click(article_id=a.article_id) for a in past]

                # get the candidate articles
                today = self.lookup_articles(imp_id, relation="candidate")

                # FIXME the profile ID should probably be the user ID
                profile = InterestProfile(profile_id=imp_uuid, click_history=clicks, onboarding_topics=[])
                yield RecommendationRequest(
                    todays_articles=today, past_articles=past, interest_profile=profile, num_recs=TEST_REC_COUNT
                )

    def lookup_articles(self, imp_id: int, *, relation: Literal["history", "candidate"]) -> list[Article]:
        # run the query for the articles we're looking for
        if relation == "history":
            self.duck.execute(
                """
                SELECT article_uuid, category, subcategory, title
                FROM impression_history
                JOIN articles USING (article_id)
                WHERE imp_id = ?
                """,
                [imp_id],
            )
        elif relation == "candidate":
            self.duck.execute(
                """
                SELECT article_uuid, category, subcategory, title
                FROM impressions
                JOIN impression_articles USING (imp_id)
                JOIN impression_article_summaries USING (article_id)
                JOIN articles USING (article_id)
                WHERE imp_id = ?
                AND first_day <= imp_day
                AND last_day > imp_day - 7
                """,
                [imp_id],
            )

        articles = []
        while row := self.duck.fetchone():
            articles.append(self._make_article(*row))

        return articles

    def lookup_article(self, *, id: str | None = None, uuid: UUID | None = None):
        if uuid is not None:
            self.duck.execute(
                """
                SELECT article_uuid, category, subcategory, title
                FROM articles
                WHERE article_uuid = ?
                """,
                [uuid],
            )
        elif id is not None:
            assert id[0] == "N"
            id_num = int(id[1:])
            self.duck.execute(
                """
                SELECT article_uuid, category, subcategory, title
                FROM articles
                WHERE article_id = ?
                """,
                [id_num],
            )
        else:
            raise ValueError("must provide one of uuid or id")

        row = self.duck.fetchone()
        if row:
            return self._make_article(*row)
        else:
            raise KeyError(uuid or id)

    def _make_article(self, uuid, category, subcategory, title) -> Article:
        category = Entity(name=category, entity_type="category", source="MIND")
        subcategory = Entity(name=subcategory, entity_type="subcategory", source="MIND")

        return Article(
            article_id=uuid,
            url=f"urn:uuid:{uuid}",
            headline=title,
            mentions=[Mention(source="MIND", relevance=1, entity=entity) for entity in [category, subcategory]],
        )
