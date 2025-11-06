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

from poprox_concepts.api.recommendations import RecommendationRequestV4
from poprox_concepts.domain import Article, CandidateSet, Click, Entity, InterestProfile, Mention
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

        if not self.path.exists():
            raise FileNotFoundError(self.path)

        self.duck = duckdb.connect(self.path, read_only=True)

        # pre-fetch counts
        self.duck.execute("SELECT COUNT(*) FROM impressions")
        (n,) = self.duck.fetchone() or (0,)
        self._impression_count = n

        self.duck.execute("SELECT COUNT(*) FROM articles")
        (n,) = self.duck.fetchone() or (0,)
        self._article_count = n

    def news_uuid_for_id(self, id: str | int) -> UUID:
        if isinstance(id, int):
            id = f"N{id}"
        return uuid5(NAMESPACE_ARTICLE, id)

    def behavior_uuid_for_id(self, id: str | int) -> UUID:
        return uuid5(NAMESPACE_IMPRESSION, str(id))

    @property
    def n_profiles(self) -> int:
        return self._impression_count

    @property
    def n_articles(self) -> int:
        return self._article_count

    def list_articles(self) -> list[UUID]:
        """
        Get the list of all known article UUIDs.
        """
        self.duck.execute("SELECT article_uuid FROM articles")
        return [row[0] for row in self.duck.fetchall()]

    def profile_truth(self, user: UUID) -> pd.DataFrame | None:
        """
        Look up the ground-truth data for a particular user profile,
        in LensKit format with item UUIDs for item IDs.
        """

        self.duck.execute(
            """
            SELECT article_id AS mind_item_id,
                CAST(clicked AS INT2) AS rating
            FROM impressions
            JOIN impression_articles USING (imp_id)
            WHERE imp_uuid = ?
            """,
            [user],
        )
        truth = self.duck.fetch_df()
        truth["item_id"] = [self.news_uuid_for_id(aid) for aid in truth["mind_item_id"]]
        return truth.set_index("item_id")

    def iter_profiles(self, *, limit: int | None = None) -> Generator[RecommendationRequestV4]:
        """
        Iterate the test profiles.

        Args:
            ids_only:
                If ``True``, only yield impression IDs, not entire
                recommendation requests.
        """
        for imp_id in self.iter_profile_ids(limit=limit):
            yield self.lookup_request(id=imp_id)

    def iter_profile_ids(self, *, limit: int | None = None) -> Generator[int]:
        """
        Iterate the identifiers of profiles.
        """
        logger.info("querying for test impressions / profiles")

        # we use 2 queries: an outer query to list the impression IDs, and inner
        # queries to get the articles and article data.  outer query is in a cloned
        # connection so that they don't interfere with each other.
        with self.duck.cursor() as clone:
            query = "SELECT imp_id, imp_uuid FROM impressions"
            if limit is not None:
                assert isinstance(limit, int)
                query += f" LIMIT {limit}"

            clone.execute(query)

            # loop over the results and yield the recommendations
            logger.info("iterating test articles")
            while row := clone.fetchone():
                imp_id, imp_uuid = row

                yield imp_id

    def lookup_request(self, id: int | UUID) -> RecommendationRequestV4:
        if isinstance(id, UUID):
            uuid = id
            self.duck.execute("SELECT imp_id FROM impressions WHERE imp_uuid = ?", [uuid])
            if row := self.duck.fetchone():
                (imp_id,) = row
            else:
                raise KeyError(f"unknown impression {uuid}")
        else:
            imp_id = id
            uuid = self.behavior_uuid_for_id(id)

        assert id is not None

        # get the historical articles and click list
        past = self.lookup_articles(imp_id, relation="history")
        clicks = [Click(article_id=a.article_id) for a in past]

        # get the candidate articles
        today = self.lookup_articles(imp_id, relation="candidates")

        # FIXME the profile ID should probably be the user ID
        profile = InterestProfile(profile_id=uuid, click_history=clicks, onboarding_topics=[])
        return RecommendationRequestV4(
            candidates=CandidateSet(articles=today),
            interacted=CandidateSet(articles=past),
            interest_profile=profile,
            num_recs=TEST_REC_COUNT,
        )

    def lookup_articles(
        self, imp_id: int, *, relation: Literal["history", "candidates", "expanded-candidates"]
    ) -> list[Article]:
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
        elif relation == "candidates":
            self.duck.execute(
                """
                SELECT article_uuid, category, subcategory, title
                FROM impression_articles
                JOIN articles USING (article_id)
                WHERE imp_id = ?
                """,
                [imp_id],
            )
        elif relation == "expanded-candidates":
            self.duck.execute(
                """
                SELECT article_uuid, category, subcategory, title
                FROM impression_expanded_candidates
                JOIN articles USING (article_id)
                WHERE imp_id = ?
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

    # we cannot pickle live DuckDB connections, so drop object + reconnect at startup
    def __getstate__(self):
        return {name: val for name, val in self.__dict__.items() if name != "duck"}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.duck = duckdb.connect(self.path, read_only=True)
