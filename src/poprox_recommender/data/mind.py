"""
Support for loading MIND_ data for evaluation.

.. _MIND: https://msnews.github.io/
"""

# pyright: basic
from __future__ import annotations

import logging
import zipfile
from os import fspath
from pathlib import Path
from typing import Generator, Literal, cast
from uuid import NAMESPACE_URL, UUID, uuid5

import duckdb
import duckdb.typing as dt
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

CLI_SPEC = """
Preprocess MIND data into DuckDB for evaluation.

Usage:
    poprox_recommender.data.mind [-v] NAME

Options:
    -v, --verbose
            Enable verbose log output.
    NAME    The name of the MIND data set to process.
"""


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

    def news_id_for_uuid(self, uuid: UUID) -> str:
        return self.news_id_map[uuid]

    def news_uuid_for_id(self, id: str) -> UUID:
        return self.news_id_rmap[id]

    def behavior_id_for_uuid(self, uuid: UUID) -> str:
        return self.behavior_id_map[uuid]

    def behavior_uuid_for_id(self, id: str) -> UUID:
        return cast(UUID, self.behavior_df.loc[id, "uuid"])

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

    def _load_news_df(self) -> pd.DataFrame:
        logger.info("loading MIND news from %s", self.name)

        news_df = pd.read_table(
            self.path / "news.tsv", header=None, usecols=range(4), names=["id", "category", "subcategory", "title"]
        )
        news_df["uuid"] = [uuid5(NAMESPACE_ARTICLE, aid) for aid in news_df.index.values]
        size = news_df.memory_usage(deep=True).sum()
        logger.info("loaded %d articles from %s (%.1f MiB)", len(news_df), self.name, size / (1024 * 1024))

        return news_df

    def _load_behavior_df(self) -> pd.DataFrame:
        logger.info("loading MIND behavior from %s", self.name)

        behavior_df = pd.read_table(
            self.path / "behaviors.tsv",
            header=None,
            usecols=range(5),
            names=["impression_id", "user", "time", "clicked_news", "impressions"],
        )
        behavior_df["uuid"] = [uuid5(NAMESPACE_IMPRESSION, str(iid)) for iid in behavior_df.index.values]
        size = behavior_df.memory_usage(deep=True).sum()
        logger.info("loaded %d impressions from %s (%.1f MiB)", len(behavior_df), self.name, size / (1024 * 1024))

        return behavior_df

    def _open_behavior_db(self):
        logger.info("loading MIND data into DuckDB")
        # create an *in-memory* DuckDB connection
        self.duck = duckdb.connect()
        _bind_raw_impressions(self.duck, self.path / "behaviors.tsv")
        _transform_impressions(self.duck)
        _extract_impressed_articles(self.duck)
        logger.info("database prepared")


def _read_zipped_tsv(zf: zipfile.ZipFile, name: str, columns: list[str]) -> pd.DataFrame:
    """
    Read a TSV file from the compressed MIND data as a Pandas data frame.

    Args:
        zf: The zip file, opened for reading.
        name: The name of the file to read within the zip file (e.g. ``news.tsv``).
        columns: The column names for this zip file.
    """
    with zf.open(name, "r") as content:
        return pd.read_table(
            content,
            header=None,
            usecols=range(len(columns)),
            names=columns,
        )


def _bind_raw_impressions(db: DuckDBPyConnection, path: Path):
    """
    Create a DuckDB view that will read from the raw impression table.
    """
    db.read_csv(
        fspath(path),
        header=False,
        delimiter="\t",
        columns={
            "impression_id": "INTEGER",
            "user_id": "VARCHAR",
            "time": "VARCHAR",
            "clicked_news": "VARCHAR",
            "impression_news": "VARCHAR",
        },
    ).create_view("raw_impressions")


def _transform_impressions(db: DuckDBPyConnection):
    """
    Transform the impressions from their raw form into a structured DuckDB
    table.
    """
    # custom function for impression UUID conversion
    db.create_function("impression_uuid", _impression_uuid, [dt.VARCHAR], dt.UUID)  # type: ignore
    # define the impression table. at this point, imp_articles will contain the
    # impressed article IDs only, not their clicks!
    db.execute(
        """
        CREATE TABLE impressions (
            imp_id VARCHAR NOT NULL PRIMARY KEY,
            imp_uuid UUID NOT NULL UNIQUE,
            user_id VARCHAR NOT NULL,
            imp_time TIMESTAMP NOT NULL,
            -- Julian day number for each impression to put in days.
            imp_day INTEGER NOT NULL,
            clicked_articles VARCHAR[],
            imp_articles VARCHAR[] NOT NULL,
        )
        """
    )
    # populate the impression table
    db.execute(
        """
        INSERT INTO impressions (imp_id, imp_uuid, user_id, imp_time, imp_day, clicked_articles, imp_articles)
        SELECT
            impression_id,
            impression_uuid(CAST(impression_id AS VARCHAR)),
            user_id,
            strptime(time, '%m/%d/%Y %H:%M:%S %p'),
            CAST(julian(strptime(time, '%m/%d/%Y %H:%M:%S %p')) AS INTEGER),
            -- split apart news
            string_split_regex(clicked_news, '\\s+'),
            -- strip response signals + split apart impressed articles
            string_split_regex(regexp_replace(impression_news, '-[01]', '', 'g'), '\\s+')
        FROM raw_impressions
        """
    )


def _extract_impressed_articles(db: DuckDBPyConnection):
    """
    Process the impression articles, extracting the first and last day in which
    each was displayed.
    """
    # create a table of unique impressed articles with first and last days, sorted for fast lookup.
    db.execute(
        """
        CREATE TABLE impressed_articles AS
        SELECT article_id, MIN(imp_day) AS first_day, MAX(imp_day) AS last_day
        FROM (
            SELECT UNNEST(imp_articles) AS article_id, imp_day
            FROM impression
        )
        GROUP BY article_id
        ORDER BY first_day
        """
    )
    # index the first and last day columns for fast lookup.
    db.execute(
        """
        CREATE INDEX imp_article_first_idx ON impressed_articles (first_day)
        """
    )
    db.execute(
        """
        CREATE INDEX imp_article_last_idx ON impressed_articles (last_day)
        """
    )


def _impression_uuid(mind_id: str) -> UUID:
    return uuid5(NAMESPACE_IMPRESSION, mind_id)
