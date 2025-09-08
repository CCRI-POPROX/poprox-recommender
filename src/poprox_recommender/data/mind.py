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
from typing import Generator, cast
from uuid import NAMESPACE_URL, UUID, uuid5

import duckdb
import duckdb.typing as dt
import numpy as np
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
    "Path to the unzipped MIND dataset."
    duck: DuckDBPyConnection | None = None
    "DuckDB connection."

    news_df: pd.DataFrame
    "Data frame of news articles."
    news_id_map: dict[UUID, str]
    "Mapping from POPROX UUIDs to MIND article IDs."
    news_id_rmap: dict[str, UUID]
    "Mapping from MIND article IDs to POPROX UUIDs."
    behavior_df: pd.DataFrame
    "Data frame of interaction behavior."
    behavior_id_map: dict[UUID, str]
    "Mapping from POPROX impression UUIDs to MIND article IDs."

    def __init__(self, archive: str = "MINDsmall_dev"):
        self.name = archive
        self.path = project_root() / "data" / archive

        news_df = self._load_news_df()
        # index data frames for quick lookup of users & articles
        self.news_df = news_df.set_index("id")
        if not self.news_df.index.unique:
            logger.warning("news data has non-unique index")

        # TODO: only do this when we don't also load duck, or just use duck
        behavior_df = self._load_behavior_df()
        self.behavior_df = behavior_df.set_index("impression_id")
        if not self.behavior_df.index.unique:
            logger.warning("behavior data has non-unique index")

        # set up bidirectional maps for news IDs
        self.news_id_map = dict(zip(self.news_df["uuid"], self.news_df.index))
        self.news_id_rmap = dict(zip(self.news_df.index, self.news_df["uuid"]))

        # set up quick mapping for behavior IDs
        self.behavior_id_map = dict(zip(self.behavior_df["uuid"], self.behavior_df.index))

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
        return self.behavior_df.shape[0]

    @property
    def n_articles(self) -> int:
        return self.news_df.shape[0]

    def profile_truth(self, user: UUID) -> pd.DataFrame | None:
        """
        Look up the ground-truth data for a particular user profile,
        in LensKit format with item UUIDs for item IDs.
        """
        try:
            uid = self.behavior_id_for_uuid(user)
            imp_log = str(self.behavior_df.loc[uid, "impressions"])
        except KeyError:
            raise ValueError(f"unknown user {user}")

        # helper generator to only split articles once
        def split_records():
            for article in imp_log.split():
                iid, rv = article.split("-")
                yield iid, int(rv)

        truth = pd.DataFrame.from_records(
            split_records(),
            columns=["mind_item_id", "rating"],
        )
        ids = np.array([self.news_uuid_for_id(aid) for aid in truth["mind_item_id"]])
        truth = truth.set_index(ids)
        truth.index.name = "item_id"
        return truth

    def iter_profiles(self) -> Generator[RecommendationRequest]:
        assert self.duck is not None
        # collect the with larger impression sets
        self.duck.execute(
            """
            SELECT imp_uuid, clicked_articles,
                array_agg(article_id) AS candidate_articles
            FROM impressions, impressed_articles
            WHERE first_day <= imp_day
            AND last_day > imp_day - 7
            GROUP BY imp_uuid
            """
        )

        # loop over the results and yield the recommendations
        while row := self.duck.fetchone():
            imp_uuid, clicked, candidates = row

            # convert clicked articles into list
            clicks = [Click(article_id=self.news_uuid_for_id(aid)) for aid in clicked]
            past = [self.lookup_article(id=aid) for aid in clicked]

            today = [self.lookup_article(id=aid) for aid in candidates]

            profile = InterestProfile(profile_id=imp_uuid, click_history=clicks, onboarding_topics=[])
            yield RecommendationRequest(
                todays_articles=today, past_articles=past, interest_profile=profile, num_recs=TEST_REC_COUNT
            )

    def lookup_article(self, *, id: str | None = None, uuid: UUID | None = None):
        if uuid is None:
            if id:
                uuid = self.news_uuid_for_id(id)
            else:
                raise ValueError("must provide one of uuid, id")
        elif id is None:
            id = self.news_id_for_uuid(uuid)

        category = Entity(name=str(self.news_df.loc[id, "category"]), entity_type="category", source="MIND")
        subcategory = Entity(name=str(self.news_df.loc[id, "subcategory"]), entity_type="subcategory", source="MIND")

        article = Article(
            article_id=uuid,
            url=f"urn:uuid:{uuid}",
            headline=str(self.news_df.loc[id, "title"]),
            mentions=[Mention(source="MIND", relevance=1, entity=entity) for entity in [category, subcategory]],
        )
        return article

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
            imp_uuid UUID NOT NULL,
            user_id VARCHAR NOT NULL,
            imp_time TIMESTAMP NOT NULL,
            -- Julian day number for each impression to put in days.
            imp_day INTEGER NOT NULL,
            clicked_articles VARCHAR[] NOT NULL,
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
            impression_uuid(impression_id)
            user_id,
            strptime(time, '%m/%d/%Y %H:%M:%S %p'),
            CAST(julian(strptime(time, '%m/%d/%Y %H:%M:%S %p')) AS INTEGER),
            -- split apart news
            string_split_regex(clicked_news, '\\s+'),
            -- strip response signals + split apart impressed articles
            string_split_regex(regexp_replace(impression_news, '-[01]', '', 'g'), '\\s+')
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
            FROM impressions
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
