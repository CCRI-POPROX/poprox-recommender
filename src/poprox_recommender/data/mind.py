"""
Support for loading MIND_ data for evaluation.

.. _MIND: https://msnews.github.io/
"""

# pyright: basic
from __future__ import annotations

import logging
import zipfile
from typing import Generator, cast
from uuid import NAMESPACE_URL, UUID, uuid5

import numpy as np
import pandas as pd

from poprox_concepts import Article, Click, Entity, InterestProfile, Mention
from poprox_concepts.api.recommendations import RecommendationRequest
from poprox_recommender.data.eval import EvalData
from poprox_recommender.paths import project_root

logger = logging.getLogger(__name__)
TEST_REC_COUNT = 10


class MindData(EvalData):
    """
    News and behavior data loaded from MIND data.
    """

    name: str
    news_df: pd.DataFrame
    news_id_map: dict[UUID, str]
    news_id_rmap: dict[str, UUID]
    behavior_df: pd.DataFrame
    behavior_id_map: dict[UUID, str]

    def __init__(self, archive: str = "MINDsmall_dev"):
        self.name = archive
        news_df, behavior_df = load_mind_frames(archive)
        # index data frames for quick lookup of users & articles
        self.news_df = news_df.set_index("id")
        if not self.news_df.index.unique:
            logger.warning("news data has non-unique index")

        self.behavior_df = behavior_df.set_index("impression_id")
        if not self.behavior_df.index.unique:
            logger.warning("behavior data has non-unique index")

        # add and reverse-index the UUIDs
        ns_article = uuid5(NAMESPACE_URL, "https://data.poprox.io/mind/article/")
        self.news_df["uuid"] = [uuid5(ns_article, aid) for aid in self.news_df.index.values]
        # set up bidirectional maps for news IDs
        self.news_id_map = dict(zip(self.news_df["uuid"], self.news_df.index))
        self.news_id_rmap = dict(zip(self.news_df.index, self.news_df["uuid"]))

        ns_impression = uuid5(NAMESPACE_URL, "https://data.poprox.io/mind/impression/")
        self.behavior_df["uuid"] = [uuid5(ns_impression, str(iid)) for iid in self.behavior_df.index.values]
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
        for row in self.behavior_df.itertuples():
            clicked_ids: list[str] = row.clicked_news.split()  # type: ignore
            cand_pairs: list[str] = row.impressions.split()  # type: ignore

            clicks = [Click(article_id=self.news_uuid_for_id(aid)) for aid in clicked_ids]
            past = []
            for aid in clicked_ids:
                past.append(self.lookup_article(id=aid))

            today = []
            for pair in cand_pairs:
                aid, _clicked = pair.split("-")
                today.append(self.lookup_article(id=aid))

            clicks = [Click(article_id=self.news_uuid_for_id(aid)) for aid in clicked_ids]
            profile = InterestProfile(profile_id=cast(UUID, row.uuid), click_history=clicks, onboarding_topics=[])
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


def load_mind_frames(archive: str = "MINDsmall_dev") -> tuple[pd.DataFrame, pd.DataFrame]:
    data = project_root() / "data"
    logger.info("loading MIND data from %s", archive)
    with zipfile.ZipFile(data / f"{archive}.zip") as zf:
        behavior_df = _read_zipped_tsv(
            zf, "behaviors.tsv", ["impression_id", "user", "time", "clicked_news", "impressions"]
        )
        size = behavior_df.memory_usage(deep=True).sum()
        logger.info("loaded %d impressions from %s (%.1f MiB)", len(behavior_df), archive, size / (1024 * 1024))

        # FIXME: don't blanket fillna
        behavior_df.fillna("", inplace=True)

        news_df = _read_zipped_tsv(zf, "news.tsv", ["id", "category", "subcategory", "title"])
        size = news_df.memory_usage(deep=True).sum()
        logger.info("loaded %d articles from %s (%.1f MiB)", len(news_df), archive, size / (1024 * 1024))

    return news_df, behavior_df


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
