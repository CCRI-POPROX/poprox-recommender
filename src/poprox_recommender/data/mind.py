"""
Support for loading MIND_ data for evaluation.

.. _MIND: https://msnews.github.io/
"""

import json
import logging
import zipfile
from datetime import datetime, timezone
from typing import Any, List, NamedTuple
from uuid import UUID, uuid4

import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm

from poprox_recommender.paths import project_root

logger = logging.getLogger(__name__)


class MindTables(NamedTuple):
    """
    News and behavior data loaded from MIND (raw data frames).
    """

    news: pd.DataFrame
    behaviors: pd.DataFrame


class MindData(NamedTuple):
    """
    News and behavior data loaded from MIND (compiled dictionaries like original JSON).
    """

    news_uuid_ID: dict[UUID, str]
    behavior_uuid_ID: dict[UUID, str]
    test_list: list[dict[str, Any]]


class Article(BaseModel):
    article_id: UUID
    title: str
    content: str = ""
    url: str
    published_at: datetime = datetime(1970, 1, 1, 0, 0, tzinfo=timezone.utc)


class ClickHistory(BaseModel):
    account_id: UUID = None
    article_ids: List[UUID]


def convert_to_Article(data: dict) -> Article:
    if "published_at" not in data:
        data["published_at"] = datetime(1970, 1, 1, 0, 0, tzinfo=timezone.utc)
    return Article(**data)


def convert_to_ClickHistory(data):
    return ClickHistory(**data)


def load_mind_data(archive="MINDlarge_dev") -> MindData:
    frames = load_mind_frames(archive)
    news_df = frames.news
    behaviors_df = frames.behaviors

    news_df["uuid"] = [uuid4() for i in range(news_df.shape[0])]
    behaviors_df["uuid"] = [uuid4() for i in range(behaviors_df.shape[0])]

    ID_title = dict(zip(news_df.id, news_df.title))
    ID_newsuuid = dict(zip(news_df.id, news_df.uuid))

    news_uuid_ID = dict(zip(news_df.uuid, news_df.id))  # uuid - news id
    behavior_uuid_ID = dict(zip(behaviors_df.uuid, behaviors_df.impression_id))  # uuid - impression id

    test_list = []

    # todays_articles (candidates)
    # ID_newsuuid

    logger.info("converting MIND impressions to POPROX requests")
    for i in tqdm(range(behaviors_df.shape[0])):
        test_json = {}
        test_json["past_articles"] = []
        test_json["num_recs"] = 10
        test_json["todays_articles"] = []  # list of json

        row = behaviors_df.iloc[i]

        impression_uuid = row.uuid

        for candidate_pair in row.impressions.split(" "):
            single_news = {}
            single_news["article_id"] = ID_newsuuid[candidate_pair.split("-")[0]]
            single_news["url"] = str(ID_newsuuid[candidate_pair.split("-")[0]])
            single_news["title"] = single_news["content"] = ID_title[candidate_pair.split("-")[0]]

            single_news = convert_to_Article(single_news)
            single_news = single_news.model_dump()
            test_json["todays_articles"].append(single_news)

        for article in row.clicked_news.split():
            single_news = {}
            single_news["article_id"] = ID_newsuuid[article]
            single_news["url"] = str(ID_newsuuid[article])
            single_news["title"] = single_news["content"] = ID_title[article]

            single_news = convert_to_Article(single_news)
            single_news = single_news.model_dump()
            test_json["past_articles"].append(single_news)

        click_data = {
            "account_id": impression_uuid,
            "article_ids": [ID_newsuuid[id] for id in row.clicked_news.split()],
        }

        click_data = convert_to_ClickHistory(click_data)

        click_data = click_data.model_dump()
        test_json["click_data"] = click_data

        test_list.append(test_json)

    return MindData(news_uuid_ID, behavior_uuid_ID, test_list)


def load_mind_frames(archive="MINDlarge_dev") -> MindTables:
    """
    Load the news and behavior data frames from MIND data.
    """
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

        news_df = _read_zipped_tsv(zf, "news.tsv", ["id", "topic", "subtopic", "title"])
        size = news_df.memory_usage(deep=True).sum()
        logger.info("loaded %d articles from %s (%.1f MiB)", len(news_df), archive, size / (1024 * 1024))

    return MindTables(news_df, behavior_df)


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


def export_main():
    logging.basicConfig(level=logging.INFO)
    data = load_mind_data()
    out = project_root() / "data" / "mind-converted"
    out.mkdir(exist_ok=True, parents=True)
    with open(out / "news-uuid-id.json", "wt") as jsf:
        json.dump(data.news_uuid_ID, jsf)
    with open(out / "behavior-uuid-id.json", "wt") as jsf:
        json.dump(data.behavior_uuid_ID, jsf)
    with open(out / "test-data.json", "wt") as jsf:
        json.dump(data.test_list, jsf)


if __name__ == "__main__":
    export_main()
