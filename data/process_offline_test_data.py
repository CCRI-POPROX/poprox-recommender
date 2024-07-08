import json
from datetime import datetime
from uuid import UUID, uuid4

import pandas as pd
from tqdm import tqdm

from poprox_concepts import Article, ClickHistory


def custom_encoder(obj):
    if isinstance(obj, UUID):
        return str(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()


def load_data():
    val_behavior_df = pd.read_table(
        "./val_mind_large/behaviors.tsv",
        header="infer",
        usecols=range(5),
        names=["impression_id", "user", "time", "clicked_news", "impressions"],
    )

    val_behavior_df.fillna("", inplace=True)

    val_news_df = pd.read_table(
        "./val_mind_large/news.tsv", header="infer", usecols=range(4), names=["id", "topic", "subtopic", "title"]
    )

    print(
        f"The total unique uuid is {val_behavior_df.shape[0] + val_news_df.shape[0]}"
    )  # The total unique uuid is 448494
    return val_behavior_df, val_news_df


def assign_uuid(val_behavior_df, val_news_df):
    uuids = [uuid4() for _ in range(448494)]
    val_news_df["uuid"] = uuids[: val_news_df.shape[0]]
    val_news_df["str_uuid"] = val_news_df["uuid"].apply(lambda x: str(x))

    val_behavior_df["uuid"] = uuids[val_news_df.shape[0] :]
    val_behavior_df["str_uuid"] = val_behavior_df["uuid"].apply(lambda x: str(x))

    ID_title = dict(zip(val_news_df.id, val_news_df.title))
    ID_newsuuid = dict(zip(val_news_df.id, val_news_df.uuid))

    news_struuid_ID = dict(zip(val_news_df.str_uuid, val_news_df.id))  # uuid - news id
    behavior_struuid_ID = dict(zip(val_behavior_df.str_uuid, val_behavior_df.impression_id))  # uuid - impression id

    with open("./news_uuid_ID.json", "w") as file:
        json.dump(news_struuid_ID, file, indent=4)
    with open("./behavior_uuid_ID.json", "w") as file:
        json.dump(behavior_struuid_ID, file, indent=4)

    return ID_title, ID_newsuuid


def generate_test_data(val_behavior_df, ID_title, ID_newsuuid):
    test_list = []
    for i in tqdm(range(val_behavior_df.shape[0])):
        test_json = {}
        test_json["past_articles"] = []
        test_json["num_recs"] = 10
        test_json["todays_articles"] = []  # list of json

        row = val_behavior_df.iloc[i]

        impression_uuid = row.str_uuid

        for candidate_pair in row.impressions.split(" "):
            single_news = {}
            single_news["article_id"] = ID_newsuuid[candidate_pair.split("-")[0]]
            single_news["url"] = str(ID_newsuuid[candidate_pair.split("-")[0]])
            single_news["title"] = single_news["content"] = ID_title[candidate_pair.split("-")[0]]

            single_news = Article(**single_news)
            single_news = single_news.model_dump()
            test_json["todays_articles"].append(single_news)

        for article in row.clicked_news.split():
            single_news = {}
            single_news["article_id"] = ID_newsuuid[article]
            single_news["url"] = str(ID_newsuuid[article])
            single_news["title"] = single_news["content"] = ID_title[article]

            single_news = Article(**single_news)
            single_news = single_news.model_dump()
            test_json["past_articles"].append(single_news)

        click_data = {
            "account_id": impression_uuid,
            "article_ids": [ID_newsuuid[id] for id in row.clicked_news.split()],
        }

        click_data = ClickHistory(**click_data)

        click_data = click_data.model_dump()
        test_json["click_data"] = click_data

        test_list.append(test_json)

    print(len(test_list))  # 376471
    with open("./mind_test.json", "w") as file:
        json.dump(test_list, file, default=custom_encoder, indent=4)


if __name__ == "__main__":
    """expected test MIND data format
    {
            "past_articles": [
                {'article_id': UUID,
                "title": "title1",
                "content": "content",
                "url": "url 1"
                }
            ],

            "todays_articles": [
                {'article_id': UUID,
                "title": "title1",
                "url": "url 1"
                },
            ],

            "click_data": {
                UUID: [list of articel ids]
            }

            "num_recs": int
    }
    """
    # 1. load the data for original news and behavior
    val_behavior_df, val_news_df = load_data()

    # 2. Assign uuid to news id and impression id to produce some data file we need for constructing mind test file
    ID_title, ID_newsuuid = assign_uuid(val_behavior_df, val_news_df)

    # 3. Generate the mind test data
    generate_test_data(val_behavior_df, ID_title, ID_newsuuid)
