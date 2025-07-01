import json
from copy import deepcopy
from uuid import uuid4

import pandas as pd

from poprox_concepts import AccountInterest, Article, Click, Entity, InterestProfile, Mention
from poprox_recommender.paths import project_root

with open(project_root() / "tests/request_data/onboarding.json", "r") as req_file:
    base_request_data = json.load(req_file)

topics = base_request_data["interest_profile"]["onboarding_topics"]
topic_names = [topic["entity_name"] for topic in topics]

# fetching historical data
data = project_root() / "data"
articles_df = pd.read_parquet(data / "Test" / "articles.parquet")
mentions_df = pd.read_parquet(data / "Test" / "mentions.parquet")

all_dates = sorted(articles_df["published_at"].dt.normalize().unique())

history_dates = all_dates
history_df = articles_df[articles_df["published_at"].dt.normalize().isin(history_dates)]
interacted_articles = []


for row in history_df.itertuples():
    article_mentions_df = mentions_df[mentions_df["article_id"] == row.article_id]

    mentions = [
        Mention(
            mention_id=m_row.mention_id,
            article_id=m_row.article_id,
            source=m_row.source,
            relevance=m_row.relevance,
            entity=Entity(**json.loads(m_row.entity)) if m_row.entity else None,
        )
        for m_row in article_mentions_df.itertuples()
    ]

    article = Article(
        article_id=row.article_id,
        headline=row.headline,
        subhead=row.subhead,
        body=row.body,
        published_at=row.published_at,
        mentions=mentions,
        source="AP",
        external_id="",
        raw_data=json.loads(row.raw_data) if row.raw_data else None,
    )

    interacted_articles.append(article)

# fetching historical data

topical_clicks = {}

for article in interacted_articles:
    all_mentions = list({mention.entity.name for mention in article.mentions if mention.entity.name in topic_names})

    if len(all_mentions) == 1:
        topic_name = all_mentions[0]
        if topic_name not in topical_clicks:
            topical_click = Click(article_id=article.article_id)
            topical_clicks[topic_name] = topical_click


single_topic_personas = {}

for persona in topics:
    if persona["entity_name"] in topical_clicks:
        topic_profile = InterestProfile(
            profile_id=uuid4(),
            click_history=[topical_clicks[persona["entity_name"]]],
            click_topic_counts=None,
            click_locality_counts=None,
            article_feedbacks={},
            onboarding_topics=[],
        )

        for topic in topics:
            preference = 5 if topic["entity_name"] == persona["entity_name"] else 1
            topic_profile.onboarding_topics.append(
                AccountInterest(entity_id=topic["entity_id"], entity_name=topic["entity_name"], preference=preference)
            )

        single_topic_personas[persona["entity_name"]] = topic_profile

# breakpoint()
