import json
from collections import defaultdict
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

history_dates = all_dates[20:35]
history_df = articles_df[articles_df["published_at"].dt.normalize().isin(history_dates)]
interacted_articles = []
threshold = 5


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

topical_clicks = defaultdict(list)
article_by_tag_counts = defaultdict(list)
max_tag_count = 0

for article in interacted_articles:
    all_mentions = list({mention.entity.name for mention in article.mentions if mention.entity.name in topic_names})
    tag_counts = len(all_mentions)
    if len(all_mentions) > 0:
        article_by_tag_counts[tag_counts].append((article, all_mentions))
        if tag_counts > max_tag_count:
            max_tag_count = tag_counts

for each_count in range(1, max_tag_count + 1):
    for article, mentions in article_by_tag_counts[each_count]:
        for topic_name in mentions:
            if len(topical_clicks[topic_name]) < threshold:
                topical_click = Click(article_id=article.article_id)
                topical_clicks[topic_name].append(topical_click)

# for key, clicks in topical_clicks.items():
#     for click in clicks:
#         for article in interacted_articles:
#             if click.article_id == article.article_id:
#                 headline = article.headline
#                 all_mentions = list({mention.entity.name for mention in article.mentions if mention.entity.name in topic_names})
#                 print(f"{headline} || {key} || {all_mentions}")


at_least_one_news_per_topic_personas = {}

for persona in topics:
    if persona["entity_name"] in topical_clicks.keys():
        topic_profile = InterestProfile(
            profile_id=uuid4(),
            click_history=topical_clicks[persona["entity_name"]],
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

        at_least_one_news_per_topic_personas[persona["entity_name"]] = topic_profile

# breakpoint()
