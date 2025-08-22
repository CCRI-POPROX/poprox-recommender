import json
from collections import defaultdict

import pandas as pd

from poprox_concepts import Article, Click, Entity, Mention
from poprox_recommender.components.topical_description import TOPIC_DESCRIPTIONS
from poprox_recommender.paths import project_root


def complete_article_generator(row, mentions_df):
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
    return article


def unique_topic_mentions(article):
    seen = set()
    out = []
    for mention in article.get("mentions", []):
        entity = mention.get("entity") or {}
        name = entity.get("name")
        if name in topic_names and name not in seen:
            seen.add(name)
            out.append(name)
    return out


# fetching article and mention data
data = project_root() / "data" / "Test"
articles_df = pd.read_parquet(data / "articles.parquet")
mentions_df = pd.read_parquet(data / "mentions.parquet")

# dividing dates into history & candidate
all_dates = sorted(articles_df["published_at"].dt.normalize().unique())
# print(len(all_dates))

history_dates = all_dates[-60:-30]


# preparing interacted article
history_df = articles_df[articles_df["published_at"].dt.normalize().isin(history_dates)]
topical_article_obj = []

for row in history_df.itertuples():
    article = complete_article_generator(row, mentions_df)
    topical_article_obj.append(article)

topical_articles = [a.model_dump(mode="json", exclude_none=True) for a in topical_article_obj]

topic_names = TOPIC_DESCRIPTIONS.keys()


topical_clicks = defaultdict(list)
article_by_tag_counts = defaultdict(list)
max_tag_count = 0
threshold = 20

for article in topical_articles:
    all_mentions = []
    for mention in article.get("mentions", []):
        entity = mention.get("entity")
        topic = entity["name"]
        if topic in topic_names:
            all_mentions.append(topic)
    tag_counts = len(all_mentions)
    if len(all_mentions) > 0:
        article_by_tag_counts[tag_counts].append((article, all_mentions))
        if tag_counts > max_tag_count:
            max_tag_count = tag_counts

for each_count in range(1, max_tag_count + 1):
    for article, mentions in article_by_tag_counts[each_count]:
        for topic_name in mentions:
            if len(topical_clicks[topic_name]) < threshold:
                topical_click = Click(article_id=article["article_id"])
                topical_clicks[topic_name].append(topical_click)

for key, clicks in topical_clicks.items():
    print(f"{key}")
    for click in clicks:
        for article in topical_articles:
            # breakpoint()
            if str(click.article_id) == article["article_id"]:
                headline = article["headline"]
                all_mentions = []
                for mention in article.get("mentions", []):
                    entity = mention.get("entity")
                    topic = entity["name"]
                    if topic in topic_names:
                        all_mentions.append(topic)
                print(f"{headline} || {key} || {all_mentions}")
