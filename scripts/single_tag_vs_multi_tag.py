import json
from collections import defaultdict

import pandas as pd

from poprox_concepts import Article, Entity, Mention
from poprox_recommender.paths import project_root


def mention_included_historical_article_generator(mentions_df, history_df):
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

    return interacted_articles


def n_tagged_article_generator(topic_names, article_by_tag_counts, threshold, mn, mx):
    topical_articles = defaultdict(list)

    for each_count in range(mn, mx):
        for article, mentions in article_by_tag_counts[each_count]:
            for topic_name in mentions:
                if len(topical_articles[topic_name]) < threshold:
                    all_mentions = list(
                        {mention.entity.name for mention in article.mentions if mention.entity.name in topic_names}
                    )
                    topical_article = {"headline": article.headline, "mentions": all_mentions}
                    topical_articles[topic_name].append(topical_article)

    return topical_articles


def print_articles(topical_articles):
    for topic, articles in topical_articles.items():
        for article in articles:
            print(f"{topic} -> {article['headline']} mentions {article['mentions']}")


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

interacted_articles = mention_included_historical_article_generator(mentions_df, history_df)

article_by_tag_counts = defaultdict(list)
max_tag_count = 0

for article in interacted_articles:
    all_mentions = list({mention.entity.name for mention in article.mentions if mention.entity.name in topic_names})
    tag_counts = len(all_mentions)
    if len(all_mentions) > 0:
        article_by_tag_counts[tag_counts].append((article, all_mentions))
        if tag_counts > max_tag_count:
            max_tag_count = tag_counts


no_article_per_tag_count = 1

single_topical_articles = n_tagged_article_generator(
    topic_names, article_by_tag_counts, threshold=no_article_per_tag_count, mn=1, mx=2
)

multi_topical_articles = n_tagged_article_generator(
    topic_names, article_by_tag_counts, threshold=no_article_per_tag_count, mn=2, mx=max_tag_count
)


# print_articles(single_topical_articles)
# print("break ---------------------- break ---------------------- break")
# print_articles(multi_topical_articles)
