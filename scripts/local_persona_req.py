import json
from pathlib import Path

import pandas as pd

# from single_tagged_click_persona_generator import single_topic_personas
from multi_tagged_click_presona_generator import at_least_one_news_per_topic_personas
from persona_generator import single_topic_personas
from tqdm import tqdm

from poprox_concepts import Article, CandidateSet, Entity, Mention
from poprox_concepts.api.recommendations.v2 import RecommendationRequestV2, RecommendationResponseV2
from poprox_recommender.api.main import root
from poprox_recommender.paths import project_root

data = project_root() / "data"
articles_df = pd.read_parquet(data / "Test" / "articles.parquet")
mentions_df = pd.read_parquet(data / "Test" / "mentions.parquet")

output_path = Path(data / "Test" / "recommendation")
rec_response = []

all_dates = sorted(articles_df["published_at"].dt.normalize().unique())
static_num_recs = 10


### preserved some articles for user history

history_dates = all_dates[20:35]
history_df = articles_df[articles_df["published_at"].dt.normalize().isin(history_dates)]
interactable_articles = {}


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

    interactable_articles[article.article_id] = article

print("After History")
### preserved some articles for user history


### day to day caddidate article

cadidate_dates = all_dates[-1:]

for day in tqdm(cadidate_dates):
    day_df = articles_df[articles_df["published_at"].dt.normalize() == day]

    candidate_articles = []
    for row in day_df.itertuples():
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

        candidate_articles.append(article)

    print("Candidate Append")

    if len(candidate_articles) < static_num_recs:
        continue

    for persona_topic, persona_profile in at_least_one_news_per_topic_personas.items():
        clicked_article = []
        for click in persona_profile.click_history:
            clicked_article.append(interactable_articles[click.article_id])

        print(persona_topic)
        full_request = {
            "interest_profile": persona_profile.model_dump(),
            "interacted": CandidateSet(articles=clicked_article),
            "candidates": CandidateSet(articles=candidate_articles),
            "num_recs": static_num_recs,
        }

        req = RecommendationRequestV2.model_validate(full_request)

        response = root(req.model_dump(), pipeline="nrms_topic_scores")
        response = RecommendationResponseV2.model_validate(response)

        candidate_count = 0
        for article in candidate_articles:
            article_topics = set([m.entity.name for m in article.mentions if m.entity is not None])
            if persona_topic in article_topics:
                candidate_count += 1

        rec_response.append(
            {
                "persona_name": persona_topic,
                "persona_profile_id": str(persona_profile.profile_id),
                "candidate_day": day.date().isoformat(),
                "response": response.model_dump_json(),
                "candidate_count": candidate_count,
            }
        )

### day to day caddidate article

df = pd.DataFrame(rec_response)
output_path = output_path / "nrms_topic_pn_scores_click.parquet"
df.to_parquet(output_path)
