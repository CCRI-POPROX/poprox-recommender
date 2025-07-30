import json
from collections import defaultdict

# from pathlib import Path
from uuid import uuid4

import pandas as pd
from tqdm import tqdm

from poprox_concepts import AccountInterest, Article, CandidateSet, Click, Entity, InterestProfile, Mention
from poprox_concepts.api.recommendations.v2 import RecommendationRequestV2, RecommendationResponseV2
from poprox_recommender.api.main import root
from poprox_recommender.paths import project_root

TOPIC_TO_UUID = {
    "U.S. news": "66ba9689-3ad7-4626-9d20-03930d88e302",
    "World news": "45770171-36d1-4568-a270-bf80d6fe18e7",
    "Politics": "ec489f76-18c6-4f78-b53b-fda5849a1056",
    "Business": "5f6de24a-9a1b-4863-ab01-1ecacf4c54b7",
    "Entertainment": "4554dcf2-6472-43a3-bfd6-e904a2936315",
    "Sports": "f984b26b-4333-42b3-a463-bc232bf95d5f",
    "Health": "b967a4f4-ac9d-4c09-81d3-af228f846d06",
    "Science": "1e813fd6-0998-43fb-9839-75fa96b69b32",
    "Technology": "606afcb8-3fc1-47a7-9da7-3d95115373a3",
    "Lifestyle": "e531cbd0-d967-4d87-ad13-3649fc00ffb4",
    "Religion": "b2bb4e26-6684-4cbd-9fd8-fa98ae87ca57",
    "Climate and environment": "b822877a-c1e2-44a9-8d6c-4610d8047a9a",
    "Education": "c74a986e-3bd9-4be0-b8c1-bd95e376d064",
    "Oddities": "16323227-4b42-4363-b67c-fd2be57c9aa1",
}


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


def topic_profile_generator(persona, persona_topical_clicks):
    topic_profile = InterestProfile(
        profile_id=uuid4(),
        click_history=persona_topical_clicks,
        click_topic_counts=None,
        click_locality_counts=None,
        article_feedbacks={},
        onboarding_topics=[],
    )
    for topic, uuid in TOPIC_TO_UUID.items():
        preference = 5 if topic == persona else 3
        topic_profile.onboarding_topics.append(
            AccountInterest(entity_id=uuid, entity_name=topic, preference=preference)
        )
    return topic_profile


def article_tag_counter(interacted_articles):
    article_by_tag_counts = defaultdict(list)
    max_tag_count = 0
    for article in interacted_articles:
        all_mentions = list(
            {mention.entity.name for mention in article.mentions if mention.entity.name in TOPIC_TO_UUID.keys()}
        )
        tag_counts = len(all_mentions)
        if len(all_mentions) > 0:
            article_by_tag_counts[tag_counts].append((article, all_mentions))
            if tag_counts > max_tag_count:
                max_tag_count = tag_counts
    return article_by_tag_counts, max_tag_count


def n_topical_click_generator(article_by_tag_counts, max_tag_count, threshold):
    topical_clicks = defaultdict(list)
    for each_count in range(1, max_tag_count + 1):
        for article, mentions in article_by_tag_counts[each_count]:
            for topic_name in mentions:
                if len(topical_clicks[topic_name]) < threshold:
                    topical_click = Click(article_id=article.article_id)
                    topical_clicks[topic_name].append(topical_click)
    return topical_clicks


def single_topic_persona_generator():
    single_topic_personas = {}
    for persona in TOPIC_TO_UUID.keys():
        persona_topical_clicks = []
        topic_profile = topic_profile_generator(persona, persona_topical_clicks)
        single_topic_personas[persona] = topic_profile
    return single_topic_personas


def single_clicked_topic_personas_generator(interacted_articles):
    article_by_tag_counts, max_tag_count = article_tag_counter(interacted_articles)
    threshold = 5

    topical_clicks = n_topical_click_generator(article_by_tag_counts, max_tag_count, threshold)

    single_clicked_topic_personas = {}

    for persona in TOPIC_TO_UUID.keys():
        if persona in topical_clicks.keys():
            persona_topical_clicks = topical_clicks[persona]
            topic_profile = topic_profile_generator(persona, persona_topical_clicks)
            single_clicked_topic_personas[persona] = topic_profile
    return single_clicked_topic_personas


# fetching article and mention data
data = project_root() / "data"
articles_df = pd.read_parquet(data / "Test" / "articles.parquet")
mentions_df = pd.read_parquet(data / "Test" / "mentions.parquet")

# article_counts = articles_df.groupby(articles_df["published_at"].dt.normalize()).size()
# print(article_counts)
all_dates = sorted(articles_df["published_at"].dt.normalize().unique())
# print(len(all_dates))

# Preparing interacted article
history_dates = all_dates[13:35]
history_df = articles_df[articles_df["published_at"].dt.normalize().isin(history_dates)]
interacted_articles = []

for row in history_df.itertuples():
    article = complete_article_generator(row, mentions_df)
    interacted_articles.append(article)

interacted_articles_dict = {a.article_id: a for a in interacted_articles}

# Preparing candidate article
cadidate_dates = all_dates[-8:]


# Varation01::           for no click single topic persona
single_topic_personas = single_topic_persona_generator()
# End of Varation01::    for no click single topic persona


# Varation02::           for at least one news per topic personas with regarding topical preference
# single_clicked_topic_personas = single_clicked_topic_personas_generator(interacted_articles)
# End of Varation02::    for at least one news per topic personas with regarding topical preference


persona_wise_rec_recall = defaultdict(list)
rec_response = []
static_num_recs = 10

### day to day caddidate article
for day in tqdm(cadidate_dates):
    day_df = articles_df[articles_df["published_at"].dt.normalize() == day]

    candidate_articles = []
    for row in day_df.itertuples():
        article = complete_article_generator(row, mentions_df)
        candidate_articles.append(article)

    if len(candidate_articles) < static_num_recs:
        continue

    for persona_topic, persona_profile in single_topic_personas.items():
        clicked_article = []
        for click in persona_profile.click_history:
            clicked_article.append(interacted_articles_dict[click.article_id])

        full_request = {
            "interest_profile": persona_profile.model_dump(),
            "interacted": CandidateSet(articles=clicked_article),
            "candidates": CandidateSet(articles=candidate_articles),
            "num_recs": static_num_recs,
        }

        req = RecommendationRequestV2.model_validate(full_request)

        response = root(req.model_dump(), pipeline="nrms_topic_scores")
        response = RecommendationResponseV2.model_validate(response)
        response = response.model_dump()

        candidate_count = 0
        for article in candidate_articles:
            article_topics = set([m.entity.name for m in article.mentions if m.entity is not None])
            if persona_topic in article_topics:
                candidate_count += 1

        articles = response["recommendations"]["articles"]
        topical_count = 0
        for i, article in enumerate(articles):
            article_topics = set([m["entity"]["name"] for m in article["mentions"] if m["entity"] is not None])
            if persona_topic in article_topics:
                topical_count += 1

        recall = topical_count / candidate_count if candidate_count else float("nan")
        precision = topical_count / 10
        persona_wise_rec_recall[persona_topic].append((day, precision, recall))


for topic, values in persona_wise_rec_recall.items():
    print(f"\nTopic: {topic}")
    for day, precision, recall in values:
        print(f"{day.date()}: Precision = {precision:.5f}, Recall = {recall:.5f}")
