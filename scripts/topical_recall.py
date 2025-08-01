import csv
import json
import math
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


# utility functions for article generator, rec_request generator and result store


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


def full_request_generator(persona_profile, interacted_articles_dict, candidate_articles, static_num_recs):
    clicked_article = []
    for click in persona_profile.click_history:
        clicked_article.append(interacted_articles_dict[click.article_id])

    full_request = {
        "interest_profile": persona_profile.model_dump(),
        "interacted": CandidateSet(articles=clicked_article),
        "candidates": CandidateSet(articles=candidate_articles),
        "num_recs": static_num_recs,
    }
    return RecommendationRequestV2.model_validate(full_request)


def store_rec_recall_as_csv(sorted_persona_wise_rec_recall, pipeline, variation, def_pref):
    csv_rows = [("Topic", "Avg_Precision", "Avg_Recall")]
    for persona, values in sorted_persona_wise_rec_recall.items():
        csv_rows.append((persona, f"{values['avg_precision']:.5f}", f"{values['avg_recall']:.5f}"))

    with open(data / f"{pipeline}_{variation}_{def_pref}.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_rows)


# utility functions for synthetic data generation


def topic_profile_generator(persona, persona_topical_clicks, def_pref, interest_score):
    topic_profile = InterestProfile(
        profile_id=uuid4(),
        click_history=persona_topical_clicks,
        click_topic_counts=None,
        click_locality_counts=None,
        article_feedbacks={},
        onboarding_topics=[],
    )
    for topic, uuid in TOPIC_TO_UUID.items():
        preference = interest_score if topic == persona else def_pref
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


def n_topical_click_generator(article_by_tag_counts, max_tag_count, no_of_historical_click):
    topical_clicks = defaultdict(list)
    for each_count in range(1, max_tag_count + 1):
        for article, mentions in article_by_tag_counts[each_count]:
            for topic_name in mentions:
                if len(topical_clicks[topic_name]) < no_of_historical_click:
                    topical_click = Click(article_id=article.article_id)
                    topical_clicks[topic_name].append(topical_click)
    return topical_clicks


# variation of synthetic data


def single_topic_persona_generator(def_pref):
    interest_score = 5
    single_topic_personas = {}
    for persona in TOPIC_TO_UUID.keys():
        persona_topical_clicks = []
        topic_profile = topic_profile_generator(persona, persona_topical_clicks, def_pref, interest_score)
        single_topic_personas[persona] = topic_profile
    return single_topic_personas


def single_clicked_topic_personas_generator(interacted_articles, def_pref):
    article_by_tag_counts, max_tag_count = article_tag_counter(interacted_articles)
    no_of_historical_click = 5

    topical_clicks = n_topical_click_generator(article_by_tag_counts, max_tag_count, no_of_historical_click)

    # # just for confirming right clicks
    # for key, clicks in topical_clicks.items():
    #     for click in clicks:
    #         for article in interacted_articles:
    #             if click.article_id == article.article_id:
    #                 headline = article.headline
    #                 all_mentions = list({mention.entity.name
    #                                      for mention in article.mentions
    #                                      if mention.entity.name in TOPIC_TO_UUID.keys()})
    #                 print(f"{headline} || {key} || {all_mentions}")
    # # just for confirming right clicks

    interest_score = 5
    single_clicked_topic_personas = {}

    for persona in TOPIC_TO_UUID.keys():
        if persona in topical_clicks.keys():
            persona_topical_clicks = topical_clicks[persona]
            topic_profile = topic_profile_generator(persona, persona_topical_clicks, def_pref, interest_score)
            single_clicked_topic_personas[persona] = topic_profile
    return single_clicked_topic_personas


def single_click_personas_generator(interacted_articles, def_pref):
    article_by_tag_counts, max_tag_count = article_tag_counter(interacted_articles)
    no_of_historical_click = 5

    topical_clicks = n_topical_click_generator(article_by_tag_counts, max_tag_count, no_of_historical_click)

    # # just for confirming right clicks
    # for key, clicks in topical_clicks.items():
    #     for click in clicks:
    #         for article in interacted_articles:
    #             if click.article_id == article.article_id:
    #                 headline = article.headline
    #                 all_mentions = list({mention.entity.name
    #                                      for mention in article.mentions
    #                                      if mention.entity.name in TOPIC_TO_UUID.keys()})
    #                 print(f"{headline} || {key} || {all_mentions}")
    # # just for confirming right clicks
    interest_score = 3
    single_click_personas = {}

    for persona in TOPIC_TO_UUID.keys():
        if persona in topical_clicks.keys():
            persona_topical_clicks = topical_clicks[persona]
            topic_profile = topic_profile_generator(persona, persona_topical_clicks, def_pref, interest_score)
            single_click_personas[persona] = topic_profile
    return single_click_personas


# parameter passing for synthetic data generation


def synthetic_personas_generator(def_pref, variation, interacted_articles):
    if variation == "topical_pref_only":
        return single_topic_persona_generator(def_pref)
    elif variation == "clicked_topic_personas":
        return single_clicked_topic_personas_generator(interacted_articles, def_pref)
    elif variation == "topical_click_only":
        return single_click_personas_generator(interacted_articles, def_pref)
    else:
        raise ValueError(f"Unknown variation type: {variation}")


# all the calculation functions


def daily_persona_wise_recall_calculator(candidate_articles, persona_topic, response):
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
    return recall, precision


def avg_persona_wise_rec_recall_over_days_calculator(persona_wise_rec_recall):
    for persona, values in persona_wise_rec_recall.items():
        daily_values = values["daily_scores"]
        total_precision = 0
        total_recall = 0
        total_precison_count = 0
        total_recall_count = 0
        for day, precision, recall in daily_values:
            total_precision += precision
            total_precison_count += 1
            if not math.isnan(recall):
                total_recall += recall
                total_recall_count += 1
        values["avg_precision"] = total_precision / total_precison_count
        values["avg_recall"] = total_recall / total_recall_count
    return dict(sorted(persona_wise_rec_recall.items(), key=lambda x: x[0]))


# fetching article and mention data
data = project_root() / "data" / "Test"
articles_df = pd.read_parquet(data / "articles.parquet")
mentions_df = pd.read_parquet(data / "mentions.parquet")

# article_counts = articles_df.groupby(articles_df["published_at"].dt.normalize()).size()
# print(article_counts)


# dividing dates into history & candidate
all_dates = sorted(articles_df["published_at"].dt.normalize().unique())
# print(len(all_dates))

history_dates = all_dates[-60:-30]
cadidate_dates = all_dates[-30:]


# preparing interacted article
history_df = articles_df[articles_df["published_at"].dt.normalize().isin(history_dates)]
interacted_articles = []

for row in history_df.itertuples():
    article = complete_article_generator(row, mentions_df)
    interacted_articles.append(article)

interacted_articles_dict = {a.article_id: a for a in interacted_articles}


# preparing resulting dict
persona_wise_rec_recall = defaultdict(lambda: {"daily_scores": []})


# setting parameters for different condition
static_num_recs = 10
# topical_pref_only || clicked_topic_personas || topical_click_only
variation = "topical_click_only"
pipeline = "nrms_topic_scores"
def_pref = 3

# synthetic data generation
synthetic_personas = synthetic_personas_generator(def_pref, variation, interacted_articles)

# day to day caddidate article
for day in tqdm(cadidate_dates):
    day_df = articles_df[articles_df["published_at"].dt.normalize() == day]

    candidate_articles = []
    for row in day_df.itertuples():
        article = complete_article_generator(row, mentions_df)
        candidate_articles.append(article)

    if len(candidate_articles) < static_num_recs:
        continue

    # taking each persona and generating full recommendation request based on topical preference
    # and interacted article as well as passing all the candidate articles for that day.
    # finally passing the pipeline and generating recommendation response.
    for persona_topic, persona_profile in synthetic_personas.items():
        req = full_request_generator(persona_profile, interacted_articles_dict, candidate_articles, static_num_recs)

        response = root(req.model_dump(), pipeline=pipeline)
        response = RecommendationResponseV2.model_validate(response)
        response = response.model_dump()

        # calculating the daily recall and precision for each persona
        recall, precision = daily_persona_wise_recall_calculator(candidate_articles, persona_topic, response)

        persona_wise_rec_recall[persona_topic]["daily_scores"].append((day, precision, recall))


# calculating persona wise avg recall over days
avg_persona_wise_rec_recall = avg_persona_wise_rec_recall_over_days_calculator(persona_wise_rec_recall)

# store result in CSV
store_rec_recall_as_csv(avg_persona_wise_rec_recall, pipeline, variation, def_pref)


# for persona, values in avg_persona_wise_rec_recall.items():
#     print(f"\nTopic: {persona}|| Precision: {values["avg_precision"]:.5f}|| Recall: {values["avg_recall"]:.5f}")
