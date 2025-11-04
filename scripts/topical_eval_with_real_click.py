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


def full_request_generator(user_profile, interacted_articles_dict, candidate_articles, static_num_recs):
    clicked_article = []
    for click in user_profile.click_history:
        clicked_article.append(interacted_articles_dict[click.article_id])

    full_request = {
        "interest_profile": user_profile.model_dump(),
        "interacted": CandidateSet(articles=clicked_article),
        "candidates": CandidateSet(articles=candidate_articles),
        "num_recs": static_num_recs,
    }
    return RecommendationRequestV2.model_validate(full_request)


def store_rec_recall_as_csv(sorted_user_wise_rec_recall, variation, time_frame):
    csv_rows = [("Topic", "Avg_Precision", "Avg_Recall")]
    for persona, values in sorted_user_wise_rec_recall.items():
        csv_rows.append((persona, f"{values['avg_precision']:.5f}", f"{values['avg_recall']:.5f}"))

    with open(data / f"{variation}_{time_frame}.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_rows)


# utility functions for synthetic data generation


def user_profile_generator(user_account_ID, CLK_df, interest_df):
    user_profile_batch = {}

    for uuid in user_account_ID:
        user_clicks = [
            Click(
                article_id=clk.article_id,
                newsletter_id=clk.newsletter_id,
                timestamp=clk.clicked_at,
            )
            for clk in CLK_df
            if clk.profile_id == uuid
        ]
        user_interests = [
            AccountInterest(
                account_id=interest.account_id,
                entity_id=interest.entity_id,
                entity_name=interest.entity_name,
                preference=interest.preference,
            )
            for interest in interest_df
            if interest.account_id == uuid
        ]
        user_profile = InterestProfile(
            profile_id=uuid,
            click_history=user_clicks,
            click_topic_counts=None,
            click_locality_counts=None,
            article_feedbacks={},
            onboarding_topics=user_interests,
        )
        user_profile_batch[uuid] = user_profile

    return user_profile_batch


# all the calculation functions


def daily_user_wise_JSD(candidate_articles, uuid, response):
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


def avg_user_wise_rec_recall_over_days_calculator(persona_wise_rec_recall):
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
data = project_root() / "data" / "Test_Real_Click"
Cand_articles_df = pd.read_parquet(data / "articles.parquet")
Cand_mentions_df = pd.read_parquet(data / "mentions.parquet")
cadidate_dates = sorted(Cand_articles_df["published_at"].dt.normalize().unique())

CLK_df = pd.read_parquet(data / "click.parquet")
CLK_account_ID = CLK_df["profile_id"].unique()

interest_df = pd.read_parquet(data / "insterests.parquet")
interest_account_ID = interest_df["account_id"].unique()

user_account_ID = set(CLK_account_ID) | set(interest_account_ID)

Clk_articles_df = pd.read_parquet(data / "clicked" / "articles.parquet")
Clk_mentions_df = pd.read_parquet(data / "clicked" / "mentions.parquet")


# preparing interacted article
interacted_articles = []

for row in tqdm(Clk_articles_df.itertuples()):
    article = complete_article_generator(row, Clk_mentions_df)
    interacted_articles.append(article)

interacted_articles_dict = {a.article_id: a for a in interacted_articles}


# preparing resulting dict
user_wise_rec_recall = defaultdict(lambda: {"daily_scores": []})


# setting parameters for different condition
static_num_recs = 10
# topical_pref_only || clicked_topic_personas || topical_click_only

variation = "topical_pref_only"
pipeline = "nrms_topic_scores"
def_pref = 3

# topic_embeddings_cand_11_months || topic_embeddings_cand_15_15_days ||
# topic_embeddings_cand_15_days   || topic_embeddings_cand_30_days ||
# topic_embeddings_def_llm || topic_embeddings_hybrid
time_frame = "topic_embeddings_hybrid_AP"

# synthetic data generation
user_profiles = user_profile_generator(user_account_ID, CLK_df, interest_df)

# day to day caddidate article
for day in tqdm(cadidate_dates):
    day_df = Cand_articles_df[Cand_articles_df["published_at"].dt.normalize() == day]

    candidate_articles = []
    for row in day_df.itertuples():
        article = complete_article_generator(row, Cand_mentions_df)
        candidate_articles.append(article)

    if len(candidate_articles) < static_num_recs:
        continue

    # taking each persona and generating full recommendation request based on topical preference
    # and interacted article as well as passing all the candidate articles for that day.
    # finally passing the pipeline and generating recommendation response.
    for uuid, user_profile in user_profiles.items():
        req = full_request_generator(user_profile, interacted_articles_dict, candidate_articles, static_num_recs)

        response = root(req.model_dump(), pipeline=pipeline)
        response = RecommendationResponseV2.model_validate(response)
        response = response.model_dump()

        # calculating the daily recall and precision for each persona
        recall, precision = daily_user_wise_JSD(candidate_articles, uuid, response)

        user_wise_rec_recall[uuid]["daily_scores"].append((day, precision, recall))


# calculating persona wise avg recall over days
avg_user_wise_rec_recall = avg_user_wise_rec_recall_over_days_calculator(user_wise_rec_recall)

# store result in CSV
store_rec_recall_as_csv(avg_user_wise_rec_recall, variation, time_frame)


# for persona, values in avg_persona_wise_rec_recall.items():
#     print(f"\nTopic: {persona}|| Precision: {values['avg_precision']:.5f}|| Recall: {values['avg_recall']:.5f}")
