import json
import math
import os
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from poprox_concepts import AccountInterest, Article, CandidateSet, Click, Entity, InterestProfile, Mention
from poprox_concepts.api.recommendations.v2 import RecommendationRequestV2, RecommendationResponseV2
from poprox_recommender.api.main import root
from poprox_recommender.components.topical_description import TOPIC_DESCRIPTIONS
from poprox_recommender.paths import project_root

# utility functions for article generator, rec_request generator


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


# utility functions for user profile generation


def user_profile_generator(clk_df, interest_df, min_CLK, max_CLK):
    user_profile_batch = {}

    for uuid in interest_df["account_id"].unique():
        user_clicks = [
            Click(
                article_id=clk["article_id"],
                newsletter_id=clk["newsletter_id"],
                timestamp=clk["clicked_at"],
            )
            for _, clk in clk_df.iterrows()
            if clk["profile_id"] == uuid
        ]

        if not (min_CLK <= len(user_clicks) <= max_CLK):
            continue

        user_interests = [
            AccountInterest(
                account_id=interest["account_id"],
                entity_id=interest["entity_id"],
                entity_name=interest["entity_name"],
                preference=interest["preference"],
            )
            for _, interest in interest_df.iterrows()
            if interest["account_id"] == uuid
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


def daily_user_wise_JSD(user_profile, user_topic_counts):
    topical_pref_dist = {t.entity_name: t.preference for t in user_profile.onboarding_topics}
    pref_total = sum(topical_pref_dist.values())
    topical_pref_dist_norm = {k: (v / pref_total) if pref_total > 0 else 0 for k, v in topical_pref_dist.items()}

    rec_topical_dist_norm = {}
    rec_topic_total = sum(user_topic_counts.values())
    for topic in topical_pref_dist_norm.keys():
        rec_topical_dist_norm[topic] = user_topic_counts.get(topic, 0) / rec_topic_total if rec_topic_total > 0 else 0

    kl_p_m = 0.0
    kl_q_m = 0.0

    for t in topical_pref_dist.keys():
        p = topical_pref_dist_norm[t]
        q = rec_topical_dist_norm[t]
        m = 0.5 * (p + q)

        if p > 0 and m > 0:
            kl_p_m += p * math.log2(p / m)
        if q > 0 and m > 0:
            kl_q_m += q * math.log2(q / m)

    jsd = 0.5 * kl_p_m + 0.5 * kl_q_m

    return jsd, topical_pref_dist_norm, rec_topical_dist_norm


# utility functions for data storage


def compute_jsd_and_generate_outputs(
    user_wise_rec_result,
    csv_path="TP_user_jsd_topic_distributions.csv",
    plots_dir="TP_jsd_user_topic_plots",
):
    rows_for_csv = []

    os.makedirs(plots_dir, exist_ok=True)

    for uuid, (jsd, topical_pref_dist_norm, rec_topical_dist_norm) in user_wise_rec_result.items():
        row = {"uuid": str(uuid), "jsd": jsd}

        for topic in topical_pref_dist_norm.keys():
            row[f"pref_{topic}"] = topical_pref_dist_norm[topic]
            row[f"rec_{topic}"] = rec_topical_dist_norm.get(topic, 0)

        rows_for_csv.append(row)

        topics = list(topical_pref_dist_norm.keys())
        pref_vals = [topical_pref_dist_norm[t] for t in topics]
        rec_vals = [rec_topical_dist_norm[t] for t in topics]

        x = np.arange(len(topics))
        width = 0.6

        fig, ax = plt.subplots(figsize=(8, 4))

        # Background prefs
        ax.bar(x, pref_vals, width=width, alpha=0.4, label="Topical Prefs")
        # Foreground recs
        ax.bar(x, rec_vals, width=width * 0.7, label="Recommended Articles")

        ax.set_xticks(x)
        ax.set_xticklabels(topics, rotation=45, ha="right")
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1)
        ax.set_title(f"Topic distributions for user {uuid}\nJSD = {jsd:.3f}")
        ax.legend()
        fig.tight_layout()

        fig.savefig(os.path.join(plots_dir, f"{uuid}_topics.png"))
        plt.close(fig)

    df = pd.DataFrame(rows_for_csv)
    df.to_csv(csv_path, index=False)

    return df


topics = {t for t in TOPIC_DESCRIPTIONS.keys()}

data = project_root() / "data" / "Test_Real_Click"
cand_articles_df = pd.read_parquet(data / "articles.parquet")
cand_mentions_df = pd.read_parquet(data / "mentions.parquet")
candidate_dates = sorted(cand_articles_df["published_at"].dt.normalize().unique())
last_30_days_candidate = candidate_dates[-30:]


clk_df = pd.read_parquet(data / "clicks.parquet")
interest_df = pd.read_parquet(data / "interests.parquet")


clk_articles_df = pd.read_parquet(data / "clicked" / "articles.parquet")
clk_mentions_df = pd.read_parquet(data / "clicked" / "mentions.parquet")
clk_dates = sorted(clk_articles_df["published_at"].dt.normalize().unique())
candidate_excluded_clk_dates = clk_dates[:-30]


clk_article_filtered = clk_articles_df[
    clk_articles_df["published_at"].dt.normalize().isin(candidate_excluded_clk_dates)
]
clk_filtered = clk_df[clk_df["article_id"].isin(clk_article_filtered["article_id"])]


interacted_articles = []

for row in clk_article_filtered.itertuples():
    article = complete_article_generator(row, clk_mentions_df)
    interacted_articles.append(article)

interacted_articles_dict = {a.article_id: a for a in interacted_articles}


static_num_recs = 10
pipeline = "nrms_topic_scores"


# topic_embeddings_cand_11_months || topic_embeddings_def || topic_embeddings_def_llm
# topic_embeddings_hybrid || topic_embeddings_llm_hybrid
# topic_embed_method = "topic_embeddings_cand"


user_wise_rec_result = {}


min_CLK = 10
max_CLK = 50
users_profile = user_profile_generator(clk_filtered, interest_df, min_CLK, max_CLK)


user_topic_counts = {uuid: defaultdict(int) for uuid in users_profile.keys()}


topic_counts_30d = Counter({topic: 0 for topic in topics})

for day in tqdm(last_30_days_candidate):
    day_df = cand_articles_df[cand_articles_df["published_at"].dt.normalize() == day]

    candidate_articles = []
    for row in day_df.itertuples():
        article = complete_article_generator(row, cand_mentions_df)
        candidate_articles.append(article)

    if len(candidate_articles) < static_num_recs:
        continue

    for article in candidate_articles:
        article_topics = {m.entity.name for m in article.mentions if m.entity.name in topics}
        topic_counts_30d.update(article_topics)

    # taking each persona and generating full recommendation request based on topical preference
    # and interacted article as well as passing all the candidate articles for that day.
    # finally passing the pipeline and generating recommendation response.
    for uuid, user_profile in users_profile.items():
        req = full_request_generator(user_profile, interacted_articles_dict, candidate_articles, static_num_recs)

        response = root(req.model_dump(), pipeline=pipeline)
        response = RecommendationResponseV2.model_validate(response)
        response = response.model_dump()

        articles = response["recommendations"]["articles"]

        for article in articles:
            article_topics = {m["entity"]["name"] for m in article["mentions"] if m["entity"]["name"] in topics}

            for topic in article_topics:
                user_topic_counts[uuid][topic] += 1


for uuid, user_profile in users_profile.items():
    jsd, topical_pref_dist_norm, rec_topical_dist_norm = daily_user_wise_JSD(user_profile, user_topic_counts[uuid])

    user_wise_rec_result[uuid] = (jsd, topical_pref_dist_norm, rec_topical_dist_norm)

df = compute_jsd_and_generate_outputs(user_wise_rec_result)
