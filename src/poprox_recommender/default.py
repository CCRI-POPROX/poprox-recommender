import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from typing import Any
from uuid import UUID

import numpy as np
import pandas as pd
import swifter  # noqa: F401 # pylint: disable=unused-import
import torch as th

sys.path.append("../")
from safetensors.torch import load_file
from tqdm import tqdm
from transformers import AutoTokenizer

from poprox_concepts import Article, ClickHistory, InterestProfile
from poprox_recommender.model.nrms import NRMS
from poprox_recommender.paths import model_file_path
from poprox_recommender.topics import extract_general_topic


@dataclass
class ModelConfig:
    num_epochs: float = 10

    num_clicked_news_a_user: float = 50
    word_freq_threshold: float = 1
    dropout_probability: float = 0.2
    word_embedding_dim: float = 300
    category_embedding_dim: float = 100
    query_vector_dim: float = 200
    additive_attn_hidden_dim: float = 200
    num_attention_heads: float = 16
    hidden_size: int = 768

    pretrained_model = "distilbert-base-uncased"


def load_checkpoint(device_name=None):
    checkpoint = None

    if device_name is None:
        # device_name = "cuda" if th.cuda.is_available() else "cpu"
        device_name = "cpu"

    load_path = model_file_path("model.safetensors")

    checkpoint = load_file(load_path)
    return checkpoint, device_name


def load_model(checkpoint, device):
    model = NRMS(ModelConfig()).to(device)
    model.load_state_dict(checkpoint)
    model.eval()

    return model


TOKENIZER = AutoTokenizer.from_pretrained("distilbert-base-uncased", cache_dir="/tmp/")
CHECKPOINT, DEVICE = load_checkpoint()
MODEL = load_model(CHECKPOINT, DEVICE)


def parse_row(tokenizer, row):
    new_row = [str(row.article_id), []]

    try:
        new_row[1] = tokenizer.encode(row.title, padding="max_length", max_length=30, truncation=True)

    except IndexError:
        pass

    return pd.Series(
        new_row,
        index=["article_id", "title"],
    )


# check the input type of articles
def is_list_of_dicts(articles):
    if isinstance(articles, list) and all(isinstance(item, dict) for item in articles):
        return True
    return False


# prepare the news.tsv with information for NRMS model
def transform_article_features(articles: list[Article], tokenizer):
    article_df = pd.DataFrame([article.model_dump() for article in articles])
    article_df.fillna(" ", inplace=True)
    parse_fn = partial(parse_row, tokenizer)

    return article_df.swifter.apply(parse_fn, axis=1)


# Compute a vector for each news story
def build_article_embeddings(article_features, model, device):
    articles = {
        "id": article_features["article_id"],
        "title": th.tensor(article_features["title"]),
    }
    article_embeddings = {}
    article_vectors = model.get_news_vector(articles["title"].to(device))
    for article_id, article_vector in zip(articles["id"], article_vectors, strict=False):
        if article_id not in article_embeddings:
            article_embeddings[article_id] = article_vector

    article_embeddings["PADDED_NEWS"] = th.zeros(list(article_embeddings.values())[0].size(), device=device)
    return article_embeddings, article_vectors


# Compute a vector for each user
def build_user_embedding(click_history: ClickHistory, article_embeddings, model, device, max_clicks_per_user):
    article_ids = list(dict.fromkeys(click_history.article_ids))[
        -max_clicks_per_user:
    ]  # deduplicate while maintaining order

    padded_positions = max_clicks_per_user - len(article_ids)
    assert padded_positions >= 0

    article_ids = ["PADDED_NEWS"] * padded_positions + article_ids
    default = article_embeddings["PADDED_NEWS"]
    clicked_article_embeddings = [
        article_embeddings.get(clicked_article, default).to(device) for clicked_article in article_ids
    ]
    clicked_news_vector = (
        th.stack(
            clicked_article_embeddings,
            dim=0,
        )
        .unsqueeze(0)
        .to(device)
    )

    return model.get_user_vector(clicked_news_vector)


def mmr_diversification(rewards, similarity_matrix, theta: float, topk: int):
    # MR_i = \theta * reward_i - (1 - \theta)*max_{j \in S} sim(i, j) # S us
    # R is all candidates (not selected yet)

    S = []  # final recommendation (topk index)
    # first recommended item
    S.append(rewards.argmax())

    for k in range(topk - 1):
        candidate = None  # next item
        best_MR = float("-inf")

        for i, reward_i in enumerate(rewards):  # iterate R for next item
            if i in S:
                continue
            max_sim = float("-inf")

            for j in S:
                sim = similarity_matrix[i][j]
                if sim > max_sim:
                    max_sim = sim

            mr_i = theta * reward_i - (1 - theta) * max_sim
            if mr_i > best_MR:
                best_MR = mr_i
                candidate = i

        if candidate is not None:
            S.append(candidate)
    return S


def pfar_diversification(rewards, articles, topic_preferences, lamb, tau, topk):
    # p(v|u) + lamb*tau \sum_{d \in D} P(d|u)I{v \in d} \prod_{i \in S} I{i \in d} for each user

    S = []  # final recommendation LIST[candidate index]
    initial_item = rewards.argmax()
    S.append(initial_item)

    S_topic = set()
    article = articles[int(initial_item)]
    S_topic.update([mention.entity.name for mention in article.mentions])

    for k in range(topk - 1):
        candidate = None
        best_PFAR = float("-inf")

        for i, reward_i in enumerate(rewards):  # iterate R for next item
            if i in S:
                continue
            product = 1
            summation = 0

            candidate_topics = [mention.entity.name for mention in articles[int(i)].mentions]
            for topic in candidate_topics:
                if topic in S_topic:
                    product = 0
                    break

            for topic in candidate_topics:
                if topic in topic_preferences:
                    summation += topic_preferences[topic]

            PFAR_i = reward_i + lamb * tau * summation * product

            if PFAR_i > best_PFAR:
                best_PFAR = PFAR_i
                candidate = i

        if candidate is not None:
            S.append(candidate)
            S_topic.update(candidate_topics)

    return S  # LIST(candidate index)


def generate_recommendations(
    model,
    articles,
    article_vectors,
    similarity_matrix,
    user_embedding,
    interest_profile: InterestProfile,
    num_slots: int = 10,
    algo_params: dict[str, Any] | None = None,
) -> list[Article]:
    algo_params = algo_params or {}
    theta = float(algo_params.get("theta", 0.8))
    lamb = float(algo_params.get("pfar_lamb", 1))
    tau = float(algo_params.get("pfar_tau", 1))
    diversify = str(algo_params.get("diversity_algo", "mmr"))

    pred = model.get_prediction(article_vectors, user_embedding.squeeze())
    pred = pred.cpu().detach().numpy()
    if diversify == "mmr":
        recs = mmr_diversification(pred, similarity_matrix, theta=theta, topk=num_slots)
    if diversify == "pfar":
        topic_preferences: dict[str, int] = {}

        for interest in interest_profile.onboarding_topics:
            topic_preferences[interest.entity_name] = max(interest.preference - 1, 0)

        for topic, click_count in interest_profile.click_topic_counts.items():
            topic_preferences[topic] = click_count

        normalized_topic_prefs = normalized_topic_count(topic_preferences)

        recs = pfar_diversification(pred, articles, normalized_topic_prefs, lamb, tau, topk=num_slots)

    return [articles[int(rec)] for rec in recs]


def select_with_model(
    todays_articles: list[Article],
    todays_article_vectors: list[Article],
    article_similarity_matrix,
    past_article_features,
    interest_profile: InterestProfile,
    model,
    model_device,
    num_slots: int = 10,
    max_clicks_per_user: int = 50,
    algo_params: dict[str, Any] | None = None,
):
    # Build embedding tables
    past_article_embeddings, _ = build_article_embeddings(past_article_features, model, model_device)

    user_embedding = build_user_embedding(
        interest_profile.click_history,
        past_article_embeddings,
        model,
        model_device,
        max_clicks_per_user,
    )

    recommendations = generate_recommendations(
        model,
        todays_articles,
        todays_article_vectors,
        article_similarity_matrix,
        user_embedding,
        interest_profile,
        num_slots=num_slots,
        algo_params=algo_params,
    )

    return recommendations


def compute_similarity_matrix(todays_article_vectors):
    num_values = len(todays_article_vectors)
    similarity_matrix = np.zeros((num_values, num_values))
    for i, value1 in tqdm(enumerate(todays_article_vectors), total=num_values):
        value1 = value1.detach().cpu()
        for j, value2 in enumerate(todays_article_vectors):
            if i <= j:
                value2 = value2.detach().cpu()
                similarity_matrix[i, j] = similarity_matrix[j, i] = np.dot(value1, value2)
    return similarity_matrix


def find_topic(past_articles: list[Article], article_id: UUID):
    # each article might correspond to multiple topic
    for article in past_articles:
        if article.article_id == article_id:
            return extract_general_topic(article)


def normalized_topic_count(topic_counts: dict[str, int]):
    total_count = sum(topic_counts.values())
    normalized_counts = {key: value / total_count for key, value in topic_counts.items()}
    return normalized_counts


def user_topic_preference(past_articles: list[Article], click_history: ClickHistory) -> dict[str, int]:
    """Topic preference only based on click history"""
    clicked_articles = click_history.article_ids  # List[UUID]

    topic_count_dict = defaultdict(int)

    for article_id in clicked_articles:
        clicked_topics = find_topic(past_articles, article_id)
        for topic in clicked_topics:
            topic_count_dict[topic] += 1

    return topic_count_dict


def select_articles(
    todays_articles: list[Article],
    past_articles: list[Article],
    interest_profile: InterestProfile,
    num_slots: int,
    algo_params: dict[str, Any] | None = None,
) -> dict[UUID, list[Article]]:
    click_history = interest_profile.click_history

    # Transform news to model features
    todays_article_features = transform_article_features(todays_articles, TOKENIZER)

    clicked_articles = filter(lambda a: a.article_id in set(click_history.article_ids), past_articles)

    # Convert clicked article attributes into model features
    past_article_features = transform_article_features(
        clicked_articles,
        TOKENIZER,
    )

    # Compute today's article similarity matrix
    _, todays_article_vectors = build_article_embeddings(todays_article_features, MODEL, DEVICE)
    similarity_matrix = compute_similarity_matrix(todays_article_vectors)

    interest_profile.click_topic_counts = user_topic_preference(past_articles, interest_profile.click_history)

    recommendations = {}
    account_id = click_history.account_id
    if MODEL and TOKENIZER and click_history.article_ids:
        user_recs = select_with_model(
            todays_articles,
            todays_article_vectors,
            similarity_matrix,
            past_article_features,
            interest_profile,
            MODEL,
            DEVICE,
            num_slots=num_slots,
            algo_params=algo_params,
        )
        recommendations[account_id] = user_recs
    else:
        recommendations[account_id] = random.sample(todays_articles, num_slots)

    return recommendations
