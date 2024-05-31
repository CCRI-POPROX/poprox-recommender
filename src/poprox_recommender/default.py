import random
from dataclasses import dataclass
from functools import partial
from typing import Dict, List
from uuid import UUID

import swifter  # noqa: F401 # pylint: disable=unused-import
import numpy as np
import pandas as pd
import torch as th
import sys

sys.path.append("../")
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer

from poprox_concepts import Article, ClickHistory
from poprox_recommender.model.nrms import NRMS


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


def parse_row(token_mapping, row):
    new_row = [str(row.article_id), []]

    try:
        # token_mapping is the name of pre-trained tokenizer
        tokenizer = AutoTokenizer.from_pretrained(token_mapping)
        new_row[1] = tokenizer.encode(
            row.title, padding="max_length", max_length=30, truncation=True
        )

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
def transform_article_features(articles: List[Article], token_mapping):

    article_df = pd.DataFrame([article.model_dump() for article in articles])
    article_df.fillna(" ", inplace=True)
    parse_fn = partial(parse_row, token_mapping)

    return article_df.swifter.apply(parse_fn, axis=1)


def load_model(checkpoint, device):
    model = NRMS(ModelConfig()).to(device)

    model.load_state_dict(checkpoint)
    model.eval()

    return model


# Compute a vector for each news story
def build_article_embeddings(article_features, model, device):
    articles = {
        "id": article_features["article_id"],
        "title": th.tensor(article_features["title"]),
    }
    article_embeddings = {}
    article_vectors = model.get_news_vector(articles["title"].to(device))
    for article_id, article_vector in zip(articles["id"], article_vectors):
        if article_id not in article_embeddings:
            article_embeddings[article_id] = article_vector

    article_embeddings["PADDED_NEWS"] = th.zeros(
        list(article_embeddings.values())[0].size(), device=device
    )
    return article_embeddings, article_vectors


def build_clicks_df(click_history: ClickHistory):
    user_df = pd.DataFrame()
    user_df["user"] = [str(click_history.account_id)]
    user_df["clicked_news"] = [[str(id_) for id_ in click_history.article_ids]]
    return user_df


# Compute a vector for each user
def build_user_embeddings(
    user_df, article_embeddings, model, device, max_clicks_per_user
):
    user_embeddings = {}
    for _, row in user_df.iterrows():
        clicked_news = list(
            dict.fromkeys(row.clicked_news)
        )  # deduplicate while maintaining order
        user = {
            "user": row.user,
            "clicked_news": clicked_news[-max_clicks_per_user:],
        }
        user["clicked_news_length"] = len(user["clicked_news"])
        repeated_times = max_clicks_per_user - len(user["clicked_news"])
        assert repeated_times >= 0
        user["clicked_news"] = ["PADDED_NEWS"] * repeated_times + user["clicked_news"]
        default = article_embeddings["PADDED_NEWS"]
        clicked_article_embeddings = [
            article_embeddings.get(clicked_article, default).to(device)
            for clicked_article in user["clicked_news"]
        ]
        clicked_news_vector = (
            th.stack(
                clicked_article_embeddings,
                dim=0,
            )
            .unsqueeze(0)
            .to(device)
        )

        user_vector = model.get_user_vector(clicked_news_vector)
        user_embeddings[row.user] = user_vector
    return user_embeddings


def mmr_diversification(rewards, similarity_matrix, theta, topk):
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

        if candidate != None:
            S.append(candidate)
    return S  # LIST(candidate index)


def generate_recommendations(
    model, articles, article_vectors, similarity_matrix, user_embeddings, num_slots=10
):
    recommendations = {}
    for user, user_vector in user_embeddings.items():
        pred = model.get_prediction(article_vectors, user_vector.squeeze())
        pred = pred.cpu().detach().numpy()
        recs = mmr_diversification(pred, similarity_matrix, theta=1, topk=num_slots)
        recommendations[user] = [articles[int(rec)] for rec in recs]
    return recommendations


def select_with_model(
    todays_articles: List[Article],
    todays_article_vectors: List[Article],
    article_similarity_matrix,
    past_article_features,
    click_history: ClickHistory,
    model,
    model_device,
    num_slots=10,
    max_clicks_per_user=50,
):
    # Translate clicks JSON to dataframe
    user_df = build_clicks_df(click_history)

    # Build embedding tables
    past_article_embeddings, _ = build_article_embeddings(
        past_article_features, model, model_device
    )

    user_embeddings = build_user_embeddings(
        user_df, past_article_embeddings, model, model_device, max_clicks_per_user
    )

    recommendations = generate_recommendations(
        model,
        todays_articles,
        todays_article_vectors,
        article_similarity_matrix,
        user_embeddings,
        num_slots=num_slots,
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
                similarity_matrix[i, j] = similarity_matrix[j, i] = np.dot(
                    value1, value2
                )
    return similarity_matrix


def select_articles(
    todays_articles: List[Article],
    past_articles: List[Article],
    click_histories: List[ClickHistory],
    model_checkpoint,
    model_device,
    token_mapping,
    num_slots,
):

    # Transform news to model features
    todays_article_features = transform_article_features(todays_articles, token_mapping)

    # Extract the set of articles that have actually been clicked
    clicked_article_ids = set()
    for history in click_histories:
        clicked_article_ids.update(history.article_ids)

    clicked_articles = [
        article
        for article in past_articles
        if article.article_id in clicked_article_ids
    ]

    # Convert clicked article attributes into model features
    past_article_features = transform_article_features(
        clicked_articles,
        token_mapping,
    )

    # Load model
    model = load_model(model_checkpoint, model_device)

    # Compute today's article similarity matrix
    _, todays_article_vectors = build_article_embeddings(
        todays_article_features, model, model_device
    )
    similarity_matrix = compute_similarity_matrix(todays_article_vectors)

    recommendations = {}
    for click_history in click_histories:
        account_id = click_history.account_id
        if model_checkpoint and token_mapping and click_history.article_ids:
            user_recs = select_with_model(
                todays_articles,
                todays_article_vectors,
                similarity_matrix,
                past_article_features,
                click_history,
                model,
                model_device,
                num_slots=num_slots,
            )
            recommendations[account_id] = user_recs[str(account_id)]
        else:
            recommendations[account_id] = random.sample(todays_articles, num_slots)

    return recommendations
