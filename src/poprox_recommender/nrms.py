import random
from dataclasses import dataclass
from functools import partial

import swifter  # noqa: F401 # pylint: disable=unused-import
import numpy as np
import pandas as pd
import torch as th

from tqdm import tqdm
from nltk.tokenize import word_tokenize

from poprox_recommender.model.nrms import NRMS


@dataclass
class ModelConfig:
    num_epochs: float = 10
    num_batches_show_loss: float = 100
    num_batches_validate: float = 1000
    batch_size: float = 128
    learning_rate: float = 0.0001
    num_workers: float = 4
    num_clicked_news_a_user: float = 50
    word_freq_threshold: float = 1
    dropout_probability: float = 0.2
    word_embedding_dim: float = 300
    category_embedding_dim: float = 100
    query_vector_dim: float = 200
    num_attention_heads: float = 15

    num_words: int = 101222


def parse_row(token_mapping, row):
    new_row = [row.url, [0] * 20]

    try:
        for i, w in enumerate(word_tokenize(row.title.lower())):
            if w in token_mapping:
                new_row[1][i] = token_mapping[w]

    except IndexError:
        pass

    return pd.Series(
        new_row,
        index=["url", "title"],
    )


# prepare the news.tsv with information for NRMS model
def transform_article_features(ap_news_df, token_mapping):
    ap_news_df.fillna(" ", inplace=True)
    parse_fn = partial(parse_row, token_mapping)
    return ap_news_df.swifter.apply(parse_fn, axis=1)


def load_model(checkpoint, device):
    model = NRMS(ModelConfig()).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


# Compute a vector for each news story
def build_article_embeddings(article_features, model, device):
    articles = {
        "id": article_features["url"],
        "title": th.tensor(article_features["title"]),
    }
    article_embeddings = {}
    article_vectors = model.get_news_vector(articles)
    for article_id, article_vector in zip(articles["id"], article_vectors):
        if article_id not in article_embeddings:
            article_embeddings[article_id] = article_vector

    article_embeddings["PADDED_NEWS"] = th.zeros(list(article_embeddings.values())[0].size(), device=device)
    return article_embeddings, article_vectors


def build_clicks_df(user_clicked):
    # Currently, every impression contains ALL candidates
    user_df = pd.DataFrame()
    user_df["user"] = list(user_clicked.keys())
    user_df["clicked_news"] = [" ".join(sublist) for sublist in list(user_clicked.values())]
    return user_df


# Compute a vector for each user
def build_user_embeddings(user_df, article_embeddings, model, device, max_clicks_per_user):
    user_embeddings = {}
    for _, row in user_df.iterrows():
        user = {
            "user": row.user,
            "clicked_news_string": row.clicked_news,
            "clicked_news": row.clicked_news.split()[-max_clicks_per_user:],
        }
        user["clicked_news_length"] = len(user["clicked_news"])
        repeated_times = max_clicks_per_user - len(user["clicked_news"])
        assert repeated_times >= 0
        user["clicked_news"] = ["PADDED_NEWS"] * repeated_times + user["clicked_news"]

        default = article_embeddings["PADDED_NEWS"]
        clicked_news_vector = (
            th.stack(
                [
                    th.stack(
                        [article_embeddings.get(x, default).to(device) for x in [news_list]],
                        dim=0,
                    )
                    for news_list in user["clicked_news"]
                ],
                dim=0,
            )
            .transpose(0, 1)
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

    for k in range(topk):
        candidate = None
        best_MR = float("-inf")
        for i, reward_i in enumerate(rewards):  # iterate R
            max_sim = float("-inf")

            for j in S:
                sim = similarity_matrix[i][j]
                if sim >= max_sim:
                    max_sim = sim

            mr_i = theta * reward_i - (1 - theta) * max_sim
            if mr_i >= best_MR and i not in S:
                best_MR = mr_i
                candidate = i
        S.append(candidate)

    return S


def generate_recommendations(model, articles, article_vectors, similarity_matrix, user_embeddings, num_slots=10):
    recommendations = {}
    for user, user_vector in user_embeddings.items():
        pred = model.get_prediction(article_vectors, user_vector.squeeze())
        pred = pred.cpu().detach().numpy()
        recs = mmr_diversification(pred, similarity_matrix, theta=0.9, topk=num_slots)
        recommendations[user] = [articles[int(rec)] for rec in recs]
    return recommendations


def select_with_model(
    article_similarity_matrix,
    past_article_features,
    clicked_articles,
    model,
    model_device,
    num_slots=10,
    max_clicks_per_user=10,
):
    # Translate clicks JSON to dataframe
    user_df = build_clicks_df(clicked_articles)

    # Build embedding tables
    past_article_embeddings, _ = build_article_embeddings(past_article_features, model, model_device)
    user_embeddings = build_user_embeddings(user_df, past_article_embeddings, model, model_device, max_clicks_per_user)

    recommendations = generate_recommendations(
        model,
        article_similarity_matrix,
        user_embeddings,
        num_slots=num_slots,
    )

    return recommendations


def select_articles(
    todays_articles,
    past_articles,
    click_data,
    model_checkpoint,
    model_device,
    token_mapping,
    num_slots=10,
):
    # Transform news to model features
    todays_article_features = transform_article_features(
        pd.DataFrame(todays_articles),
        token_mapping,
    )

    # Extract the set of articles that have actually been clicked
    clicked_urls = set()
    for clicks in click_data.values():
        clicked_urls.update(clicks)

    clicked_articles = [article for article in past_articles if article["url"] in clicked_urls]

    # Convert clicked article attributes into model features
    past_article_features = transform_article_features(
        pd.DataFrame(clicked_articles),
        token_mapping,
    )

    # Load model
    model = load_model(model_checkpoint, model_device)

    # Compute today's article similarity matrix
    _, todays_article_vectors = build_article_embeddings(todays_article_features, model, model_device)
    num_values = len(todays_article_vectors)
    similarity_matrix = np.zeros((num_values, num_values))
    for i, value1 in tqdm(enumerate(todays_article_vectors), total=num_values):
        value1 = value1.detach().cpu()
        for j, value2 in enumerate(todays_article_vectors):
            if i <= j:
                value2 = value2.detach().cpu()
                similarity_matrix[i, j] = similarity_matrix[j, i] = np.dot(value1, value2)

    recommendations = {}
    for user, clicks in click_data.items():
        if model_checkpoint and token_mapping and clicks:
            user_recs = select_with_model(
                todays_articles,
                todays_article_vectors,
                similarity_matrix,
                past_article_features,
                {user: clicks},
                model,
                model_device,
                num_slots=num_slots,
            )
            recommendations[user] = user_recs[user]
        else:
            recommendations[user] = random.sample(todays_articles, num_slots)

    return recommendations
