import json

import torch as th
from safetensors.torch import load_file

from poprox_concepts import Article, ClickHistory
from poprox_recommender.default import select_articles
from poprox_recommender.paths import project_root


def load_model(device_name=None):
    checkpoint = None

    if device_name is None:
        device_name = "cuda" if th.cuda.is_available() else "cpu"

    load_path = f"{project_root()}/models/model.safetensors"

    checkpoint = load_file(load_path)
    return checkpoint, device_name


MODEL, DEVICE = load_model()
TOKEN_MAPPING = "distilbert-base-uncased"  # can be modified


def generate_recs(event, context):
    req_body = json.loads(event["body"])

    todays_articles = [
        Article.model_validate(attrs) for attrs in req_body["todays_articles"]
    ]
    past_articles = [
        Article.model_validate(attrs) for attrs in req_body["past_articles"]
    ]
    click_data = [
        ClickHistory.model_validate(attrs) for attrs in req_body["click_data"]
    ]
    num_recs = req_body["num_recs"]

    recommendations = select_articles(
        todays_articles,
        past_articles,
        click_data,
        MODEL,
        DEVICE,
        TOKEN_MAPPING,
        num_recs,
    )

    body = {
        "recommendations": {
            str(account_id): [
                json.loads(article.model_dump_json()) for article in articles
            ]
            for account_id, articles in recommendations.items()
        },
    }

    response = {"statusCode": 200, "body": json.dumps(body)}

    return response
