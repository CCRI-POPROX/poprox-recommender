import io
import json

import pandas as pd
import torch as th
from smart_open import open as smart_open

from poprox_recommender.default import select_articles
from poprox_recommender.paths import project_root


def load_model(device_name=None):
    buffer = None
    checkpoint = None

    if device_name is None:
        device_name = "cuda" if th.cuda.is_available() else "cpu"

    load_path = f"{project_root()}/models/ckpt-30000.pth"
    with smart_open(load_path, "rb") as f:
        buffer = io.BytesIO(f.read())

    if buffer:
        device = th.device(device_name)
        checkpoint = th.load(buffer, map_location=device)

    return checkpoint, device


def load_token_mapping():
    token_mapping = None

    token_mapping_path = f"{project_root()}/models/word2int.tsv"
    with smart_open(token_mapping_path, "rb") as f:
        token_mapping = dict(pd.read_table(f, na_filter=False).values.tolist())

    return token_mapping


MODEL, DEVICE = load_model()
TOKEN_MAPPING = load_token_mapping()


def hello(event, context):
    request_body = json.loads(event["body"])

    todays_articles = request_body["todays_articles"]
    past_articles = request_body["past_articles"]
    click_data = request_body["click_data"]
    num_recs = request_body["num_recs"]

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
        "recommendations": recommendations,
    }

    response = {"statusCode": 200, "body": json.dumps(body)}

    return response
