import json

import torch as th
from safetensors.torch import load_file

from poprox_concepts import Article, ClickHistory
from poprox_recommender.default import select_articles
from poprox_recommender.paths import project_root

from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import sigmoid


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
    topic_matched_dict, todays_article_matched_topics = match_news_topics_to_general(todays_articles)

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
        todays_article_matched_topics
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

def classify_news_topic(model, tokenizer, general_topics, topic):
    inputs = tokenizer(topic, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = sigmoid(logits).squeeze().detach().numpy()
    
    # Threshold for classification.
    threshold = 0.5
    classified_topics = [general_topics[i] for i, prob in enumerate(probabilities) if prob > threshold]

    return classified_topics

def match_news_topics_to_general(articles):
    general_topics = [
        "US News",
        "World News",
        "Politics",
        "Business",
        "Entertainment",
        "Sports",
        "Health",
        "Science",
        "Tech ",
        "Lifestyle",
        "Religion",
        "Climate",
        "Education",
        "Oddities",
    ]
    # Load the pre-trained tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(general_topics))
    topic_matched_dict = {} # we might be able to connect this to DB to read previous matching?
    article_to_new_topic = {}
    for article in articles:
        news_topics = [mention.entity.name for mention in article.mentions]
        article_topic = set()
        for topic in news_topics:
            if topic not in topic_matched_dict:
                matched_general_topics = classify_news_topic(model, tokenizer, general_topics, topic)
                topic_matched_dict[topic] = matched_general_topics # again, we can store this into db
                for t in matched_general_topics:
                    article_topic.add(t)
        article_to_new_topic[article.article_id] = article_topic
    return topic_matched_dict, article_to_new_topic