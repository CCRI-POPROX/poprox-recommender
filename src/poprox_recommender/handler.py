import json

from poprox_concepts.api.recommendations import (
    RecommendationRequest,
    RecommendationResponse,
)
from poprox_recommender.default import select_articles


def generate_recs(event, context):

    req = RecommendationRequest.model_validate_json(event["body"])

    recommendations = select_articles(
        req.todays_articles,
        req.past_articles,
        req.click_histories,
        req.num_recs,
    )

    resp_body = RecommendationResponse.model_validate(
        {"recommendations": recommendations}
    )

    # Dumping to JSON serializes UUIDs properly but requests
    # wants a Python data structure as the body. There's gotta
    # be a better way, but this workaround bridges the gap for now.
    response = {"statusCode": 200, "body": json.loads(resp_body.model_dump_json())}

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