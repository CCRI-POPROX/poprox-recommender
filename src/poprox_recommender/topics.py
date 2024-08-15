from collections import defaultdict
from uuid import UUID

from torch.nn.functional import cosine_similarity
from transformers import AutoModel, AutoTokenizer

from poprox_concepts import Article, Click
from poprox_concepts.domain.topics import GENERAL_TOPICS
from poprox_recommender.paths import model_file_path

NEWS_MODEL_NAME = "news-category-classifier-distilbert"


def extract_general_topics(article: Article):
    article_topics = set([mention.entity.name for mention in article.mentions])
    return article_topics.intersection(GENERAL_TOPICS)


def find_topic(past_articles: list[Article], article_id: UUID):
    # each article might correspond to multiple topic
    for article in past_articles:
        if article.article_id == article_id:
            return extract_general_topics(article)


def normalized_topic_count(topic_counts: dict[str, int]):
    total_count = sum(topic_counts.values())
    normalized_counts = {key: value / total_count for key, value in topic_counts.items()}
    return normalized_counts


def user_topic_preference(past_articles: list[Article], click_history: list[Click]) -> dict[str, int]:
    """Topic preference only based on click history"""
    clicked_articles = [c.article_id for c in click_history]  # List[UUID]

    topic_count_dict = defaultdict(int)

    for article_id in clicked_articles:
        clicked_topics = find_topic(past_articles, article_id) or set()
        for topic in clicked_topics:
            topic_count_dict[topic] += 1

    return topic_count_dict


def classify_news_topic(model, tokenizer, general_topics, topic):
    inputs = tokenizer.batch_encode_plus([topic] + general_topics, return_tensors="pt", pad_to_max_length=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    output = model(input_ids, attention_mask=attention_mask)[0]

    sentence_rep = output[:1].mean(dim=1)
    label_reps = output[1:].mean(dim=1)

    # now find the labels with the highest cosine similarities to the sentence
    similarities = cosine_similarity(sentence_rep, label_reps)
    closest = similarities.argsort(descending=True)
    classified_topics = []
    for ind in closest:
        classified_topics.append(general_topics[ind])
    return classified_topics


def match_news_topics_to_general(articles: list[Article]):
    # Load the pre-trained tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        model_file_path(NEWS_MODEL_NAME)
    )  # a highly downloaded fine-tuned model on news
    model = AutoModel.from_pretrained(model_file_path(NEWS_MODEL_NAME))

    topic_matched_dict = {}  # we might be able to connect this to DB to read previous matching?
    article_to_new_topic = {}

    for article in articles:
        news_topics = [mention.entity.name for mention in article.mentions]
        article_topic = set()
        for topic in news_topics:
            if topic not in topic_matched_dict:
                matched_general_topics = classify_news_topic(model, tokenizer, GENERAL_TOPICS, topic)
                topic_matched_dict[topic] = matched_general_topics  # again, we can store this into db
                for t in matched_general_topics:
                    article_topic.add(t)
        article_to_new_topic[article.article_id] = article_topic
    return topic_matched_dict, article_to_new_topic
