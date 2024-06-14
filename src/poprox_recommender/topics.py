from transformers import AutoTokenizer, AutoModel
from torch.nn.functional import cosine_similarity

from poprox_concepts import Article

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

def classify_news_topic(model, tokenizer, general_topics, topic):
    inputs = tokenizer.batch_encode_plus([topic] + general_topics, return_tensors='pt', pad_to_max_length=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
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
    tokenizer = AutoTokenizer.from_pretrained('dima806/news-category-classifier-distilbert') # a highly downloaded fine-tuned model on news
    model = AutoModel.from_pretrained('dima806/news-category-classifier-distilbert')
    
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

