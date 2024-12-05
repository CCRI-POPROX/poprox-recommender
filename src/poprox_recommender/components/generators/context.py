from datetime import datetime

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from poprox_concepts import Article, ArticleSet
from poprox_recommender.lkpipeline import Component
from poprox_recommender.topics import extract_general_topics

model = SentenceTransformer("all-MiniLM-L6-v2")

dev_mode = True

if not dev_mode:
    client = OpenAI(
        api_key="Put your key here",
    )


class ContextGenerator(Component):
    def __init__(self, text_generation=False, time_decay=True, topk_similar=5, other_filter="topic"):
        self.text_generation = text_generation
        self.time_decay = time_decay
        self.topk_similar = topk_similar
        self.other_filter = other_filter

    def __call__(self, clicked: ArticleSet, recommended: ArticleSet) -> ArticleSet:
        if not dev_mode:
            for article in recommended.articles:
                generated_subhead = generated_context(
                    article, clicked, self.time_decay, self.topk_similar, self.other_filter
                )
                article.subhead = generated_subhead

        return recommended


def generated_context(
    article: Article, clicked_articles: ArticleSet, time_decay: bool, topk_similar: int, other_filter: str | None = None
):
    # TODO: add fallback that based on user interests

    topk_similar = min(topk_similar, len(clicked_articles.articles))
    related_articles = related_context(article, clicked_articles, time_decay, topk_similar, other_filter)

    input_prompt = []
    input_prompt.append({"ID": "Main News", "subhead": article.subhead})

    for i in range(topk_similar):
        input_prompt.append({"ID": "Related News", "subhead": related_articles[i].subhead})

    generated_subhead = generate_narrative(input_prompt)
    return generated_subhead


def related_context(
    article: Article, clicked: ArticleSet, time_decay: bool, topk_similar: int, other_filter: str | None = None
):
    selected_subhead = article.subhead
    selected_date = article.published_at
    selected_topic = extract_general_topics(article)

    if other_filter == "topic":
        filtered_candidates = [
            candidate for candidate in clicked.articles if set(extract_general_topics(candidate)) & set(selected_topic)
        ]
        clicked_articles = filtered_candidates if filtered_candidates else clicked.articles

    else:
        clicked_articles = clicked.articles

    candidate_indices = related_indices(selected_subhead, selected_date, clicked_articles, time_decay, topk_similar)

    return [clicked_articles[index] for index in candidate_indices]


def related_indices(
    selected_subhead: str, selected_date: datetime, clicked_articles: list, time_decay: bool, topk_similar: int
):
    all_subheads = [selected_subhead] + [article.subhead for article in clicked_articles]
    embeddings = model.encode(all_subheads)

    target_embedding = embeddings[0].reshape(1, -1)
    clicked_embeddings = embeddings[1:]
    similarities = cosine_similarity(target_embedding, clicked_embeddings)[0]

    if time_decay:
        weights = [
            get_time_weight(selected_date, published_date)
            for published_date in [article.published_at for article in clicked_articles]
        ]
        weighted_similarities = similarities * weights
        return np.argsort(weighted_similarities)[-topk_similar:][::-1]

    return np.argsort(similarities)[-topk_similar:][::-1]


def generate_narrative(news_list):
    system_prompt = (
        "You are a personalized text generator."
        " First, i will provide you with a news list that"
        " includes both the [Main News] and [Related News]."
        " Based on the input news list and user interests,"
        " please generate a new personalized news summary centered around the [Main News]."
    )

    input_prompt = "News List: \n" + f"{news_list}"
    return gpt_generate(system_prompt, input_prompt)


def get_time_weight(published_target, published_clicked):
    time_distance = abs((published_clicked - published_target).days)
    weight = 1 / np.log(1 + time_distance) if time_distance > 0 else 1  # Avoid log(1) when x = 0
    return weight


###################### text generation part
def gpt_generate(system_prompt, content_prompt):
    message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": content_prompt}]
    temperature = 0.2
    max_tokens = 512
    frequency_penalty = 0.0

    chat_completion = client.chat.completions.create(
        messages=message,
        temperature=temperature,
        max_tokens=max_tokens,
        frequency_penalty=frequency_penalty,
        model="gpt-4o-mini",
    )
    return chat_completion.choices[0].message.content


"""
# TODO: check backward or forward for past k articles
def user_interest_generate(past_articles: Article, past_k: int):
    system_prompt = (
        "You are asked to describe user interest based on his/her browsed news list."
        " User interest includes the news [categories] and news [topics]"
        " (under each [category] that users are interested in."
    )

    return gpt_generate(system_prompt, f"{past_article_infor}")
"""
