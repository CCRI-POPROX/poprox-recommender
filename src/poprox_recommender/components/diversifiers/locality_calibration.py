import datetime
from collections import defaultdict

import numpy as np
import torch as th
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from poprox_concepts import Article, ArticleSet, InterestProfile
from poprox_recommender.components.diversifiers.calibration import compute_kl_divergence
from poprox_recommender.lkpipeline import Component
from poprox_recommender.topics import extract_general_topics, extract_locality, normalized_category_count


class LocalityCalibrator(Component):
    def __init__(self, theta_local: float = 0.1, theta_topic: float = 0.1, num_slots=10):
        """
        TODOs: If set different theta_topic and theta_local values for different users, then can save them in interest_profile
        """
        self.theta_local = theta_local
        self.theta_topic = theta_topic
        self.num_slots = num_slots

    def __call__(self, candidate_articles: ArticleSet, interest_profile: InterestProfile) -> ArticleSet:
        normalized_topic_prefs = self.compute_topic_prefs(interest_profile)
        normalized_locality_prefs = self.compute_local_prefs(candidate_articles)

        if candidate_articles.scores is not None:
            article_scores = th.sigmoid(th.tensor(candidate_articles.scores))
        else:
            article_scores = th.zeros(len(candidate_articles.articles))

        article_scores = article_scores.cpu().detach().numpy()

        article_indices = self.calibration(
            article_scores,
            candidate_articles.articles,
            normalized_topic_prefs,
            normalized_locality_prefs,
            theta_topic,
            self.theta_local,
            topk=self.num_slots,
        )

        return ArticleSet(
            articles=[candidate_articles.articles[idx] for idx in article_indices]
        )  # all selected articles

    def add_article_to_categories(self, rec_topics, article):
        rec_topics = rec_topics.copy()
        topics = extract_general_topics(article)
        for topic in topics:
            rec_topics[topic] = rec_topics[topic] + 1
        return rec_topics

    def add_article_to_localities(self, rec_localities, article):
        rec_localities = rec_localities.copy()
        localities = extract_locality(article)
        for local in localities:
            rec_localities[local] = rec_localities[local] + 1
        return rec_localities

    def normalized_categories_with_candidate(self, rec_categories, article):
        rec_categories_with_candidate = rec_categories.copy()
        rec_categories_with_candidate = self.add_article_to_categories(rec_categories_with_candidate, article)
        return normalized_category_count(rec_categories_with_candidate)

    def normalized_localities_with_candidate(self, rec_localities, article):
        rec_localities_with_candidate = rec_localities.copy()
        rec_localities_with_candidate = self.add_article_to_localities(rec_localities_with_candidate, article)
        return normalized_category_count(rec_localities_with_candidate)

    def calibration(
        self, relevance_scores, articles, topic_preferences, locality_preferences, theta_topic, theta_local, topk
    ) -> list[Article]:
        # MR_i = (1 - theta_topic - theta_local) * reward_i - theta_topic * C_topic - theta_local * C_local
        # R is all candidates (not selected yet)

        recommendations = []  # final recommendation (topk index)

        topic_categories = defaultdict(int)
        local_categories = defaultdict(int)

        for _ in range(topk):
            candidate = None  # next item
            best_candidate_score = float("-inf")

            for article_idx, article_score in enumerate(relevance_scores):  # iterate R for next item
                if article_idx in recommendations:
                    continue

                normalized_candidate_topics = self.normalized_categories_with_candidate(
                    topic_categories, articles[article_idx]
                )
                normalized_candidate_locality = self.normalized_localities_with_candidate(
                    local_categories, articles[article_idx]
                )

                calibration_topic = compute_kl_divergence(topic_preferences, normalized_candidate_topics)
                calibration_local = compute_kl_divergence(locality_preferences, normalized_candidate_locality)

                # TODO or other MOE
                adjusted_candidate_score = (
                    (1 - theta_local - theta_topic) * article_score
                    - (theta_topic * calibration_topic)
                    - (theta_local * calibration_local)
                )
                if adjusted_candidate_score > best_candidate_score:
                    best_candidate_score = adjusted_candidate_score
                    candidate = article_idx

            if candidate is not None:
                recommendations.append(candidate)
                topic_categories = self.add_article_to_categories(topic_categories, articles[candidate])
                local_categories = self.add_article_to_localities(local_categories, articles[candidate])

        return recommendations

    def compute_local_prefs(self, candidate_articles: ArticleSet):
        locality_preferences: dict[str, int] = defaultdict(int)
        candidate_articles = candidate_articles.articles

        for article in candidate_articles:
            candidate_locality = extract_locality(article) or set()
            for locality in candidate_locality:
                locality_preferences[locality] += 1

        return locality_preferences

    def compute_topic_prefs(self, interest_profile):
        topic_preferences: dict[str, int] = defaultdict(int)

        for interest in interest_profile.onboarding_topics:
            topic_preferences[interest.entity_name] = max(interest.preference - 1, 0)

        if interest_profile.click_topic_counts:
            for topic, click_count in interest_profile.click_topic_counts.items():
                topic_preferences[topic] += click_count

        normalized_topic_prefs = normalized_category_count(topic_preferences)
        return normalized_topic_prefs


###################### text generation part
model = SentenceTransformer("all-MiniLM-L6-v2")

client = OpenAI(
    api_key="Put your key here",
)


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


def related_context(
    article: Article, clicked_articles: ArticleSet, time_decay: bool, topk_similar: int, other_filter: None
):
    selected_subhead = article.subhead
    selected_date = article.published_at
    selected_topic = extract_general_topics(article)

    if other_filter == "topic":
        filtered_candidates = [
            candidate
            for candidate in clicked_articles.articles
            if set(extract_general_topics(candidate)) & set(selected_topic)
        ]
        clicked_articles = filtered_candidates if filtered_candidates else clicked_articles.articles

    else:
        clicked_articles = clicked_articles.articles

    candidate_indices = related_indices(selected_subhead, selected_date, clicked_articles, time_decay, topk_similar)

    return [clicked_articles[index] for index in candidate_indices]


def generated_context(
    article: Article, clicked_articles: ArticleSet, time_decay: bool, topk_similar: int, other_filter: None
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
