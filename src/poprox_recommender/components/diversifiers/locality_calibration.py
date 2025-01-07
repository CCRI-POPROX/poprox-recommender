from collections import defaultdict

import numpy as np
import torch as th
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from poprox_concepts import Article, ArticleSet, InterestProfile
from poprox_recommender.lkpipeline import Component
from poprox_recommender.topics import extract_general_topics, extract_locality, normalized_category_count


def compute_kl_divergence(interacted_distr, reco_distr, kl_div=0.0, alpha=0.01):
    """
    KL (p || q), the lower the better.

    alpha is not really a tuning parameter, it's just there to make the
    computation more numerically stable.
    """
    for category, score in interacted_distr.items():
        reco_score = reco_distr.get(category, 0.0)
        reco_score = (1 - alpha) * reco_score + alpha * score
        if reco_score != 0.0:
            kl_div += score * np.log2(score / reco_score)

    return kl_div


class LocalityCalibrator(Component):
    def __init__(self, theta_local: float = 0.1, theta_topic: float = 0.1, num_slots=10):
        self.theta_local = theta_local
        self.theta_topic = theta_topic
        self.num_slots = num_slots

    def __call__(self, candidate_articles: ArticleSet, interest_profile: InterestProfile) -> ArticleSet:
        normalized_topic_prefs = self.compute_topic_dist(interest_profile)
        normalized_locality_prefs = self.compute_local_dist(candidate_articles)

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
            self.theta_topic,
            self.theta_local,
            topk=self.num_slots,
        )

        recommended_list = text_generation(candidate_articles.articles, article_indices, interest_profile)
        return ArticleSet(articles=recommended_list)

    def add_article_to_categories(self, rec_topics, article):
        topics = extract_general_topics(article)
        for topic in topics:
            rec_topics[topic] = rec_topics.get(topic, 0) + 1

    def add_article_to_localities(self, rec_localities, article):
        localities = extract_locality(article)
        for local in localities:
            rec_localities[local] = rec_localities.get(local, 0) + 1

    def normalized_categories_with_candidate(self, rec_categories, article):
        rec_categories_with_candidate = rec_categories.copy()
        self.add_article_to_categories(rec_categories_with_candidate, article)
        return normalized_category_count(rec_categories_with_candidate)

    def normalized_localities_with_candidate(self, rec_localities, article):
        rec_localities_with_candidate = rec_localities.copy()
        self.add_article_to_localities(rec_localities_with_candidate, article)
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
                self.add_article_to_categories(topic_categories, articles[candidate])
                self.add_article_to_localities(local_categories, articles[candidate])

        return recommendations

    def compute_local_dist(self, candidate_articles: ArticleSet):
        locality_preferences: dict[str, int] = defaultdict(int)
        candidate_articles = candidate_articles.articles

        for article in candidate_articles:
            candidate_locality = extract_locality(article) or set()
            for locality in candidate_locality:
                locality_preferences[locality] += 1

        return locality_preferences

    def compute_topic_dist(self, interest_profile):
        topic_preferences: dict[str, int] = defaultdict(int)

        for interest in interest_profile.onboarding_topics:
            topic_preferences[interest.entity_name] = max(interest.preference - 1, 0)

        if interest_profile.click_topic_counts:
            for topic, click_count in interest_profile.click_topic_counts.items():
                topic_preferences[topic] += click_count

        normalized_topic_prefs = normalized_category_count(topic_preferences)
        return normalized_topic_prefs


###################### text generation part

client = OpenAI(
    api_key="PUT KEY HERE",
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


def news_factor(headline, slug):
    news = f"headline: {headline}\nslug: {slug}"
    system_prompt = (
        "Based on the given news information, summarize what topics that the news is related to."
        "Each news article is related to 1-3 topics and each topic shuold not exceed five words."
    )
    return gpt_generate(system_prompt, news)


# TODO: check backward or forward for past k articles
def user_interest_generate(past_articles: Article, past_k: int):
    past_article_infor = []
    past_count = 0
    for article in past_articles.articles:
        past_count += 1
        if past_count == past_k:
            break
        temp_article = {}
        temp_article["headline"] = article.headline
        temp_article["subhead"] = article.subhead
        temp_article["categories"] = set([mention.entity.name for mention in article.mentions])
        temp_article["topics"] = news_factor(article.headline, article.subhead)
        past_article_infor.append(temp_article)

    system_prompt = (
        "You are asked to describe user interest based on his/her browsed news list."
        " User interest includes the news [categories] and news [topics]"
        " (under each [category] that users are interested in."
    )

    return gpt_generate(system_prompt, f"{past_article_infor}")


def generate_narrative(news_list, user_interests):
    system_prompt = (
        "You are a personalized text generator."
        " First, i will provide you with a news list that"
        " includes both the [main news] and [related news]."
        " Second, I will provide you with user interests,"
        " including the [categories] of news that the user is interested in."
        " Based on the input news list and user interests,"
        " you are required to generate a personalized news summary centered around the [main news]."
    )

    input_prompt = "News List: \n" + f"{news_list}" + "User Interest: \n" + user_interests
    return gpt_generate(system_prompt, input_prompt)


model = SentenceTransformer("all-MiniLM-L6-v2")


def related_article(current_news_slug, current_topics, clicked_articles, k):
    # TODO: do we need to consider the cold start?
    target_embedding = model.encode([current_news_slug])[0]

    subject_candidate_headlines = []
    for idx in range(len(clicked_articles)):
        slug = clicked_articles[int(idx)].subhead
        # category = set([mention.entity.name for mention in clicked_articles[int(idx)].mentions])

        # if any(item in current_topics for item in category):
        subject_candidate_headlines.append(slug)

    candidate_embeddings = model.encode(subject_candidate_headlines)
    similarities = cosine_similarity([target_embedding], candidate_embeddings)[0]
    top_k_indices = np.argsort(similarities)[-k:][::-1]

    return top_k_indices


def text_generation(candidate_articles, article_indices, interest_profile):
    clicked_articles = interest_profile.clicked_articles.articles

    recommended = []
    for idx in article_indices:
        news_list = []  # for context generation
        current_news_slug = candidate_articles[int(idx)].subhead
        current_topics = set([mention.entity.name for mention in candidate_articles[int(idx)].mentions])

        temp_article = {}
        temp_article["ID"] = "Main news"
        temp_article["headline"] = candidate_articles[int(idx)].headline
        temp_article["subhead"] = candidate_articles[int(idx)].subhead
        temp_article["categories"] = set([mention.entity.name for mention in candidate_articles[int(idx)].mentions])
        temp_article["topics"] = news_factor(
            candidate_articles[int(idx)].headline, candidate_articles[int(idx)].subhead
        )
        news_list.append(temp_article)

        # TODO: determine the k (3) value
        clicked_articles_idx = related_article(current_news_slug, current_topics, clicked_articles, 3)

        for idx_clicked in clicked_articles_idx:
            temp_article = {}
            temp_article["ID"] = "related articles"
            temp_article["headline"] = clicked_articles[int(idx_clicked)].headline
            temp_article["subhead"] = clicked_articles[int(idx_clicked)].subhead
            temp_article["categories"] = set(
                [mention.entity.name for mention in clicked_articles[int(idx_clicked)].mentions]
            )
            temp_article["topics"] = news_factor(
                clicked_articles[int(idx_clicked)].headline, clicked_articles[int(idx_clicked)].subhead
            )
            news_list.append(temp_article)

        candidate_articles[int(idx)].headline = generate_narrative(news_list, interest_profile.user_interests)
        recommended.append(candidate_articles)
    return recommended[0]
