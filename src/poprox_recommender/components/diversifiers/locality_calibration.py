import datetime
from collections import defaultdict

import numpy as np
import torch as th
import time
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from poprox_concepts import Article, ArticleSet, InterestProfile
from poprox_recommender.components.diversifiers.calibration import compute_kl_divergence
from poprox_recommender.lkpipeline import Component
from poprox_recommender.topics import extract_general_topics, extract_locality, normalized_category_count

# Only uncomment this in offline theta value exploration
# KL_VALUE_PATH = '/home/sun00587/research/News_Locality_Polarization/poprox-recommender-locality/outputs/theta_kl_values_11-17.txt'

MAX_RETRIES = 3
DELAY = 2
SEMANTIC_THRESHOLD = 0.2

class LocalityCalibrator(Component):
    def __init__(self, theta_local: float = 0.1, theta_topic: float = 0.1, num_slots=10):
        """
        TODOs: If set different theta_topic and theta_local values for different users,
        then can save them in interest_profile
        """
        self.theta_local = theta_local
        self.theta_topic = theta_topic
        self.num_slots = num_slots

    def __call__(self, candidate_articles: ArticleSet, interest_profile: InterestProfile, theta_topic: float, theta_locality: float) -> ArticleSet:
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
            theta_locality,
            topk=self.num_slots,
        )

        # Save computed kl divergence for topic and locality
        # Only uncomment this in offline theta value exploration
        # with open(KL_VALUE_PATH, 'a') as file:
        #     file.write('{}_top_{}_loc_{},{},{}\n'.format(str(interest_profile.profile_id), theta_topic, theta_locality, final_calibrations[0], final_calibrations[1]))

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
        
        normalized_locality_pres = normalized_category_count(locality_preferences)
        return normalized_locality_pres

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
'''
Logic behind context generation: 
find the top1 clicked article with semantic similarity above a pre-defined threshold 
if all clicked articles have semantic similarity below the threshold, then generate using the general topic interests 
'''

model = SentenceTransformer("all-MiniLM-L6-v2")

client = OpenAI(
    api_key="Put your key here",
)

def gpt_generate(system_prompt, content_prompt):
    retries = 0 
    message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": content_prompt}]
    temperature = 0.2
    max_tokens = 512
    frequency_penalty = 0.0

    while retries < MAX_RETRIES: 
        try:
            chat_completion = client.chat.completions.create(
                messages=message,
                temperature=temperature,
                max_tokens=max_tokens,
                frequency_penalty=frequency_penalty,
                model="gpt-4o-mini",
            )
            return chat_completion.choices[0].message.content
        
        except Exception as e: 
            print(f'Fail to call OPENAI API: {e}')
            retries += 1
            if retries < MAX_RETRIES: 
                print(f"{retries} try to regenerate the context")
                time.sleep(DELAY)
            else: 
                raise


def get_time_weight(published_target, published_clicked):
    time_distance = abs((published_clicked - published_target).days)
    weight = 1 / np.log(1 + time_distance) if time_distance > 0 else 1  # Avoid log(1) when x = 0
    return weight


def related_indices(
    selected_subhead: str, selected_date: datetime, clicked_articles: list, time_decay: bool, topk_similar: int
):
    all_subheads = [selected_subhead] + [article.subhead for article in clicked_articles] 
    #is_original = [1 if article.is_original == 1 else 0 for article in clicked_articles]

    embeddings = model.encode(all_subheads)

    target_embedding = embeddings[0].reshape(1, -1)
    clicked_embeddings = embeddings[1:]
    similarities = cosine_similarity(target_embedding, clicked_embeddings)[0]
    #similarities = [similarities[i] * is_original[i] for i in range(len(is_original))]

    if time_decay:
        weights = [
            get_time_weight(selected_date, published_date)
            for published_date in [article.published_at for article in clicked_articles]
        ]
        similarities = similarities * weights

    # check if has similarity above the threshold 
    if np.max(similarities) < SEMANTIC_THRESHOLD:
        # If no articles are similar enough, 
        # we'll default to a general appraoch that 
        # uses the topk similar articles to generate a broad topic inspired context for the rec?
        return np.argsort(similarities)[-topk_similar:][::-1]
    else: 
        return np.argsort(similarities)[-1:][::-1]

    


def related_context(
    article: Article, clicked_articles: ArticleSet, time_decay: bool, topk_similar: int
):
    selected_subhead = article.subhead
    selected_date = article.published_at

    clicked_articles = clicked_articles.articles

    candidate_indices = related_indices(selected_subhead, selected_date, clicked_articles, time_decay, topk_similar)

    return [clicked_articles[index] for index in candidate_indices]


def generated_context(
    article: Article, clicked_articles: ArticleSet, time_decay: bool, topk_similar: int
):

    topk_similar = min(topk_similar, len(clicked_articles.articles))
    related_articles = related_context(article, clicked_articles, time_decay, topk_similar)

    if len(related_articles) == 1: 
        # high similarity, use the article to rewrite the subhead 

        news_list = {'MAIN NEWS': article.subhead , 
                     'RELATED NEWS': related_articles[0].subhead}
        
        input_prompt = f"{news_list}"

        generated_subhead = semantic_narrative(input_prompt)

    else: 
        news_list = {'MAIN NEWS': article.subhead, 
                     'RELATED NEWS': [related_articles[i].subhead for i in range(topk_similar)]}
        
        input_prompt = f"{news_list}"
        generated_subhead = highlevel_narrative(input_prompt)

    return generated_subhead


def semantic_narrative(news_list):
    system_prompt = ("You are an expert to rewrite the subheadline of MAIN NEWS in a natural and factural tone. "
    "I will provide you with two news articles: MAIN NEWS and RELATED NEWS that occurred before the MAIN NEWS. "
    "Please rewrite the subhead of MAIN NEWS by incorporting the user interests detected from the RELATED NEWS "
    "and how the RELATED NEWS set the stage for the MAIN NEWS. "
    "Please ensure that you maintain a logical flow, and present in a neutral and fatual tone in the rewritten subhead."
    )
    
    input_prompt = "News List: \n" + f"{news_list}"
    return gpt_generate(system_prompt, input_prompt)

def highlevel_narrative(news_list):
    system_prompt = ("You are an expert to rewrite the subheadline of MAIN NEWS in a natural and factural tone. "
    "I will provide you with MAIN NEWS and a list of RELATED NEWS that occurred before the MAIN NEWS."
    "Please rewrite the subhead of MAIN NEWS by incorporting the user interests detected from the RELATED NEWS "
    "and how the RELATED NEWS set the stage for the MAIN NEWS. "
    "Please ensure that you maintain a logical flow, and present in a neutral and fatual tone in the rewritten subhead."
    )
    
    input_prompt = "News List: \n" + f"{news_list}"
    return gpt_generate(system_prompt, input_prompt)