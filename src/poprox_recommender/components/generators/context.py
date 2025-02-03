import asyncio
import time
from datetime import datetime, timedelta

import numpy as np
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from poprox_concepts import Article, ArticleSet, InterestProfile
from poprox_recommender.components.diversifiers.locality_calibration import (
    LocalityCalibrator,
)
from poprox_recommender.lkpipeline import Component
from poprox_recommender.paths import model_file_path

MAX_RETRIES = 3
DELAY = 2
SEMANTIC_THRESHOLD = 0.2
BASELINE_THETA_TOPIC = 0.3
NUM_TOPICS = 3
DAYS = 4
USED_indices = set()


class ContextGenerator(Component):
    def __init__(self, text_generation=False, time_decay=True, dev_mode="true"):
        self.text_generation = text_generation
        self.time_decay = time_decay
        self.dev_mode = dev_mode
        if self.dev_mode:
            self.client = AsyncOpenAI(
                api_key="PUT YOUR ACCESS KEY HERE",
            )
        self.model = SentenceTransformer(str(model_file_path("all-MiniLM-L6-v2")))

    def __call__(
        self,
        clicked: ArticleSet,
        recommended: ArticleSet,
        interest_profile: InterestProfile,
    ) -> ArticleSet:

        if self.dev_mode:
            recommended = asyncio.run(self.generate_newsletter(clicked, recommended, interest_profile))
        return recommended

    async def generate_newsletter(
        self,
        clicked: ArticleSet,
        recommended: ArticleSet,
        interest_profile: InterestProfile,
    ):
        topic_distribution = LocalityCalibrator.compute_topic_prefs(interest_profile)
        treatment = recommended.treatment_flags
        tasks = []

        for i in range(len(recommended.articles)):
            article = recommended.articles[i]
            if treatment[i]:
                task = self.generated_context(
                    article, clicked, self.time_decay, topic_distribution
                )
                tasks.append((article, task))

        results = await asyncio.gather(
            *(task[1] for task in tasks), return_exceptions=True
        )

        for (article, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                print(f"Error generating context for article: {result}")
            else:
                article.subhead = result
        
        return recommended
    
    async def generated_context(
        self,
        article: Article,
        clicked_articles: ArticleSet,
        time_decay: bool,
        topic_distribution: dict,
    ):
        related_articles = self.related_context(article, clicked_articles, time_decay)

        if len(related_articles) == 1:
            # high similarity, use the top-1 article to rewrite the subhead
            news_list = {"MAIN NEWS": article.subhead, "RELATED NEWS": related_articles[0].subhead}

            input_prompt = f"{news_list}"
            generated_subhead = await self.semantic_narrative(input_prompt)

        else:
            generated_subhead = await self.highlevel_narrative(article.subhead, topic_distribution)

        return generated_subhead

    def related_context(
        self,
        article: Article,
        clicked: ArticleSet,
        time_decay: bool,
    ):
        selected_subhead = article.subhead
        selected_date = article.published_at

        clicked_articles = clicked.articles
        time0 = selected_date - timedelta(days=DAYS)

        clicked_articles = [
            article for article in clicked_articles if article.published_at >= time0
        ]

        candidate_indices = self.related_indices(selected_subhead, selected_date, clicked_articles, time_decay)
        if len(candidate_indices) == 0:
            return []

        return [clicked_articles[index] for index in candidate_indices]

    def related_indices(
        self,
        selected_subhead: str,
        selected_date: datetime,
        clicked_articles: list,
        time_decay: bool,
    ):
        all_subheads = [selected_subhead] + [
            article.subhead for article in clicked_articles
        ]
        embeddings = self.model.encode(all_subheads)

        target_embedding = embeddings[0].reshape(1, -1)
        clicked_embeddings = embeddings[1:]
        similarities = cosine_similarity(target_embedding, clicked_embeddings)[0]

        # CHECK threshold [0.2, 0, 0.2]
        for i in range(len(similarities)):
            val = similarities[i]
            if val < SEMANTIC_THRESHOLD or i in USED_indices:
                similarities[i] = 0

        if np.sort(similarities)[-1] < SEMANTIC_THRESHOLD:
            return []

        elif time_decay:
            weights = [
                self.get_time_weight(selected_date, published_date)
                for published_date in [
                    article.published_at for article in clicked_articles
                ]
            ]
            weighted_similarities = similarities * weights

            selected_indices = np.argsort(weighted_similarities)[-1:]
            USED_indices.add(selected_indices)
            return selected_indices

        else:
            selected_indices = np.argsort(weighted_similarities)[-1:]
            USED_indices.add(selected_indices)
            return selected_indices

    async def semantic_narrative(self, news_list):
        system_prompt = (
            "You are an editor to rewrite the MAIN NEWS in a natural and factual tone. "
            "You are provided a MAIN NEWS to be recommended and a RELATED NEWS that a user read before. "
            "Please rewrite the MAIN NEWS by implicitly connecting it to RELATED NEWS and "
            "incorporating the relevant user interests detected from RELATED NEWS. "
            "Please ensure that the rewritten MAIN NEWS is presented concisely in a neutral and factual tone."
        )

        input_prompt = "News List: \n" + f"{news_list}"
        return await self.gpt_generate(system_prompt, input_prompt)

    async def highlevel_narrative(self, main_news, topic_distribution):
        sorted_items = sorted(
            topic_distribution.items(), key=lambda item: item[1], reverse=True
        )
        top_keys = [key for key, _ in sorted_items[:NUM_TOPICS]]

        system_prompt = (
            "You are an editor to rewrite the MAIN NEWS in one sentence, using a natural and factual tone. "
            "You are provided a MAIN NEWS to be recommended and user INTERESTED TOPICS. "
            "Please rewrite the MAIN NEWS to make it more attractive, "
            "and the user can feel that the news is more inclined to his INTERESTED TOPICS. "
            "Please ensure that the rewritten MAIN NEWS is presented concisely and narratively. "
            "Make sure the rewritten MAIN NEWS is more attractive. "
        )

        news_list = {"MAIN NEWS": main_news, "INTERESTED TOPICS": top_keys}

        input_prompt = f"{news_list}"
        return await self.gpt_generate(system_prompt, input_prompt)

    def get_time_weight(self, published_target, published_clicked):
        time_distance = abs((published_clicked - published_target).days)
        weight = (
            1 / np.log(1 + time_distance) if time_distance > 0 else 1
        )  # Avoid log(1) when x = 0
        return weight

    async def gpt_generate(self, system_prompt, content_prompt):
        retries = 0
        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_prompt},
        ]
        temperature = 0.2
        max_tokens = 256
        frequency_penalty = 0.0

        while retries < MAX_RETRIES:
            try:
                chat_completion = await self.client.chat.completions.create(
                    messages=message,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    frequency_penalty=frequency_penalty,
                    model="gpt-4o-mini",
                )
                return chat_completion.choices[0].message.content

            except Exception as e:
                print(f"Fail to call OPENAI API: {e}")
                retries += 1
                if retries < MAX_RETRIES:
                    print(f"{retries} try to regenerate the context")
                    await asyncio.sleep(DELAY)
                else:
                    raise
