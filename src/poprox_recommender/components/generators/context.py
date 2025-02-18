import ast
import asyncio
import json
import logging
from datetime import datetime, timedelta
import re

import numpy as np
from lenskit.pipeline import Component
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from poprox_concepts.domain import (
    Article,
    CandidateSet,
    InterestProfile,
    RecommendationList,
)
from poprox_recommender.components.diversifiers.locality_calibration import (
    LocalityCalibrator,
)
from poprox_recommender.paths import model_file_path

MAX_RETRIES = 3
DELAY = 2
SEMANTIC_THRESHOLD = 0.2
BASELINE_THETA_TOPIC = 0.3
NUM_TOPICS = 3
DAYS = 14

logger = logging.getLogger(__name__)


class ContextGenerator(Component):
    def __init__(self, text_generation=False, time_decay=True, dev_mode="true"):
        self.text_generation = text_generation
        self.time_decay = time_decay
        self.dev_mode = dev_mode
        self.previous_context_articles = []
        if self.dev_mode:
            self.client = AsyncOpenAI(api_key="Insert your key here.")
        self.model = SentenceTransformer(str(model_file_path("all-MiniLM-L6-v2")))

    def __call__(
        self,
        clicked: CandidateSet,
        selected: CandidateSet,
        interest_profile: InterestProfile,
    ) -> RecommendationList:
        if self.dev_mode:
            selected = asyncio.run(self.generate_newsletter(clicked, selected, interest_profile))
        return selected

    async def generate_newsletter(
        self,
        clicked: CandidateSet,
        selected: CandidateSet,
        interest_profile: InterestProfile,
    ):
        topic_distribution = LocalityCalibrator.compute_topic_prefs(interest_profile)
        treatment = selected.treatment_flags
        tasks = []

        for i in range(len(selected.articles)):
            article = selected.articles[i]
            if treatment[i]:
                task = self.generated_context(article, clicked, self.time_decay, topic_distribution)
                tasks.append((article, task))

        results = await asyncio.gather(*(task[1] for task in tasks), return_exceptions=True)

        for (article, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                logger.error(f"Error generating context for article: {result}")
            else:
                article.headline, article.subhead = result

        return selected

    async def generated_context(
        self,
        article: Article,
        clicked_articles: CandidateSet,
        time_decay: bool,
        topic_distribution: dict,
    ):
        related_article = self.related_context(article, clicked_articles, time_decay)

        if related_article is not None:
            # high similarity, use the top-1 article to rewrite the rec
            main_news = {"HEADING": article.headline, "SUB_HEADING": article.subhead}
            related_news = {
                "HEADING": related_article.headline,
                "SUB_HEADING": related_article.subhead,
            }

            generated_rec = await self.semantic_narrative(main_news, related_news)

        else:
            if topic_distribution:
                generated_rec = await self.highlevel_narrative(article, topic_distribution)
            else:
                generated_rec = {
                    "HEADLINE": article.headline,
                    "SUB_HEADLINE": article.subhead,
                }

        generated_dict = json.loads(generated_rec)
        generated_headline = generated_dict.get("HEADLINE", "")
        generated_subhead = generated_dict.get("SUB_HEADLINE", "")
        if generated_headline == "" or generated_subhead == "":
            logger.warning("GPT response invald, falling back to original headline...")
            return article.headline, article.subhead
        else:
            return generated_headline, generated_subhead

    def related_context(
        self,
        article: Article,
        clicked: CandidateSet,
        time_decay: bool,
    ):
        selected_subhead = article.subhead
        selected_date = article.published_at

        clicked_articles = clicked.articles
        time0 = selected_date - timedelta(days=DAYS)

        clicked_articles = [
            article
            for article in clicked_articles
            if article.published_at >= time0 and article not in self.previous_context_articles
        ]

        candidate_indices = self.related_indices(selected_subhead, selected_date, clicked_articles, time_decay)
        if len(candidate_indices) == 0:
            return None

        self.previous_context_articles.append(clicked_articles[candidate_indices[0]])

        return clicked_articles[candidate_indices[0]]

    def related_indices(
        self,
        selected_subhead: str,
        selected_date: datetime,
        clicked_articles: list,
        time_decay: bool,
    ):
        all_subheads = [selected_subhead] + [article.subhead for article in clicked_articles]
        embeddings = self.model.encode(all_subheads)

        target_embedding = embeddings[0].reshape(1, -1)
        clicked_embeddings = embeddings[1:]
        if len(clicked_embeddings) != 0:
            similarities = cosine_similarity(target_embedding, clicked_embeddings)[0]
        else:
            return []

        # CHECK threshold [0.2, 0, 0.2]
        for i in range(len(similarities)):
            val = similarities[i]
            if val < SEMANTIC_THRESHOLD:
                similarities[i] = 0

        if np.sort(similarities)[-1] < SEMANTIC_THRESHOLD:
            return []

        elif time_decay:
            weights = [
                self.get_time_weight(selected_date, published_date)
                for published_date in [article.published_at for article in clicked_articles]
            ]
            weighted_similarities = similarities * weights

            selected_indices = np.argsort(weighted_similarities)[-1:]
            return selected_indices

        else:
            selected_indices = np.argsort(weighted_similarities)[-1:]
            return selected_indices

    async def semantic_narrative(self, main_news, related_news):
        system_prompt = (
            "You are an Associated Press editor tasked to rewrite the [[MAIN_NEWS]] HEADING and SUB_HEADING in a natural and factual tone. "
            "You are provided a [[MAIN_NEWS]] to be recommended and a [[RELATED_NEWS]] that a user read before. "
            "Rewrite the HEADLINE and SUB_HEADLING of [[MAIN_NEWS]] by implicitly connecting it to [[RELATED_NEWS]] and "
            "highlight points from [[RELATED_NEWS]] relevant to why the user should also be interested in [[MAIN_NEWS]]. "
            'Your response should only include JSON parsable by json.loads() in the format {"HEADLINE": "[REWRITTEN_HEADLINE]", "SUB_HEADLINE": "[REWRITTEN_SUBHEADLINE]"}\'. '  # "Your response should only include a rewritten healdine and subheadling always in the form '##HEADLINE##: [REWRITTEN_HEADLINE] ##SUB_HEADLINE##: [REWRITTEN_SUBHEADLINE]' "
            "[REWRITTEN_HEADLINE] should be 15 or less words and [REWRITTEN_SUBHEADLINE] should be a single sentence, "
            "no more than 30 words, and shouldn't end in punctuation. Ensure both are neutral and accurately describe [[MAIN_NEWS]]."
        )

        input_prompt = f"[[MAIN_NEWS]]: {main_news} \n[[RELATED_NEWS]]: {related_news}"

        logger.info(f"Semantic narrative: {input_prompt}")
        return await self.gpt_generate(system_prompt, input_prompt)

    async def highlevel_narrative(self, main_news, topic_distribution):
        logger.info(f"Topic distribution narrative: {topic_distribution}")
        sorted_items = sorted(topic_distribution.items(), key=lambda item: item[1], reverse=True)
        top_keys = [key for key, _ in sorted_items[:NUM_TOPICS]]

        system_prompt = (
            "You are an Associated Press editor tasked to rewrite the [[MAIN_NEWS]] HEADING and SUB_HEADING in a natural and factual tone. "
            "You are provided a [[MAIN_NEWS]] to be recommended to a user interested in [[INTERESTED_TOPICS]]."
            "Rewrite the HEADLINE and SUB_HEADLING of [[MAIN_NEWS]] by implicitly connecting it to [[INTERESTED_TOPICS]] "
            "and highlight points relevant to why the user should also be interested in [[MAIN_NEWS]]. "
            'Your response should only include JSON parsable by json.loads() in the format {"HEADLINE": "[REWRITTEN_HEADLINE]", "SUB_HEADLINE": "[REWRITTEN_SUBHEADLINE]"}\'. '  # a rewritten healdine and subheadling always in the form '##HEADLINE##: [REWRITTEN_HEADLINE] ##SUB_HEADLINE##: [REWRITTEN_SUBHEADLINE]' "
            "[REWRITTEN_HEADLINE] should be 15 or less words and [REWRITTEN_SUBHEADLINE] should be a single sentence, "
            "no more than 30 words, and shouldn't end in punctuation. Ensure both are neutral and accurately describe [[MAIN_NEWS]]."
        )

        main_news = {"HEADING": main_news.headline, "SUB_HEADING": main_news.subhead}
        input_prompt = f"[[MAIN_NEWS]]: {main_news} \n[[INTERESTED_TOPICS]]: {top_keys}"

        logger.info(f"Highlevel narrative: {input_prompt}")
        return await self.gpt_generate(system_prompt, input_prompt)

    def get_time_weight(self, published_target, published_clicked):
        time_distance = abs((published_clicked - published_target).days)
        weight = 1 / np.log(1 + time_distance) if time_distance > 0 else 1  # Avoid log(1) when x = 0
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
                chat_completion = await self.client.beta.chat.completions.parse(
                    messages=message,
                    response_format={"type": "json_object"},
                    temperature=temperature,
                    max_tokens=max_tokens,
                    frequency_penalty=frequency_penalty,
                    model="gpt-4o-mini",
                )
                logger.info(f"GPT response: {chat_completion.choices[0].message.content}")
                return chat_completion.choices[0].message.content

            except Exception as e:
                print(f"Fail to call OPENAI API: {e}")
                retries += 1
                if retries < MAX_RETRIES:
                    print(f"{retries} try to regenerate the context")
                    await asyncio.sleep(DELAY)
                else:
                    raise
