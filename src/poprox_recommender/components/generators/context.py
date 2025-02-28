import json
import logging
from datetime import datetime, timedelta

import numpy as np
from lenskit.pipeline import Component
from openai import OpenAI
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
            logger.info("Dev_mode is true, using live OpenAI client...")
            self.client = OpenAI(api_key="Insert your key here...")
        self.model = SentenceTransformer(str(model_file_path("all-MiniLM-L6-v2")))

    def __call__(
        self,
        clicked: CandidateSet,
        selected: CandidateSet,
        interest_profile: InterestProfile,
    ) -> RecommendationList:
        if self.dev_mode:
            selected = self.generate_newsletter(clicked, selected, interest_profile)
            # selected = asyncio.run(self.generate_newsletter(clicked, selected, interest_profile))
        return selected

    def generate_newsletter(
        self,
        clicked: CandidateSet,
        selected: CandidateSet,
        interest_profile: InterestProfile,
    ):
        topic_distribution = LocalityCalibrator.compute_topic_prefs(interest_profile)
        treatment_articles = []
        treatment_map = []
        for i, (article, treatment) in enumerate(zip(selected.articles, selected.treatment_flags)):
            if treatment:
                treatment_articles.append(article)
                treatment_map.append(i)

        treated_articles = self.generate_treatment_previews(
            treatment_articles, clicked, self.time_decay, topic_distribution
        )

        for i, treated_article in enumerate(treated_articles):
            selected.articles[treatment_map[i]] = treated_article
        """
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
        """
        return selected

    def generate_treatment_previews(
        self,
        articles: list[Article],
        clicked_articles: CandidateSet,
        time_decay: bool,
        topic_distribution: dict,
    ):
        system_prompt = (
            "You are an Associated Press editor tasked to rewrite a list of news article previews in a natural "
            "and factual tone. You are provided multiple [[MAIN_NEWS]] each with a HEADLINE and SUB_HEADLINE "
            "that should be rewritten using the following rules based on [[RELATED_NEWS]] or [[INTERESTED_TOPICS]]. "
            "Your response should only include a JSON list parseable by json.loads() in the "
            '\'{"REWRITTEN_ARTICLES": [{"HEADLINE": "[REWRITTEN_HEADLINE]", "SUB_HEADLINE": "[REWRITTEN_SUBHEADLINE]"}]}\' '  # noqa: E501
            "and should include a rewritten HEADLINE and SUB_HEADLINE for each [[MAIN_NEWS]] article in the input. "
            "Rules for rewritting [[MAIN_NEWS]] with [[RELATED_NEWS]]: "
            # "1. ***The rewritten news preview should explicitly name and reference parts of the HEADLINE of the [[RELATED_NEWS]].*** "  # noqa: E501
            "1. ***Explicitly integrate key themes or implications from the [[RELATED_NEWS]] headline into the "
            "rewritten news preview, rather than just naming or referencing it. The connection should feel meaningful, "
            "not just mentioned in passing.*** "
            "2. Reframe the [[MAIN_NEWS]] headline to emphasize a natural progression, contrast, or deeper context "
            "related to the [[RELATED_NEWS]]. Highlight how the new article builds on, challenges, or expands the "
            "reader's prior understanding."
            "3. Avoid minimal rewording of the original [[MAIN_NEWS]] headline—introduce a fresh angle that makes the "
            "connection to [[RELATED_NEWS]] feel insightful and engaging."
            # "2. The rewritten news preview should highlight points that are relevant to why the user should also be "
            # "interested in [[MAIN_NEWS]] based on the fact they previously read [[RELATED_NEWS]]. "
            "Rules for rewritting [[MAIN_NEWS]] with [[INTERESTED_TOPICS]]: "
            "1. ***Explicitly integrate one or more of the user's broad [[INTERESTED_TOPICS]] into the rewritten news "
            "preview in a way that naturally reshapes the focus of the headline.*** "
            # "1. ***The rewritten news preview should explicitly name and connect the user's broad [[INTERESTED_TOPICS]] to [[MAIN_NEWS]].*** "  # noqa: E501
            "2. Reframe the original headline to emphasize an angle that directly appeals to why the user's "
            "[[INTERESTED_TOPICS]] make this news particularly relevant. Instead of merely linking the topics, adjust "
            "the framing to highlight an unexpected connection, unique insight, or compelling consequence."
            "3. Avoid simply restating the original [[MAIN_NEWS]] headline with minor adjustments; instead, introduce "
            "a fresh perspective that aligns with the user's interests while remaining true to the core facts."
            # "3. The rewritten news preview should highlight points that are relevant to why the user should also be "
            # "interested in [[MAIN_NEWS]] based on the user's top [[INTERESTED_TOPICS]]. "
            "Rules for all rewritten previews:"
            "1. All rewritten previews should have a HEADLINE and SUB_HEADLINE."
            "2. All [REWRITTEN_SUBHEADLINE]s shouldn't end in punctuation. "
            "3. All [REWRITTEN_HEADLINE]s and [REWRITTEN_SUBHEADLINE]s should be approximately the same length as the "
            "[[MAIN_NEWS]] HEADLINE and SUB_HEADLINE they are based on. "
            "4. Ensure that words and strategies used to highlight relevant points in the rewritten previews are "
            "different from one another in the resulting list. "
            "5. Ensure all rewritten articles are neutral and accurately describe the [[MAIN_NEWS]] they are based on."
        )

        input_prompt = []
        sorted_topics = sorted(topic_distribution.items(), key=lambda item: item[1], reverse=True)
        top_topics = [key for key, _ in sorted_topics[:NUM_TOPICS]]

        rewritten_article_mapping = []
        logger.info(f"Top {NUM_TOPICS} topics: {top_topics}")
        for i, article in enumerate(articles):
            related_article = self.related_context(article, clicked_articles, time_decay)
            if related_article is not None:
                # high similarity, use the top-1 article to rewrite the rec
                article_prompt = f"[[MAIN_NEWS]]\nHEADLINE: {article.headline}\nSUB_HEADLINE: {article.subhead}\n[[RELATED_NEWS]]\nHEADLINE: {related_article.headline}\nSUB_HEADLINE: {related_article.subhead}"  # noqa: E501

                logger.info(
                    f"Generating event-level narrative for '{article.headline[:15]}' from related article {related_article.headline[:15]}"  # noqa: E501
                )
                input_prompt.append(article_prompt)
                rewritten_article_mapping.append(i)
            else:
                if topic_distribution:
                    article_prompt = f"[[MAIN_NEWS]]\nHEADLINE: {article.headline}\nSUB_HEADLINE: {article.subhead}\n[[INTERESTED_TOPICS]]\n{top_topics}"  # noqa: E501

                    logger.info(f"Generating topic-level narrative for related article: {article.headline[:15]}")
                    input_prompt.append(article_prompt)
                    rewritten_article_mapping.append(i)
                else:
                    logger.warning(
                        f"No topic_distribution for generating high-level narrative for {article.headline[:15]}. Falling back to original preview..."  # noqa: E501
                    )
        try:
            rewritten_previews = self.gpt_generate(system_prompt, input_prompt, len(rewritten_article_mapping))
        except Exception as e:
            logger.error(f"Error in call to OPENAI API: {e}. Falling back to all original preview...")
            return articles

        for i, rewritten_preview in enumerate(rewritten_previews):
            articles[rewritten_article_mapping[i]].headline = rewritten_preview["HEADLINE"]
            articles[rewritten_article_mapping[i]].subhead = rewritten_preview["SUB_HEADLINE"]

        return articles

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
            selected_indices = np.argsort(similarities)[-1:]
            return selected_indices

    def get_time_weight(self, published_target, published_clicked):
        time_distance = abs((published_clicked - published_target).days)
        weight = 1 / np.log(1 + time_distance) if time_distance > 0 else 1  # Avoid log(1) when x = 0
        return weight

    def rewritten_previews_feedback(self, rewritten_previews, expected_output_n):
        if not isinstance(rewritten_previews, dict) and "REWRITTEN_ARTICLES" not in rewritten_previews:
            logger.warning("GPT response invald and doesn't contain a list of previews. Retrying...")
            feedback = (
                "Your response isn't a JSON list parseable by json.loads() in the "
                'format \'[{"HEADLINE": "[REWRITTEN_HEADLINE]", "SUB_HEADLINE": "[REWRITTEN_SUBHEADLINE]"}]\' '
                "and should include a rewritten HEADLINE and SUB_HEADLINE for each article in the list of "
                "[[MAIN_NEWS]]. Ensure your response is a valid JSON list parseable by json.loads() that "
                f"includes all {expected_output_n} rewritten articles."
            )
            return feedback
        elif len(rewritten_previews["REWRITTEN_ARTICLES"]) != expected_output_n:
            logger.warning(
                f"GPT response invald and is missing previews {len(rewritten_previews['REWRITTEN_ARTICLES'])} != {expected_output_n}. Retrying..."  # noqa: E501
            )
            feedback = (
                f"Your response JSON list of rewritten headlines doesn't include all {expected_output_n} "
                "rewritten articles. Ensure your response is a valid JSON list parseable by json.loads() that "
                f"includes all {expected_output_n} rewritten articles."
            )
            return feedback
        for item in rewritten_previews["REWRITTEN_ARTICLES"]:
            if not isinstance(item, dict) or set(item.keys()) != {"HEADLINE", "SUB_HEADLINE"}:
                logger.warning(f"GPT response invald for {item}. Retrying...")
                feedback = (
                    "Your response includes one or more articles not in the format "
                    '\'{"HEADLINE": "[REWRITTEN_HEADLINE]", "SUB_HEADLINE": "[REWRITTEN_SUBHEADLINE]"}\'. '
                    f"Ensure all {expected_output_n} rewritten articles contain both a HEADLINE and SUB_HEADLINE "
                    "and are included in a JSON list parseable by json.loads() in the "
                    'format \'[{"HEADLINE": "[REWRITTEN_HEADLINE]", "SUB_HEADLINE": "[REWRITTEN_SUBHEADLINE]"}]\' '
                )
                return feedback

        return False

    def gpt_generate(self, system_prompt, content_prompt, expected_output_n):
        message = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": main_news} for main_news in content_prompt]},
        ]
        temperature = 0.2
        max_tokens = 2000
        frequency_penalty = 0.0
        chat_completion = self.client.beta.chat.completions.parse(
            messages=message,
            response_format={"type": "json_object"},
            temperature=temperature,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
            model="gpt-4o",
        )
        logger.info(f"GPT response: {chat_completion.choices[0].message.content}")

        rewritten_previews = json.loads(chat_completion.choices[0].message.content)
        feedback = self.rewritten_previews_feedback(rewritten_previews, expected_output_n)
        if feedback:
            logger.warning(f"GPT response invalid. Retrying with feedback '{feedback}'")
            reprompt_message = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "text", "text": main_news} for main_news in content_prompt]},
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": chat_completion.choices[0].message.content}],
                },
                {"role": "user", "content": [{"type": "text", "text": feedback}]},
            ]

            chat_completion = self.client.beta.chat.completions.parse(
                messages=reprompt_message,
                response_format={"type": "json_object"},
                temperature=temperature,
                max_tokens=max_tokens,
                frequency_penalty=frequency_penalty,
                model="gpt-4o",
            )

            logger.info(f"GPT reprompt response: {chat_completion.choices[0].message.content}")
            rewritten_previews = json.loads(chat_completion.choices[0].message.content)

            feedback = self.rewritten_previews_feedback(rewritten_previews, expected_output_n)
            if feedback:
                raise ValueError(f"GPT response still invalid. Failing from feedback '{feedback}'")

        return rewritten_previews["REWRITTEN_ARTICLES"]
