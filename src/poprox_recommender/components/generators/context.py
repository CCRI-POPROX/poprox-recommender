import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Counter

import numpy as np
from lenskit.pipeline import Component
from openai import AsyncOpenAI
from pydantic import BaseModel
from rouge_score import rouge_scorer
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
BASELINE_THETA_TOPIC = 0.3
NUM_TOPICS = 3
DAYS = 14

logger = logging.getLogger(__name__)

TOPIC_DESCRIPTIONS = {
    "U.S. news": "News and events within the United States, \
covering a wide range of topics including politics, \
economy, social issues, cultural developments, crime, \
education, healthcare, and other matters of national significance.",
    "World news": "News and events from across the globe, \
focusing on international developments, global politics, \
conflicts, diplomacy, trade, cultural exchanges, and \
issues that affect multiple nations or regions.",
    "Politics": "The activities and functions of a governing body, \
the administration of its internal and external affairs, \
and the political issues that governments confront. \
Includes governance of political entities at all levels \
(country, state, city, etc.), and all government branches \
(executive, judicial, legislative, military, law enforcement). \
Also includes international governing bodies such as the UN.",
    "Business": "All commercial, industrial, financial and \
economic activities involving individuals, corporations, \
financial markets, governments and other organizations \
across all countries and regions.",
    "Entertainment": "All forms of visual and performing arts, \
design arts, books and literature, film and television, \
music, and popular entertainment. Refers primarily to \
the art and entertainment itself and to those who create, \
perform, or interpret it. For business contexts, \
see 'Media and entertainment industry'.",
    "Sports": "Organized competitive activities, usually physical in \
nature, and the systems and practices that support them. \
Includes all team and individual sports at all levels. \
Also includes sports media, business, equipment, issues, and controversies.",
    "Health": "Condition, care, and treatment of the mind and body. \
Includes diseases, illnesses, injuries, medicine, \
medical procedures, preventive care, health services, \
and public health issues.",
    "Science": "The ongoing discovery and increase of human knowledge \
through systematic and disciplined experimentation, \
and the body of knowledge thus obtained. Includes all branches \
of natural and social sciences, scientific issues \
and controversies, space exploration, and similar topics. \
May include some aspects of 'applied science', \
but for content about inventions, computers, engineering, etc., \
Technology is often a more appropriate category.",
    "Technology": "Tools, machines, systems or techniques, \
especially those derived from scientific knowledge and \
often electronic or digital in nature, for implementation \
in industry and/or everyday human activities. \
Includes all types of technological innovations and \
products, such as computers, communication and \
entertainment devices, software, industrial advancements, \
and the issues and controversies that technology gives rise to.",
    "Lifestyle": "The way a person lives, including interests, \
attitudes, personal and domestic style, values, relationships, \
hobbies, recreation, travel, personal care and grooming, \
and day-to-day activities.",
    "Religion": "All topics related to religion and its place in society, \
particularly socially and politically controversial topics. \
See terms for individual belief systems for their \
activities at all levels of organization.",
    "Climate and environment": "The natural or physical world,\
and especially the relationship between nature \
(ecosystems, wildlife, the atmosphere, water, land, etc.) \
and human beings. Includes the effects of human activities \
on the environment and vice versa, as well as the \
management of nature by humans. May also include \
discussions of the natural world that are unrelated \
to humans or human activity.",
    "Education": "The processes of teaching and learning in \
an institutional setting, including all topics related \
to the establishment and management of educational institutions.",
    "Oddities": "Unusual, quirky, or strange stories that \
capture attention due to their uniqueness, humor, or \
unexpected nature. Often includes tales of rare occurrences, \
peculiar behaviors, or bizarre phenomena.",
}


event_system_prompt = (
    "You are an Associated Press editor tasked to rewrite a news preview in a factual tone. "
    "You are provided with [[MAIN_NEWS]] with a BODY and past [[RELATED_NEWS]] with a HEADLINE, SUB_HEADLINE, "
    "and BODY. The user may not be interested in [[MAIN_NEWS]]. The user previously read the past [[RELATED_NEWS]]. "
    "Create a [[MAIN_NEWS]] HEADLINE and SUB_HEADLINE that appeals to their interests using the following rules.\n"
    "Rules:\n"
    "1. ***Explicitly*** integrate the [[RELATED_NEWS]] HEADLINE, SUB_HEADLINE, "
    "and BODY into the [[MAIN_NEWS]] HEADLINE and SUB_HEADLINE. The connection should be meaningful, "
    "not just mentioned in passing.\n"
    "2. Reframe an element of the [[MAIN_NEWS]] BODY in the [[MAIN_NEWS]] HEADLINE and SUB_HEADLINE to emphasize a "
    "natural progression, contrast, or deeper context between the [[MAIN_NEWS]] BODY the [[RELATED_NEWS]] BODY. "
    "Highlight how the [[MAIN_NEWS]] BODY builds on, challenges, or expands reader's prior understanding.\n"
    "3. The [[MAIN_NEWS]] SUB_HEADLINE should NOT end in punctuation.\n"
    "4. ***Only proper nouns should be capitalized in the [[MAIN_NEWS]] HEADLINE.***\n"
    "5. The [[MAIN_NEWS]] HEADLINE should be approximately {} words long.\n"
    "6. The [[MAIN_NEWS]] SUB_HEADLINE should be approximately {} words long.\n"
    "7. The [[MAIN_NEWS]] HEADLINE and SUB_HEADLINE should be neutral and accurately describe the [[MAIN_NEWS]] BODY.\n"
    "8. Your response should only include JSON parseable by json.loads() in the form "
    '\'{{"HEADLINE": "[REWRITTEN_HEADLINE]", "SUB_HEADLINE": "[REWRITTEN_SUBHEADLINE]"}}\'.'
)

topic_system_prompt = (
    "You are an Associated Press editor tasked to rewrite a news preview in a factual tone. "
    "You are provided with a list of a user's broad [[INTERESTED_TOPICS]] with a topics and an AP defintions "
    "and [[MAIN_NEWS]] with a BODY. The user may not be interested in [[MAIN_NEWS]]. "
    "Create a [[MAIN_NEWS]] HEADLINE and SUB_HEADLINE that appeals to their interests using the following rules.\n"
    "Rules:\n"
    "1. ***Implicitly*** integrate one or more of the user's prior reading habbits from [[INTERESTED_TOPICS]] "
    "into the [[MAIN_NEWS]] HEADLINE and SUB_HEADLINE in a way that naturally reshapes the focus of the news preview "
    "to the user's interests.\n"
    "2. Reframe an element of the [[MAIN_NEWS]] BODY in the [[MAIN_NEWS]] HEADLINE and SUB_HEADLINE to emphasize an "
    "angle or passage that directly appeals to the user's [[INTERESTED_TOPICS]] to make this news particularly "
    "relevant. Highlight an unexpected connection or unique insight between [[MAIN_NEWS]] BODY and "
    "the user's broad [[INTERESTED_TOPICS]].\n"
    "3. The [[MAIN_NEWS]] SUB_HEADLINE should NOT end in punctuation.\n"
    "4. ***Only proper nouns should be capitalized in the [[MAIN_NEWS]] HEADLINE.***\n"
    "5. The [[MAIN_NEWS]] HEADLINE should be approximately {} words long.\n"
    "6. The [[MAIN_NEWS]] SUB_HEADLINE should be approximately {} words long.\n"
    "7. The [[MAIN_NEWS]] HEADLINE and SUB_HEADLINE should be neutral and accurately describe the [[MAIN_NEWS]] BODY.\n"
    "8. Your response should only include JSON parseable by json.loads() in the form "
    '\'{{"HEADLINE": "[REWRITTEN_HEADLINE]", "SUB_HEADLINE": "[REWRITTEN_SUBHEADLINE]"}}\'.'
)

refine_system_prompt = (
    "You are an Associated Press editor tasked with refining rewritten news previews. "
    "Each preview consists of a HEADLINE and a SUB_HEADLINE. Your goal is to ensure that HEADLINEs and SUB_HEADLINEs "
    "in different previews do not rely on the same words or strategies to emphasize key topics or related news. "
    "If a HEADLINE and SUB_HEADLINE in a preview are too similar to others in how they highlight key points, rewrite "
    "them to introduce variation while preserving their meaning. However, if a preview already uses a distinct "
    "approach from others, do not modify it and return it unchanged. Changes should not alter the meaning of the "
    "HEADLINE or SUB_HEADLINE. Your response must include the same number of rewritten news previews as the input, "
    "keeping unmodified previews intact. The rewritten SUB_HEADLINE should NOT end in punctuation. Only proper nouns "
    "should be capitalized in the [[MAIN_NEWS]] HEADLINE. "
    "Return only JSON parseable by json.loads(), in the format: "
    '\'{"REWRITTEN_NEWS_PREVIEWS": [{"HEADLINE": "[REWRITTEN_HEADLINE]", "SUB_HEADLINE": "[REWRITTEN_SUBHEADLINE]"}]}\''
)


class ContextGeneratorConfig(BaseModel):
    similarity_threshold: float = 0.4


class ContextGenerator(Component):
    config: ContextGeneratorConfig

    def __init__(self, time_decay=True, is_gpt_live=True):
        self.time_decay = time_decay
        self.is_gpt_live = is_gpt_live

        if self.is_gpt_live:
            logger.info("is_gpt_live is true, using live OpenAI client...")
            self.client = AsyncOpenAI(api_key="<<Your key>>")
            logger.info("Successfully instantiated OpenAI client...")
        self.model = SentenceTransformer(str(model_file_path("all-MiniLM-L6-v2")))

    def __call__(
        self,
        clicked: CandidateSet,
        selected: CandidateSet,
        interest_profile: InterestProfile,
        similarity_threshold: float | None,
    ) -> RecommendationList:
        similarity_threshold = 0.4 if similarity_threshold is None else similarity_threshold

        logger.error(f"clicked embeddings {len(clicked.embeddings)}")
        logger.error(f"selected embeddings {len(selected.embeddings)}")
        # logger.error(f"Selected object fields: {', '.join(vars(selected).keys())}")
        extras = []
        # selected = self.generate_newsletter(clicked, selected, interest_profile)
        selected, extras = asyncio.run(
            self.generate_newsletter(clicked, selected, interest_profile, similarity_threshold)
        )
        logger.error(f"Final extras: {extras}")
        return RecommendationList(articles=selected.articles, extras=extras)

    async def generate_newsletter(
        self,
        clicked: CandidateSet,
        selected: CandidateSet,
        interest_profile: InterestProfile,
        similarity_threshold: float,
    ):
        topic_distribution = LocalityCalibrator.compute_topic_prefs(interest_profile)
        top_topics = []
        if topic_distribution:
            topic_distribution.pop("U.S. news", None)
            topic_distribution.pop("World news", None)
            sorted_topics = sorted(topic_distribution.items(), key=lambda item: item[1], reverse=True)
            top_topics = [(key, count) for key, count in sorted_topics[:NUM_TOPICS]]

        treated_articles = []
        treated_extras = []
        tasks = []
        extras = [{} for _ in range(len(selected.articles))]
        related_articles = []
        previously_used_article_ids = []

        for i in range(len(selected.embeddings)):
            article_embedding = selected.embeddings[i]
            article = selected.articles[i]
            if selected.treatment_flags[i]:
                related_article = self.related_context(
                    article,
                    article_embedding,
                    clicked,
                    self.time_decay,
                    similarity_threshold,
                    previously_used_article_ids,
                    extras[i],
                )
                related_articles.append(related_article)
                if related_article is not None:
                    previously_used_article_ids.append(related_article.article_id)
            else:
                related_articles.append(None)

        for i in range(len(selected.articles)):
            article = selected.articles[i]
            if selected.treatment_flags[i]:
                extras[i]["original_headline"] = article.headline
                extras[i]["original_subhead"] = article.subhead

                task = self.generate_treatment_preview(
                    article,
                    top_topics,
                    related_articles[i],
                    extras[i],
                    # article, clicked, self.time_decay, top_topics,  extras[i]
                )
                tasks.append((article, extras[i], task))

        results = await asyncio.gather(*(task[2] for task in tasks), return_exceptions=True)

        for (article, extra, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                logger.error(f"Error generating context for article: {result}")
            else:
                article.headline, article.subhead = result  # type: ignore
                treated_articles.append(article)
                treated_extras.append(extra)

        if treated_articles:
            treated_articles, treated_extras = await self.diversify_treatment_previews(treated_articles, treated_extras)

        for article, extra in zip(treated_articles, treated_extras):
            # Subtract rougel score of rewritten headline + subhead and body from baseline
            rouge1, rouge2, rougeL = self.offline_metric_calculation(
                extra["original_headline"], extra["original_subhead"], article.headline, article.subhead
            )
            extra["rouge1"] = rouge1
            extra["rouge2"] = rouge2
            extra["rougeL"] = rougeL

        return selected, extras

    async def generate_treatment_preview(
        self,
        article: Article,
        # clicked_articles: CandidateSet,
        # time_decay: bool,
        top_topics: list,
        related_article: Article,
        extra_logging: dict,
    ):
        # related_article = self.related_context(article, clicked_articles, time_decay, extra_logging)
        headline_length = len(article.headline.split())
        subhead_length = len(article.subhead.split())

        if related_article is not None:
            # high similarity, use the top-1 article to rewrite the rec
            article_prompt = f"""
[[MAIN_NEWS]]
    BODY_TEXT: {article.body}
[[RELATED_NEWS]]
    HEADLINE: {related_article.headline}
    SUB_HEADLINE: {related_article.subhead}
    BODY_TEXT: {related_article.body}
"""  # noqa: E501
            # HEADLINE: {article.headline}
            # SUB_HEADLINE: {article.subhead}

            logger.info(
                f"Generating event-level narrative for '{article.headline[:30]}' from related article '{related_article.headline[:30]}'"  # noqa: E501
            )
            # logger.info(
            #     f"Using prompt: {topic_system_prompt.format(headline_length, subhead_length)}\n\n{article_prompt}"
            # )
            extra_logging["prompt_level"] = "event"
            if self.is_gpt_live:
                rec_headline, rec_subheadline = await self.async_gpt_generate(
                    event_system_prompt.format(headline_length, subhead_length),
                    article_prompt,
                    article.headline,
                    article.subhead,
                )
            else:
                raise Exception("Not generating GPT previews in dev mode")
        else:
            if top_topics:
                article_prompt = f"""
[[INTERESTED_TOPICS]]: { {f"{top_count_pair[0]}: {TOPIC_DESCRIPTIONS[top_count_pair[0]]}" for top_count_pair in top_topics} }
[[MAIN_NEWS]]
    BODY_TEXT: {article.body}
"""  # noqa: E501
                # [[INTERESTED_TOPICS]]: {[top_count_pair[0] for top_count_pair in top_topics]}
                # HEADLINE: {article.headline}
                # SUB_HEADLINE: {article.subhead}

                logger.info(f"Generating topic-level narrative for related article: {article.headline[:30]}")
                # logger.info(
                #     f"Using prompt: {topic_system_prompt.format(headline_length, subhead_length)}\n\n{article_prompt}"
                # )
                extra_logging["prompt_level"] = "topic"
                for ind, top_count_pair in enumerate(top_topics):
                    extra_logging["top_{}_topic".format(ind)] = top_count_pair[0]
                    extra_logging["top_{}_topic_ratio".format(ind)] = float(top_count_pair[1])
                if self.is_gpt_live:
                    rec_headline, rec_subheadline = await self.async_gpt_generate(
                        topic_system_prompt.format(headline_length, subhead_length),
                        article_prompt,
                        article.headline,
                        article.subhead,
                    )
                else:
                    raise Exception("Not generating GPT previews in dev mode")
            else:
                logger.warning(
                    f"No topic_distribution for generating high-level narrative for {article.headline[:30]}. Falling back to original preview..."  # noqa: E501
                )
                extra_logging["prompt_level"] = "none"
                rec_headline, rec_subheadline = article.headline, article.subhead
        return rec_headline, rec_subheadline

    async def diversify_treatment_previews(
        self,
        articles: list[Article],
        extras: list[dict],
    ):
        input_prompt = []
        for i, article in enumerate(articles):
            article_prompt = f"""
                HEADLINE: {article.headline}
                SUB_HEADLINE: {article.subhead}
            """  # noqa: E501
            input_prompt.append(article_prompt)

        try:
            if self.is_gpt_live:
                rewritten_previews = await self.async_gpt_diversify(
                    refine_system_prompt, input_prompt, len(input_prompt)
                )
            else:
                raise Exception("Not generating GPT previews in dev mode")
        except Exception as e:
            logger.error(f"Error in call to OPENAI API: {e}. Falling back to all original preview...")
            return articles

        for i, rewritten_preview in enumerate(rewritten_previews):
            articles[i].headline = rewritten_preview["HEADLINE"]
            articles[i].subhead = rewritten_preview["SUB_HEADLINE"]

            extras[i]["rewritten_headline"] = rewritten_preview["HEADLINE"]
            extras[i]["rewritten_subheadline"] = rewritten_preview["SUB_HEADLINE"]

        return articles, extras

    def offline_metric_calculation(self, headline, subhead, gen_headline, gen_subhead):
        logger.info(f"Calculating metrics for Article: '{headline[:30]}'")
        # RougeL
        r_scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        article_preview = f"{headline} {subhead}"
        gen_article_preview = f"{gen_headline} {gen_subhead}"
        score = r_scorer.score(article_preview, gen_article_preview)  # ["rougeL"]
        rouge1 = score["rouge1"].fmeasure
        rouge2 = score["rouge2"].fmeasure
        rougeL = score["rougeL"].fmeasure
        logger.info(f"    Rouge1 {rouge1}, Rouge2: {rouge2}, RougeL: {rougeL}")
        # TODO add nli_model metric and return

        return rouge1, rouge2, rougeL

    def related_context(
        self,
        article: Article,
        article_embedding: list,
        clicked_set: CandidateSet,
        time_decay: bool,
        similarity_threshold: float,
        related_articles: list,
        extra_logging: dict,
    ) -> Article:
        # selected_subhead = article.subhead
        selected_date = article.published_at

        clicked_articles = clicked_set.articles
        time0 = selected_date - timedelta(days=DAYS)

        logger.info(f"Previously used article ids: {related_articles}")
        clicked_articles = [
            article
            for article in clicked_articles
            if article.published_at >= time0 and article.article_id not in related_articles
        ]
        filtered_clicked_embeddings = [
            embedding
            for article, embedding in zip(clicked_articles, clicked_set.embeddings)
            if article.published_at >= time0 and article.article_id not in related_articles
        ]
        candidate_indices = self.related_indices(
            # selected_subhead, selected_date, clicked_articles, time_decay, extra_logging
            article_embedding,
            selected_date,
            clicked_articles,
            filtered_clicked_embeddings,
            time_decay,
            similarity_threshold,
            extra_logging,
        )
        if len(candidate_indices) == 0:
            return None

        return clicked_articles[candidate_indices[0]]

    def related_indices(
        self,
        # selected_subhead: str,
        selected_article_embedding,
        selected_date: datetime,
        clicked_articles: list,
        clicked_article_embeddings,
        time_decay: bool,
        similarity_threshold: float,
        extra_logging: dict,
    ):
        # all_subheads = [selected_subhead] + [article.subhead for article in clicked_articles]
        # embeddings = self.model.encode(all_subheads)

        target_embedding = selected_article_embedding.reshape(1, -1)
        # clicked_embeddings = embeddings[1:]
        if len(clicked_article_embeddings) != 0:
            similarities = cosine_similarity(target_embedding, clicked_article_embeddings)[0]
        else:
            return []

        logger.info(f"Similarity scores: {similarities}")
        # CHECK threshold [0.2, 0, 0.2]
        for i in range(len(similarities)):
            val = similarities[i]
            if val < similarity_threshold:
                similarities[i] = 0

        most_sim_article_ind = np.argmax(similarities)
        highest_sim = float(similarities[most_sim_article_ind])
        if highest_sim < similarity_threshold:
            extra_logging["similarity"] = float(similarities[most_sim_article_ind])
            extra_logging["context_article"] = str(clicked_articles[most_sim_article_ind].article_id)
            return []

        elif time_decay:
            weights = [
                self.get_time_weight(selected_date, published_date)
                for published_date in [article.published_at for article in clicked_articles]
            ]
            weighted_similarities = similarities * weights

            selected_indices = np.argsort(weighted_similarities)[-1:]
            extra_logging["similarity"] = float(similarities[selected_indices[0]])
            extra_logging["context_article"] = str(clicked_articles[selected_indices[0]].article_id)
            return selected_indices

        else:
            selected_indices = np.argsort(similarities)[-1:]
            extra_logging["similarity"] = float(similarities[selected_indices[0]])
            extra_logging["context_article"] = str(clicked_articles[selected_indices[0]].article_id)
            return selected_indices

    def get_time_weight(self, published_target, published_clicked):
        time_distance = abs((published_clicked - published_target).days)
        weight = 1 / np.log(1 + time_distance) if time_distance > 0 else 1  # Avoid log(1) when x = 0
        return weight

    def rewritten_previews_feedback(self, rewritten_previews, expected_output_n, fallback_on_fail=False):
        if not isinstance(rewritten_previews, dict) and "REWRITTEN_NEWS_PREVIEWS" not in rewritten_previews:
            logger.warning("GPT response invald and doesn't contain a list of previews. Retrying...")
            feedback = (
                "Your response isn't JSON parseable by json.loads() in the format "
                '\'{"REWRITTEN_NEWS_PREVIEWS": [{"HEADLINE": "[REWRITTEN_HEADLINE]", "SUB_HEADLINE": "[REWRITTEN_SUBHEADLINE]"}]}\'. '  # noqa: E501
                "It should include a rewritten HEADLINE and SUB_HEADLINE for each article in the list of "
                "REWRITTEN_NEWS_PREVIEWS. Ensure your response is valid JSON parseable by json.loads() that "
                f"includes all {expected_output_n} rewritten articles."
            )
            return feedback
        elif len(rewritten_previews["REWRITTEN_NEWS_PREVIEWS"]) != expected_output_n:
            logger.warning(
                f"GPT response invald and is missing previews {len(rewritten_previews['REWRITTEN_NEWS_PREVIEWS'])} != {expected_output_n}. Retrying..."  # noqa: E501
            )
            feedback = (
                f"Your response JSON of 'REWRITTEN_NEWS_PREVIEWS' doesn't include all {expected_output_n} "
                "rewritten articles. Ensure your response is valid JSON parseable by json.loads() that "
                f"includes all {expected_output_n} rewritten articles."
            )
            return feedback

        topic_usage_counter = Counter()
        feedback = ""
        capilization_feedback = False
        dict_feedback = False
        for item in rewritten_previews["REWRITTEN_NEWS_PREVIEWS"]:
            if not dict_feedback and not isinstance(item, dict) or set(item.keys()) != {"HEADLINE", "SUB_HEADLINE"}:
                logger.warning(f"GPT response invald for {item}. Retrying...")
                feedback_add = (
                    "Your response includes one or more articles not in the format "
                    '\'{"HEADLINE": "[REWRITTEN_HEADLINE]", "SUB_HEADLINE": "[REWRITTEN_SUBHEADLINE]"}\'. '
                    f"Ensure all {expected_output_n} rewritten articles contain both a HEADLINE and SUB_HEADLINE "
                    "and are included in JSON parseable by json.loads() in the format "
                    '\'{"REWRITTEN_NEWS_PREVIEWS": [{"HEADLINE": "[REWRITTEN_HEADLINE]", "SUB_HEADLINE": "[REWRITTEN_SUBHEADLINE]"}]}\'. '  # noqa: E501
                )
                feedback = feedback + "\n" + feedback_add
                dict_feedback = True

            headline_words = item["HEADLINE"].split()
            capitalized_words = sum(1 for word in headline_words if word[0].isupper())

            # only give feedback on capitalization if this is the first time getting feedback
            if not fallback_on_fail and not capilization_feedback and capitalized_words > len(headline_words) / 2:
                feedback_add = (
                    "Your response includes many capitalized letters. "
                    "Ensure only proper nouns, acronymns, and appropriate words are capitalized in the HEADLINE."
                )
                feedback = feedback + "\n" + feedback_add
                capilization_feedback = True

            full_text = item["HEADLINE"] + " " + item["SUB_HEADLINE"]
            for topic in TOPIC_DESCRIPTIONS.keys():
                topic_count = full_text.lower().count(topic.lower())
                topic_usage_counter[topic] += topic_count

        overused_topics = [topic for topic, count in topic_usage_counter.items() if count > 2]
        # only give feedback on overused topics if this is the first time getting feedback
        if not fallback_on_fail and overused_topics:
            feedback_add = (
                "Your response overuses certain topic names. "
                f"The following words appear more than twice: {', '.join(overused_topics)}. "
                "Ensure each topic is mentioned explicitly in no more than twice across all news previews."
            )
            feedback = feedback + "\n" + feedback_add

        if feedback.strip():
            return feedback
        else:
            return False

    def rewritten_preview_feedback(self, rewritten_preview, fallback_on_fail=False):
        if not isinstance(rewritten_preview, dict) and (
            "HEADLINE" not in rewritten_preview or "SUB_HEADLINE" not in rewritten_preview
        ):
            logger.warning(f"GPT response invald for {rewritten_preview}. Retrying...")
            feedback = (
                "Your response is not in the format "
                '\'{"HEADLINE": "[REWRITTEN_HEADLINE]", "SUB_HEADLINE": "[REWRITTEN_SUBHEADLINE]"}\'. '
                "Ensure the rewritten article contains both a HEADLINE and SUB_HEADLINE "
                "and is valid JSON parseable by json.loads()."
            )
            return feedback

        headline_words = rewritten_preview["HEADLINE"].split()
        capitalized_words = sum(1 for word in headline_words if word[0].isupper())

        if not fallback_on_fail and capitalized_words > len(headline_words) / 2:
            feedback = "Ensure only proper nouns, acronymns, and appropriate words are capitalized in the HEADLINE."
            return feedback

        return False

    async def async_gpt_generate(self, system_prompt, content_prompt, original_headline, original_subhead):
        message = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": content_prompt}]},
        ]

        temperature = 0.2
        max_tokens = 2000
        frequency_penalty = 0.0
        chat_completion = await self.client.beta.chat.completions.parse(
            messages=message,
            response_format={"type": "json_object"},
            temperature=temperature,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
            model="gpt-4o-mini",
        )
        logger.info(
            f"Original headline: {original_headline}\nOriginal subhead: {original_subhead}\n"
            f"GPT response: {chat_completion.choices[0].message.content}"
        )

        rewritten_preview = json.loads(chat_completion.choices[0].message.content)
        feedback = self.rewritten_preview_feedback(rewritten_preview)
        if feedback:
            logger.warning(f"GPT response invalid. Retrying with feedback '{feedback}'")
            reprompt_message = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "text", "text": content_prompt}]},
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": chat_completion.choices[0].message.content}],
                },
                {"role": "user", "content": [{"type": "text", "text": feedback}]},
            ]

            chat_completion = await self.client.beta.chat.completions.parse(
                messages=reprompt_message,
                response_format={"type": "json_object"},
                temperature=temperature,
                max_tokens=max_tokens,
                frequency_penalty=frequency_penalty,
                model="gpt-4o-mini",
            )

            logger.info(f"GPT reprompt response: {chat_completion.choices[0].message.content}")
            rewritten_preview = json.loads(chat_completion.choices[0].message.content)

            feedback = self.rewritten_preview_feedback(rewritten_preview, fallback_on_fail=True)
            if feedback:
                raise ValueError(f"GPT response still invalid. Failing from feedback '{feedback}'")

        return (
            rewritten_preview["HEADLINE"],
            rewritten_preview["SUB_HEADLINE"],
        )

    async def async_gpt_diversify(self, system_prompt, content_prompt, expected_output_n):
        message = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [{"type": "text", "text": rewritten_news} for rewritten_news in content_prompt],
            },
        ]
        temperature = 0.2
        max_tokens = 2000
        frequency_penalty = 0.0
        chat_completion = await self.client.beta.chat.completions.parse(
            messages=message,
            response_format={"type": "json_object"},
            temperature=temperature,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
            model="gpt-4o-mini",
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

            chat_completion = await self.client.beta.chat.completions.parse(
                messages=reprompt_message,
                response_format={"type": "json_object"},
                temperature=temperature,
                max_tokens=max_tokens,
                frequency_penalty=frequency_penalty,
                model="gpt-4o-mini",
            )

            logger.info(f"GPT reprompt response: {chat_completion.choices[0].message.content}")
            rewritten_previews = json.loads(chat_completion.choices[0].message.content)

            feedback = self.rewritten_previews_feedback(rewritten_previews, expected_output_n, fallback_on_fail=True)
            if feedback:
                raise ValueError(f"GPT response still invalid. Failing from feedback '{feedback}'")

        return rewritten_previews["REWRITTEN_NEWS_PREVIEWS"]
