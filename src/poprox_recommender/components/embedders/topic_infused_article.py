# pyright: basic
import logging
from dataclasses import dataclass

import torch as th
import torch.nn.functional as F

from poprox_concepts import CandidateSet
from poprox_recommender.components.embedders import NRMSArticleEmbedder, NRMSArticleEmbedderConfig
from poprox_recommender.paths import model_file_path
from poprox_recommender.pytorch.datachecks import assert_tensor_size
from poprox_recommender.pytorch.decorators import torch_inference

TITLE_LENGTH_LIMIT = 30
MAX_ATTRIBUTES = 15

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


def find_topical_text(topic_descriptions, article):
    unique_mention = {mention.entity.name for mention in article.mentions}
    topical_texts = []
    for mention in unique_mention:
        if mention in topic_descriptions:
            topical_texts.append(topic_descriptions[mention])
    return topical_texts


@dataclass
class FMStyleArticleEmbedder(NRMSArticleEmbedder):
    config: NRMSArticleEmbedderConfig

    def __init__(self, config: NRMSArticleEmbedderConfig | None = None, **kwargs):
        super().__init__(config, **kwargs)

    @torch_inference
    def __call__(self, article_set: CandidateSet) -> CandidateSet:
        if not article_set.articles:
            article_set.embeddings = th.zeros((0, self.news_encoder.embedding_size))  # type: ignore
            return article_set

        all_article_embeddings = []

        for article in article_set.articles:
            attribute_text = [article.headline]
            topical_text = find_topical_text(TOPIC_DESCRIPTIONS, article)
            num_topic = len(topical_text)
            attribute_text.extend(topical_text)

            while len(attribute_text) < MAX_ATTRIBUTES:
                attribute_text.append("")

            attribute_tensors = th.stack(
                [
                    th.tensor(
                        self.tokenizer.encode(
                            text, padding="max_length", max_length=TITLE_LENGTH_LIMIT, truncation=True
                        ),
                        dtype=th.int32,
                    ).to(self.config.device)
                    for text in attribute_text
                ]
            )

            attribute_embeddings = self.news_encoder(attribute_tensors)

            attribute_embeddings = F.normalize(attribute_embeddings, dim=1)

            attribute_embeddings[1 : num_topic + 1] = attribute_embeddings[1 : num_topic + 1] / num_topic

            all_article_embeddings.append(attribute_embeddings)

        embed_tensor = th.stack(all_article_embeddings)

        article_set.embeddings = embed_tensor  # type: ignore

        return article_set
