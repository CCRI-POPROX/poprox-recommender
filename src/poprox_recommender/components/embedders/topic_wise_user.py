from datetime import datetime, timezone
from uuid import uuid4

import torch as th

from poprox_concepts import Article, ArticleSet, Click, InterestProfile
from poprox_recommender.components.embedders import NRMSArticleEmbedder, NRMSUserEmbedder
from poprox_recommender.paths import model_file_path
from poprox_recommender.pytorch.decorators import torch_inference

topic_descriptions = {
    "U.S. news": "News and events within the United States, \
        covering a wide range of topics including politics, \
        economy, social issues, cultural developments, crime, \
        education, healthcare, and other matters of national significance.",
    "World news": "News and events from across the globe, \
        focusing on international developments, global politics, \
        conflicts, diplomacy, trade, cultural exchanges, and \
        issues that affect multiple nations or regions.",
    "Politics": "Regulation of the use, sale and ownership of \
        guns. Includes legislative action, political debate, \
        and general discussion of the role of government in regulating guns.",
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

topic_articles = [
    Article(
        article_id=uuid4(),
        headline=description,
        subhead=None,
        url=None,
        preview_image_id=None,
        published_at=datetime.now(timezone.utc),  # Set current time for simplicity
        mentions=[],
        source="topic",
        external_id=topic,
        raw_data={},
    )
    for topic, description in topic_descriptions.items()
]


def virtual_clicks(onboarding_topics, topic_embeddings):
    virtual_clicks = []
    for interest in onboarding_topics:
        topic_name = interest.entity_name
        preference = interest.preference or 0

        if topic_name in topic_embeddings:
            entity_id = interest.entity_id

            virtual_clicks.extend([Click(article_id=entity_id)] * preference)
    return virtual_clicks


class TopicUserEmbedder(NRMSUserEmbedder):
    @torch_inference
    def __call__(self, clicked_articles: ArticleSet, interest_profile: InterestProfile) -> InterestProfile:
        if len(clicked_articles.articles) == 0:
            interest_profile.embedding = None
        else:
            article_embedder = NRMSArticleEmbedder(
                model_path=model_file_path("nrms-mind/news_encoder.safetensors"), device=self.device
            )
            topic_article_set = article_embedder(ArticleSet(articles=topic_articles))
            topic_embeddings_by_name = {
                article.external_id: embedding
                for article, embedding in zip(topic_articles, topic_article_set.embeddings)
            }
            topic_embeddings_by_uuid = {
                article.article_id: embedding
                for article, embedding in zip(topic_articles, topic_article_set.embeddings)
            }

            topic_clicks = virtual_clicks(interest_profile.onboarding_topics, topic_embeddings_by_name)
            combined_click_history = interest_profile.click_history + topic_clicks

            embedding_lookup = {}
            for article, article_vector in zip(clicked_articles.articles, clicked_articles.embeddings, strict=True):
                if article.article_id not in embedding_lookup:
                    embedding_lookup[article.article_id] = article_vector

            embedding_lookup.update({topic_uuid: emb for topic_uuid, emb in topic_embeddings_by_uuid.items()})
            embedding_lookup["PADDED_NEWS"] = th.zeros(list(embedding_lookup.values())[0].size(), device=self.device)

            interest_profile.click_history = combined_click_history
            interest_profile.embedding = self.build_user_embedding(combined_click_history, embedding_lookup)

        return interest_profile
