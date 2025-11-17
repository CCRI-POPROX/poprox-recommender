import torch

from poprox_concepts.domain import CandidateSet, InterestProfile
from poprox_recommender.components.scorers import ArticleScorer
from poprox_recommender.pytorch.decorators import torch_inference


class TopicalArticleScorer(ArticleScorer):
    config: None

    @torch_inference
    def __call__(self, candidate_articles: CandidateSet, interest_profile: InterestProfile) -> CandidateSet:
        candidate_embeddings = candidate_articles.embeddings
        user_embedding = interest_profile.embedding

        topic_embeddings = interest_profile.topic_embeddings
        topic_weights = interest_profile.topic_weights

        if user_embedding is not None:
            user_scores = torch.matmul(candidate_embeddings, user_embedding.t()).squeeze()

            if topic_embeddings is not None and topic_weights is not None:
                topic_scores = self.compute_topic_scores(candidate_articles, topic_embeddings, topic_weights)
            else:
                topic_scores = torch.zeros_like(user_scores)

            combined_scores = 0.5 * user_scores + 0.5 * topic_scores

            candidate_articles.scores = combined_scores.cpu().detach().numpy()
        else:
            candidate_articles.scores = None

        return candidate_articles

    def compute_topic_scores(self, candidate_articles: CandidateSet, topic_embeddings: dict, topic_weights: dict):
        scores = torch.zeros(len(candidate_articles.articles), device=candidate_articles.embeddings.device)

        topic_name_to_id = {value["topic_name"]: key for key, value in topic_weights.items()}

        for i, article in enumerate(candidate_articles.articles):
            article_topics = {
                mention.entity.name for mention in article.mentions if mention.entity.entity_type == "topic"
            }

            for topic_name in article_topics:
                if topic_name in topic_name_to_id:
                    topic_id = topic_name_to_id[topic_name]
                    topic_weight = topic_weights[topic_id]["weight"]
                    topic_embedding = topic_embeddings.get(topic_id)

                    if topic_embedding is not None:
                        scores[i] += topic_weight * torch.dot(candidate_articles.embeddings[i], topic_embedding)

        return scores
