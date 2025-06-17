import torch
from lenskit.pipeline import Component

from poprox_concepts import CandidateSet, InterestProfile
from poprox_recommender.pytorch.decorators import torch_inference


class FMScorer(Component):
    config: None

    @torch_inference
    def __call__(self, candidate_articles: CandidateSet, interest_profile: InterestProfile) -> CandidateSet:
        # shape (no of cand, embedding(768))->(no of cand, no of article atributes, embedding(768))
        # shape (1, embedding)->(1, no of user atributes, embedding)
        user_embeddings = interest_profile.embedding

        candidate_copy = candidate_articles.model_copy()

        if user_embeddings is not None:
            scores = []
            for article_embeddings in candidate_copy.embeddings:
                # shape (no of article atributes, embedding(768))
                # shape (no of user atributes, embedding)
                # one candidate & one user. but for each of them there can be multiple embs so sum or take max of them.
                article_embeddings = article_embeddings.unsqueeze(dim=0)
                user_embeddings = user_embeddings.squeeze()
                pred = torch.matmul(article_embeddings, user_embeddings.t()).sum()
                scores.append(pred)
            candidate_copy.scores = scores
        else:
            candidate_copy.scores = None

        # breakpoint()

        return candidate_copy
