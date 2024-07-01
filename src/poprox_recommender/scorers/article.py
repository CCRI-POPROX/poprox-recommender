import numpy as np
import torch as th


class ArticleScorer:
    def __init__(self, model):
        self.model = model

    def __call__(self, candidate_embeddings: th.Tensor, user_embedding: th.Tensor) -> np.ndarray:
        pred = self.model.get_prediction(candidate_embeddings, user_embedding.squeeze())
        return pred.cpu().detach().numpy()
