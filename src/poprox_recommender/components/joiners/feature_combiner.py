import torch as th

# import torch.nn as nn
# import torch.nn.functional as F
from lenskit.pipeline import Component

from poprox_concepts.domain import InterestProfile
from poprox_recommender.pytorch.decorators import torch_inference


class FeatureCombiner(Component):
    config: None

    @torch_inference
    def __call__(
        self, profiles_1: InterestProfile, profiles_2: InterestProfile, profiles_3: InterestProfile
    ) -> InterestProfile:
        integrated_interest_profile = profiles_1.model_copy()

        if profiles_1.embedding is None and profiles_2.embedding is None and profiles_3.embedding:
            integrated_interest_profile.embedding = th.zeros((1, 50, 768), device="cpu", dtype=th.float32)
        else:
            emb1 = []  # click
            emb2 = []  # explicit topic
            emb3 = []  # implicit topic

            if profiles_1.embedding is not None and profiles_3.embedding is not None:
                emb1 = profiles_1.embedding
                emb3 = profiles_3.embedding
            else:
                emb1 = th.zeros_like(profiles_2.embedding)
                emb3 = th.zeros_like(profiles_2.embedding)

            if profiles_2.embedding is not None:
                emb2 = profiles_2.embedding
            else:
                emb2 = th.zeros_like(profiles_1.embedding)

            # attention_layer = nn.Linear(emb1.shape[-1], 1)
            # emb_1_mean = emb1.mean(dim=1)
            # emb_2_mean = emb2.mean(dim=1)

            # weight_1 = attention_layer(emb_1_mean)
            # weight_2 = attention_layer(emb_2_mean)

            # norm_weights = F.softmax((th.cat([weight_1, weight_2], dim=1)), dim=1)

            # w1, w2 = norm_weights[0, 0], norm_weights[0, 1]

            # combined = th.cat([emb1, emb2], dim=1)
            # emb1=click
            # emb2=explicit topic

            combined = th.cat([0 * emb1, 0 * emb2, 1 * emb3], dim=1)

            integrated_interest_profile.embedding = combined

        # breakpoint()
        return integrated_interest_profile
