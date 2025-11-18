import numpy as np
from lenskit.pipeline import Component
from pydantic import BaseModel

from poprox_concepts.domain import CandidateSet, ImpressedSection


class SoftmaxConfig(BaseModel):
    num_slots: int
    temperature: float = 20.0


class SoftmaxSampler(Component):
    config: SoftmaxConfig

    def __call__(self, candidate_articles: CandidateSet) -> ImpressedSection:
        if candidate_articles.scores is None:
            scores = np.ones(len(candidate_articles.articles))
        else:
            scores = sigmoid(candidate_articles.scores)

        # Exponential sort trick for sampling from a distribution without replacement from:

        # Pavlos S. Efraimidis, Paul G. Spirakis, Weighted random sampling with a reservoir,
        # Information Processing Letters, Volume 97, Issue 5, 2006, Pages 181-185, ISSN 0020-0190,
        # https://doi.org/10.1016/j.ipl.2005.11.003.

        # As implemented by Tim Vieira in "Algorithms for sampling without replacement"
        # https://timvieira.github.io/blog/post/2019/09/16/algorithms-for-sampling-without-replacement/

        # The weights for the sampling distribution are the softmax of the scores
        # Scores are squashed into the range [0,1] to make tuning the temperature easier
        weights = np.exp(self.config.temperature * scores) / np.sum(scores)

        # This is the core of the exponential sampling trick, which creates a
        # set of values that depend on both the predicted scores and random
        # variables, resulting in a set of values that will sort into an order
        # that reflects sampling without replacement according to the weight
        # distribution
        num_items = len(candidate_articles.articles)
        exponentials = -np.log(np.random.uniform(0, 1, size=(num_items,)))
        exponentials /= weights

        # This is just bookkeeping to produce the final ordered list of recs
        sorted_indices = np.argsort(exponentials)
        sampled = [candidate_articles.articles[idx] for idx in sorted_indices[: self.config.num_slots]]

        return ImpressedSection.from_articles(sampled)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
