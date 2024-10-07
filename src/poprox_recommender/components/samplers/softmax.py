import numpy as np

from poprox_concepts import ArticleSet


class SoftmaxSampler:
    def __init__(self, num_slots, temperature=20.0):
        self.num_slots = num_slots
        self.temperature = temperature

    def sort_score(self, candidate_articles: ArticleSet):
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
        weights = np.exp(self.temperature * scores) / np.sum(scores)

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

        candidate_articles.articles = [candidate_articles.articles[idx] for idx in sorted_indices]
        candidate_articles.scores = [exponentials[idx] for idx in sorted_indices]
        return candidate_articles

    def __call__(self, candidate_articles: ArticleSet):
        sorted_articles = self.sort_score(candidate_articles)

        return ArticleSet(articles=sorted_articles.articles[: self.num_slots])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
