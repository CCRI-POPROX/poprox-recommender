from inspect import signature
from typing import Callable

from poprox_concepts import ArticleSet, InterestProfile


class RecommendationPipeline:
    def __init__(self, name):
        self.name = name
        self.components = []

    def add(self, component: Callable):
        self.components.append(component)

    def __call__(self, candidate_articles: ArticleSet, interest_profile: InterestProfile) -> ArticleSet:
        # Avoid modifying the inputs
        articles = candidate_articles.model_copy()
        interests = interest_profile.model_copy()

        # Run each component in the order it was added
        for component in self.components:
            output = self.run_component(component, articles, interests)

            if isinstance(output, ArticleSet):
                articles = output
            elif isinstance(output, InterestProfile):
                interests = output
            else:
                msg = f"Pipeline components must return ArticleSet or InterestProfile, but received {type(output)}"
                raise TypeError(msg)

        # Double check that we're returning the right type for recs
        if not isinstance(articles, ArticleSet):
            msg = f"The final pipeline component must return ArticleSet, but received {type(articles)}"

        return articles

    def run_component(self, component, articles: ArticleSet, interests: InterestProfile):
        sig = signature(component)

        if len(sig.parameters) == 1:
            output = component(articles)
        elif len(sig.parameters) == 2:
            output = component(articles, interests)
        return output
