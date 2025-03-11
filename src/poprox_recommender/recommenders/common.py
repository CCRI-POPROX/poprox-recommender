"""
Common pipeline definition components shared between (most) pipeline
configurations.
"""

# pyright: basic
import logging

from lenskit.pipeline import ComponentConstructor, Node, PipelineBuilder

from poprox_concepts import CandidateSet, InterestProfile
from poprox_recommender.components.filters import TopicFilter
from poprox_recommender.components.joiners import Fill
from poprox_recommender.components.rankers.topk import TopkRanker
from poprox_recommender.components.samplers import UniformSampler
from poprox_recommender.components.scorers import ArticleScorer

logger = logging.getLogger(__name__)


def add_inputs(builder: PipelineBuilder):
    """
    Set up the standard inputs for a pipeline.  It adds the following input nodes:

    ``candidate``
        The set of candidate articles.

    ``clicked``
        The set of articles the user has clicked.

    ``profile``
        The user's :class:`InterestProfile`.
    """
    # Define pipeline inputs
    builder.create_input("candidate", CandidateSet)
    builder.create_input("clicked", CandidateSet)
    builder.create_input("profile", InterestProfile)


def add_article_embedder(builder: PipelineBuilder, embedder: ComponentConstructor, config=None, **kwargs):
    """
    Add an article embedder for both candidates and user history.

    It adds the following component nodes:

    ``candidate-embedder``
        Embed the candidate articles (from ``candidate``).
    ``history-embedder``
        Embed the user's clicked articles (from ``clicked``).
    """
    if config is not None and kwargs:
        raise ValueError("cannot specify both configuration object and kwargs")

    builder.add_component("candidate-embedder", embedder, config or kwargs, article_set=builder.node("candidate"))
    builder.add_component("history-embedder", embedder, config or kwargs, article_set=builder.node("clicked"))


def add_user_embedder(builder: PipelineBuilder, embedder: ComponentConstructor, config=None, **kwargs) -> Node:
    """
    Add an user embedder component.

    It adds the ``user-embedder`` component node.
    """
    if config is not None and kwargs:
        raise ValueError("cannot specify both configuration object and kwargs")
    return builder.add_component(
        "user-embedder",
        embedder,
        config or kwargs,
        candidate_articles=builder.node("candidate"),
        clicked_articles=builder.node("history-embedder"),
        interest_profile=builder.node("profile"),
    )


def add_scorer(builder: PipelineBuilder, scorer: ComponentConstructor = ArticleScorer, config=None, **kwargs) -> Node:
    """
    Add an article scorer component.  The scorer is added as the ``scorer``
    component node, wired to the ``candidate-embedder`` and ``user-embedder`` as
    inputs.
    """
    if config is not None and kwargs:
        raise ValueError("cannot specify both configuration object and kwargs")

    return builder.add_component(
        "scorer",
        scorer,
        config or kwargs,
        candidate_articles=builder.node("candidate-embedder"),
        interest_profile=builder.node("user-embedder"),
    )


def add_rankers(
    builder: PipelineBuilder,
    reranker: ComponentConstructor | None = None,
    *,
    num_slots: int,
    recommender: bool = False,
    **kwargs,
) -> Node:
    """
    Add the ranker and, optionally, a reranker to the pipeline.

    Args:
        reranker:
            The reranker.  If supplied, it is added as the ``reranker`` node.
        num_slots:
            The number of recommendation slots.  It is passed as a configuration
            option to both the reranker and to the default :class:`TopkRanker`
            node.
        recommender:
            If ``True``, alias the ``recommender`` node to the installed ranker
            or reranker (primarily for configurations where there is no
            fallback).
    """
    node = builder.add_component(
        "ranker",
        TopkRanker,
        {"num_slots": num_slots},
        candidate_articles=builder.node("scorer"),
    )
    if reranker is not None:
        node = builder.add_component(
            "reranker",
            reranker,
            {"num_slots": num_slots} | kwargs,
            candidate_articles=builder.node("scorer"),
            interest_profile=builder.node("user-embedder"),
        )

    if recommender:
        builder.alias("recommender", node)

    return node


def add_topic_fallback(builder: PipelineBuilder, *, num_slots: int):
    """
    Add the topic-sampling fallback logic, and wire it up as the default
    recommender (the ``recommender`` node).
    """

    builder.remove_alias("recommender", exist_ok=True)

    o_filtered = builder.add_component(
        "topic-filter", TopicFilter, candidate=builder.node("candidate"), interest_profile=builder.node("profile")
    )
    o_sampled = builder.add_component(
        "sampler", UniformSampler, candidates1=o_filtered, candidates2=builder.node("candidate")
    )

    o_rank = builder.node("reranker", missing=None)
    if o_rank is None:
        o_rank = builder.node("ranker")

    builder.add_component("recommender", Fill, {"num_slots": num_slots}, recs1=o_rank, recs2=o_sampled)
