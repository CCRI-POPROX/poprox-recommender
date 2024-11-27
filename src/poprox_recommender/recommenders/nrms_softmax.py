# pyright: basic
from poprox_concepts import ArticleSet, InterestProfile
from poprox_recommender.components.embedders import NRMSArticleEmbedder, NRMSUserEmbedder
from poprox_recommender.components.filters import TopicFilter
from poprox_recommender.components.joiners import Fill
from poprox_recommender.components.rankers.topk import TopkRanker
from poprox_recommender.components.samplers import SoftmaxSampler, UniformSampler
from poprox_recommender.components.scorers import ArticleScorer
from poprox_recommender.lkpipeline import Pipeline
from poprox_recommender.paths import model_file_path


def nrms_softmax_pipeline(num_slots: int, device: str) -> Pipeline:
    """
    Create a recommendation pipeline that uses NRMS for scoring and
    softmax sampling for ranking

    Args:
        num_slots: The number of items to recommend.
        device: Controls whether to use CPU or GPU. Values are "cpu" or "cuda".
    """

    # Create the components
    article_embedder = NRMSArticleEmbedder(model_file_path("nrms-mind/news_encoder.safetensors"), device)
    user_embedder = NRMSUserEmbedder(model_file_path("nrms-mind/user_encoder.safetensors"), device)

    article_scorer = ArticleScorer()
    topic_filter = TopicFilter()
    sampler = UniformSampler(num_slots=num_slots)
    fill = Fill(num_slots=num_slots)
    topk_ranker = TopkRanker(num_slots=num_slots)
    sampler = SoftmaxSampler(num_slots=num_slots, temperature=30.0)

    # Wire the components together into a pipeline
    pipeline = Pipeline(name="mmr")

    # Define pipeline inputs
    candidates = pipeline.create_input("candidate", ArticleSet)
    clicked = pipeline.create_input("clicked", ArticleSet)
    profile = pipeline.create_input("profile", InterestProfile)

    # Compute embeddings
    e_cand = pipeline.add_component("candidate-embedder", article_embedder, article_set=candidates)
    e_click = pipeline.add_component("history-embedder", article_embedder, article_set=clicked)
    e_user = pipeline.add_component("user-embedder", user_embedder, clicked_articles=e_click, interest_profile=profile)

    # Score and rank articles
    o_scored = pipeline.add_component("scorer", article_scorer, candidate_articles=e_cand, interest_profile=e_user)
    o_topk = pipeline.add_component("ranker", topk_ranker, candidate_articles=o_scored, interest_profile=e_user)
    o_rank = pipeline.add_component("reranker", sampler, candidate_articles=o_scored, interest_profile=e_user)

    # Fallback in case not enough articles came from the ranker
    o_filtered = pipeline.add_component("topic-filter", topic_filter, candidate=candidates, interest_profile=profile)
    o_sampled = pipeline.add_component("sampler", sampler, candidate=o_filtered, backup=candidates)
    pipeline.add_component("recommender", fill, candidates1=o_rank, candidates2=o_sampled)

    return pipeline
