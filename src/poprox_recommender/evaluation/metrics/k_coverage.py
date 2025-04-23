from collections import Counter

from poprox_concepts.domain import CandidateSet


def k_coverage_score(ranked: CandidateSet, reranked: CandidateSet, k: int = 1) -> float:
    rank_articles = {article.article_id for article in ranked.articles}
    reranked_counts = Counter(
        article.article_id for article in reranked.articles if article.article_id in rank_articles
    )

    covered_articles = sum(count >= k for count in reranked_counts.values())

    return covered_articles / len(rank_articles) if rank_articles else 0.0
