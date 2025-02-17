from collections import Counter

from poprox_concepts.domain import ArticleSet


def k_coverage_score(recs_list_a: ArticleSet, recs_list_b: ArticleSet, k: int = 1) -> float:
    # items recommended at least once
    all_articles = [article.article_id for article in recs_list_a.articles] + [
        article.article_id for article in recs_list_b.articles
    ]
    article_counts = Counter(all_articles)
    covered_articles = sum(1 for count in article_counts.values() if count > k)
    total_unique_articles = len(article_counts)  # doubt - total no. of articles???
    if total_unique_articles == 0:
        return 0.0
    coverage_score = covered_articles / total_unique_articles
    return coverage_score
