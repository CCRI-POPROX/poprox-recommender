from poprox_concepts.domain import ArticleSet


def least_item_promoted(reference_article_set: ArticleSet, reranked_article_set: ArticleSet, k: int = 10) -> float:
    # final recs is sorted here
    lip_rank = 0
    for item in reference_article_set.articles:
        rank_item = reranked_article_set.index(item) + 1
        if rank_item > k:
            if rank_item > lip_rank:
                lip_rank = rank_item
    
    if lip_rank == 0:
        return 0
    
    return (lip_rank - k) / len(reference_article_set.articles)
