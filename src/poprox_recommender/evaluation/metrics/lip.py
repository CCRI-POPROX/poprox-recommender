from poprox_concepts.domain import ArticleSet


def least_item_promoted(final_recs: ArticleSet, k: int = 10) -> float:
    # final recs is sorted here
    lip_rank = 0
    for item in final_recs.articles:
        rank_item = final_recs.articles.index(item) + 1  # Shahan: what can we replace rank with?
        if rank_item > k:
            if rank_item > lip_rank:
                lip_rank = rank_item
    if lip_rank == 0:
        return 0
    return (lip_rank - k) / len(final_recs.articles)
