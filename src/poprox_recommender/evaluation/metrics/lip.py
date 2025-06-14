from poprox_concepts.domain import CandidateSet


def least_item_promoted(reference_article_set: CandidateSet, reranked_article_set: CandidateSet, k: int = 10) -> float:
    """
    ValueError: Article(article_id=UUID('6b6b748e-0212-5b29-bf90-3b97f6da540d'), headline='', subhead=None, body=None,
    url=None,preview_image_id=None, mentions=[], source=None, external_id=None, raw_data=None, images=None,
    published_at=datetime.datetime(1970, 1, 1, 0, 0, tzinfo=datetime.timezone.utc),created_at=None) is not in list
    """

    lip_rank = 0
    reranked_articles_list = reranked_article_set.articles

    for item in reference_article_set.articles:
        try:
            rank_item = (
                next(i for i, article in enumerate(reranked_articles_list[:k]) if article.article_id == item.article_id)
                + 1
            )
        except StopIteration:
            rank_item = k + 1

        if item in reranked_articles_list:
            rank_item = reranked_articles_list.index(item) + 1
        else:
            rank_item = k + 1

        if rank_item > k and rank_item > lip_rank:
            lip_rank = rank_item

    if lip_rank == 0:
        return 0

    return (lip_rank - k) / len(reference_article_set.articles)
