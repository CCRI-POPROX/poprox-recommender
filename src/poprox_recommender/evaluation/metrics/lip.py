from poprox_concepts.domain import CandidateSet


def least_item_promoted(reference_article_set: CandidateSet, reranked_article_set: CandidateSet, k: int = 10) -> float:
    reference_ids = [article.article_id for article in reference_article_set.articles]
    reranked_ids = [article.article_id for article in reranked_article_set.articles]
    reranked_position_map = {aid: i + 1 for i, aid in enumerate(reranked_ids)}
    lip_rank = 0

    for i, aid in enumerate(reference_ids):
        rerank_pos = reranked_position_map.get(aid, k + 1)
        if rerank_pos > k and rerank_pos > lip_rank:
            lip_rank = rerank_pos

    if lip_rank == 0:
        return 0.0

    return (lip_rank - k) / len(reference_ids)
