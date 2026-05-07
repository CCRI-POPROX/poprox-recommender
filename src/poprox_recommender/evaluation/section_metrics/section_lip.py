import numpy as np

from poprox_concepts.domain import ImpressedSection


def section_wise_lip(
    global_article_ids: list[str],
    sections: list[ImpressedSection],
    k: int = 3,
) -> float:
    """
    For each section, compute LIP between the global fusion ranking and the
    top-k articles chosen for that section. Return the average across sections.

    LIP (Least Item Promoted) = global rank of the least-promoted article
    among the top-k chosen, minus k. Higher value means a low-ranked article
    was pulled into the section (more promotion by the topic filter).
    """
    if not global_article_ids or not sections:
        return np.nan

    # rank_lookup maps article_id string -> 0-based position in the global list
    # position 0 = highest fusion score (rank 1)
    rank_lookup = {article_id: rank for rank, article_id in enumerate(global_article_ids)}

    section_lips = []
    for section in sections:
        lip = _lip_for_section(rank_lookup, section, k)
        if not np.isnan(lip):
            section_lips.append(lip)

    if not section_lips:
        return np.nan

    return float(np.mean(section_lips))


def _lip_for_section(
    rank_lookup: dict[str, int],
    section: ImpressedSection,
    k: int,
) -> float:
    """
    LIP for a single section.

    Same math as lip.py: start lip_rank at k (no promotion baseline).
    For each article chosen for the section, look up its global rank.
    If that rank is worse (higher number) than lip_rank, update lip_rank.
    Return lip_rank - k: how far below rank k the least-promoted article sat globally.
    """
    # collect the article IDs actually placed in this section (up to k)
    section_ids = [str(imp.article.article_id) for imp in section.impressions[:k]]

    if not section_ids:
        return np.nan

    lip_rank = k  # baseline: assume all articles were already in the top-k globally
    for article_id in section_ids:
        global_rank = rank_lookup.get(article_id)
        if global_rank is not None and global_rank > lip_rank:
            lip_rank = global_rank  # this article was pulled from further down

    return float(lip_rank - k)
