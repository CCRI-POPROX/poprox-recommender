import pandas as pd

from poprox_concepts.domain import ImpressedSection


def distinct_sections_clicked(sections: list[ImpressedSection], truth_df: pd.DataFrame) -> int:
    """
    Count the number of distinct sections in a section-based newsletter that
    had at least one article clicked (rating > 0 in truth_df).
    """
    article_to_section: dict[str, int] = {}
    for section_idx, section in enumerate(sections):
        for imp in section.impressions:
            article_to_section[str(imp.article.article_id)] = section_idx

    clicked_ids = set(truth_df[truth_df["rating"] > 0].index.astype(str))
    sections_with_click = {article_to_section[aid] for aid in clicked_ids if aid in article_to_section}
    return len(sections_with_click)
