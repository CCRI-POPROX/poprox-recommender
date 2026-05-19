import pandas as pd

from poprox_concepts.domain import ImpressedSection


def section_click_through_rate(sections: list[ImpressedSection], truth_df: pd.DataFrame) -> float:
    """
    Section Click Through Rate (SCTR): mean per-section CTR across all sections.

    For each section, CTR = clicked articles / articles shown.
    Sections with no clicks contribute 0.
    """
    clicked_ids = set(truth_df[truth_df["rating"] > 0].index.astype(str))

    per_section_ctrs = []
    for section in sections:
        total = len(section.impressions)
        if total == 0:
            continue
        clicks = sum(1 for imp in section.impressions if str(imp.article.article_id) in clicked_ids)
        per_section_ctrs.append(clicks / total)

    return sum(per_section_ctrs) / len(per_section_ctrs) if per_section_ctrs else 0.0
