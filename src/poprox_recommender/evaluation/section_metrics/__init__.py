import logging
from typing import Any
from uuid import UUID

import pandas as pd

from poprox_concepts.domain import ImpressedSection
from poprox_recommender.data.eval import EvalData
from poprox_recommender.evaluation.section_metrics.section_lip import section_wise_lip

logger = logging.getLogger(__name__)

__all__ = ["measure_section_rec_metrics"]


def measure_section_rec_metrics(
    slate_id: UUID,
    sections: list[ImpressedSection],
    truth_df: pd.DataFrame,
    global_article_ids: list[str],
    eval_data: EvalData | None = None,
) -> dict[str, Any]:
    """
    Measure section-based metrics for a single slate.

    Args:
        slate_id: Identifier for this recommendation slate.
        sections: The ordered list of sections in the newsletter.
        truth_df: Ground-truth clicks for this slate.
        global_article_ids: Article IDs ordered by fusion score (best first).
            Used as the reference ranking for section-level LIP.
        eval_data: Optional eval dataset for article metadata lookups.

    Returns:
        Dictionary of metric name → value, including ``slate_id``.
    """
    return {
        "slate_id": slate_id,
        "section_wise_lip": section_wise_lip(global_article_ids, sections),
    }
