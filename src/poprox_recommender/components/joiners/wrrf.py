from collections import defaultdict
from typing import List, Optional

from lenskit.pipeline import Component
from pydantic import BaseModel

from poprox_concepts.domain import RecommendationList


class WRRFConfig(BaseModel):
    num_slots: int
    k: int = 60
    # Default weights if not specified
    default_weights: dict = {"recs1": 1.0, "recs2": 1.0}


class WeightedReciprocalRankFusion(Component):
    config: WRRFConfig

    def __call__(
        self,
        *,
        recs1: RecommendationList,
        recs2: RecommendationList,
        additional_recs: Optional[List[RecommendationList]] = None,
        weights: Optional[dict] = None,
    ) -> RecommendationList:
        """
        Perform Weighted Reciprocal Rank Fusion on multiple recommendation lists.

        Args:
            recs1: First recommendation list (required)
            recs2: Second recommendation list (required)
            additional_recs: Optional list of additional recommendation lists
            weights: Optional dictionary of weights for each list (e.g., {"recs1": 1.5, "recs2": 1.0})

        Returns:
            RecommendationList: Fused recommendation list
        """
        # Combine all recommendation lists into a single list
        all_recs = [recs1, recs2]
        if additional_recs:
            all_recs.extend([rec for rec in additional_recs if rec is not None])

        # Use default weights if none provided, otherwise use provided weights
        if weights is None:
            weights = self.config.default_weights.copy()
        else:
            # Ensure all required weights are present, fill missing with defaults
            default_weights = self.config.default_weights.copy()
            default_weights.update(weights)
            weights = default_weights

        # Initialize scoring dictionaries
        article_scores = defaultdict(float)
        articles_by_id = {}

        # Process each recommendation list with its corresponding weight
        for list_idx, rec_list in enumerate(all_recs):
            list_name = f"recs{list_idx + 1}"
            weight = weights.get(list_name, 1.0)  # Default to 1.0 if weight not specified

            for rank, article in enumerate(rec_list.articles, 1):
                # Calculate weighted score
                score = weight * (1 / (rank + self.config.k))
                article_scores[article.article_id] += score
                articles_by_id[article.article_id] = article

        # Sort articles by their weighted scores
        sorted_article_scores = sorted(article_scores.items(), key=lambda x: x[1], reverse=True)
        sorted_article_ids = [article_id for article_id, _ in sorted_article_scores]

        # Select top articles up to num_slots
        fused_articles = [articles_by_id[article_id] for article_id in sorted_article_ids[: self.config.num_slots]]

        return RecommendationList(articles=fused_articles)
