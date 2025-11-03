# pyright: basic
from __future__ import annotations

from typing import Generator
from uuid import UUID

import pandas as pd

from poprox_concepts.api.recommendations import RecommendationRequest


class EvalData:
    name: str

    def recommendation_truth(self, recommendation_id: UUID) -> pd.DataFrame | None: ...

    def iter_recommendation_ids(self, *, limit: int | None = None) -> Generator[int | UUID]: ...

    def iter_requests(self, *, limit: int | None = None) -> Generator[RecommendationRequest]: ...

    def lookup_request(self, id: int | UUID) -> RecommendationRequest: ...

    @property
    def n_requests(self) -> int: ...

    @property
    def n_articles(self) -> int: ...
