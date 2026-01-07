# pyright: basic
from __future__ import annotations

from typing import Generator
from uuid import UUID

import pandas as pd

from poprox_concepts.api.recommendations import RecommendationRequest


class EvalData:
    name: str

    def slate_truth(self, slate_id: UUID) -> pd.DataFrame | None: ...

    def iter_slate_ids(self, *, limit: int | None = None) -> Generator[UUID]: ...

    def lookup_request(self, slate_id: UUID) -> RecommendationRequest: ...

    @property
    def n_requests(self) -> int: ...

    @property
    def n_articles(self) -> int: ...
