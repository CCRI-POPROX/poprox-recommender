# pyright: basic
from __future__ import annotations

from typing import Generator
from uuid import UUID

import pandas as pd

from poprox_concepts.api.recommendations import RecommendationRequest


class EvalData:
    name: str

    def profile_truth(self, newsletter_id: UUID) -> pd.DataFrame | None: ...

    def iter_profiles(self, *, limit: int | None = None) -> Generator[RecommendationRequest]: ...

    def iter_profile_ids(self, *, limit: int | None = None) -> Generator[int | UUID]: ...

    def lookup_request(self, id: int | UUID) -> RecommendationRequest: ...

    @property
    def n_profiles(self) -> int: ...

    @property
    def n_articles(self) -> int: ...
