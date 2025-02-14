from typing import Generator

from poprox_concepts.api.recommendations import RecommendationRequest


class Data:
    @property
    def n_users(self) -> int:
        pass

    def iter_users(self) -> Generator[RecommendationRequest, None, None]:
        pass
