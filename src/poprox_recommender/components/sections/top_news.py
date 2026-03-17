import logging

from lenskit.pipeline import Component
from pydantic import BaseModel

from poprox_concepts.domain import ImpressedSection

logger = logging.getLogger(__name__)


class PersonalizedTopNewsConfig(BaseModel):
    max_articles: int = 3


class PersonalizedTopNews(Component):
    config: PersonalizedTopNewsConfig

    def __call__(
        self,
        ptn_section: ImpressedSection,
        sections: list[ImpressedSection] | None = None,
    ) -> list[ImpressedSection]:
        sections = sections or []

        ptn_section.title = "Your Top Stories"
        ptn_section.personalized = True

        if len(ptn_section.impressions) > 0:
            sections.append(ptn_section)

        return sections
