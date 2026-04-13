import logging
from typing import TypeAlias

from lenskit.pipeline import Component
from pydantic import BaseModel

from poprox_concepts.domain import ImpressedSection

logger = logging.getLogger(__name__)


class AddSectionConfig(BaseModel):
    title: str | None = None
    personalized: bool | None = None


class AppendSection(Component):
    config: AddSectionConfig

    def __call__(
        self,
        new_section: ImpressedSection,
        existing_sections: list[ImpressedSection] | None = None,
    ) -> list[ImpressedSection]:
        existing_sections = existing_sections or []

        if self.config.title:
            new_section.title = self.config.title

        if self.config.personalized:
            new_section.personalized = self.config.personalized

        if len(new_section.impressions) > 0:
            existing_sections.append(new_section)

        return existing_sections


AddSection: TypeAlias = AppendSection


class PrependSection(Component):
    config: AddSectionConfig

    def __call__(
        self,
        new_section: ImpressedSection,
        existing_sections: list[ImpressedSection] | None = None,
    ) -> list[ImpressedSection]:
        existing_sections = existing_sections or []

        if self.config.title:
            new_section.title = self.config.title

        if self.config.personalized:
            new_section.personalized = self.config.personalized

        if len(new_section.impressions) > 0:
            existing_sections.insert(0, new_section)

        return existing_sections
