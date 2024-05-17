from datetime import datetime, timezone
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel


class Entity(BaseModel):
    entity_id: Optional[UUID] = None
    external_id: Optional[str] = None
    name: str
    entity_type: str
    source: str
    raw_data: str


class Mention(BaseModel):
    article_id: Optional[UUID] = None
    mention_id: Optional[UUID] = None
    source: str
    relevance: float
    entity: Entity


class Article(BaseModel):
    article_id: UUID = None
    title: str
    content: str = ""
    url: str
    published_at: datetime = datetime(1970, 1, 1, 0, 0, tzinfo=timezone.utc)
    mentions: List[Mention] = []
