from datetime import datetime, timezone
from uuid import UUID

from pydantic import BaseModel


class Article(BaseModel):
    article_id: UUID = None
    title: str
    content: str = ""
    url: str
    published_at: datetime = datetime(1970, 1, 1, 0, 0, tzinfo=timezone.utc)
