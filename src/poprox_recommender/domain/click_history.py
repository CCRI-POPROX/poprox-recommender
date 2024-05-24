from typing import List
from uuid import UUID

from pydantic import BaseModel


class ClickHistory(BaseModel):
    account_id: UUID = None
    article_id: List[UUID]
