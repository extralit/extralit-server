from uuid import UUID, uuid4
from typing import Any, Dict, Generic, List, Literal, Optional, TypeVar, Union
from pydantic import BaseModel, Field

class DocumentCreate(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    url: Optional[str]
    file_data: Optional[bytes]  # Expect a base64 encoded string
    file_name: Optional[str]
    pmid: Optional[str]
    doi: Optional[str]
    workspace_id: UUID  # The workspace ID to which the document belongs to

class DocumentDelete(BaseModel):
    url: Optional[str] = None
    id: Optional[Union[UUID, str]] = None
    pmid: Optional[str] = None
    doi: Optional[str] = None


class DocumentListItem(BaseModel):
    id: UUID
    url: Optional[str]
    file_name: Optional[str]
    pmid: Optional[str]
    doi: Optional[str]
    workspace_id: UUID