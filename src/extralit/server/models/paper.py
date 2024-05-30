from typing import Dict, Optional, List, Any, Union, Annotated, Any
from pydantic import BaseModel, Field, Extra

class SegmentResponse(BaseModel):
    header: str | None
    page_number: int | None
    type: str | None = Field(None, description="The type of the segment.")


class SegmentsResponse(BaseModel):
    items: List[SegmentResponse] = Field(default_factory=list)
    class Config:
        extra = Extra.ignore