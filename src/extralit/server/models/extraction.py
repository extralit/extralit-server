from pydantic.v1 import BaseModel, Field
from typing import Dict

from extralit.convert.json_table import json_to_df
from extralit.extraction.models.schema import SchemaStructure


class ExtractionRequest(BaseModel):
    reference: str
    workspace: str
    current_extraction: Dict[str, str] = Field(
        default_factory=dict,
        description="Current extraction where the key is the schema name and "
                    "the value is a tuple specifying the cell range."
    )
    previous_extractions: Dict[str, str] = Field(
        default_factory=dict,
        description="Previous extraction tables where the key is the schema name and "
                    "the value is json-encoded dataframe.")