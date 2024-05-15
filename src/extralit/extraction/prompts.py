import json
from typing import List

import pandera as pa
from llama_index.core import PromptTemplate

from extralit.extraction.models import PaperExtraction
from extralit.extraction.schema import get_extraction_schema_model, drop_type_def_from_schema_json
from extralit.extraction.utils import filter_unique_columns, stringify_to_instructions

FIGURE_TABLE_EXT_PROMPT_TMPL = PromptTemplate(
    """Given the figure from a research paper, please extract only the variables and observations names of the figure/chart as columns header and rows index in an HTML table, but do not extract any numerical data values.
Figure information is below.
---------------------
{header_str}
--------------------- 
Answer:""")


def create_extraction_prompt(
        schema: pa.DataFrameSchema, extractions: PaperExtraction, filter_unique_cols=False) -> str:
    prompt = (
        f"Your task is to extract data from a malaria research paper.\n"
        f"When given tables from the paper, use Chain-of-Thought to create a mapping of the table's columns to the "
        f"`{schema.name}` schema fields.\n"
        f"If you don't find the information in the text or tables or you are not sure, "
        f"omit the key-value in your JSON response.")
    schema_structure = extractions.schemas
    dependencies = schema_structure.upstream_dependencies[schema.name]
    if dependencies:
        prompt += \
            (f"The `{schema.name}` table you're extracting should be conditioned on the provided "
             f"`{stringify_to_instructions(dependencies, conjunction='and')}` "
             f"data, however, each combination of references may have multiple `{schema.name}` data entries. "
             f"Here are the data already extracted from the paper:\n\n")

    # Inject prior extraction data into the query
    for dep_schema_name in dependencies:
        if dep_schema_name not in extractions.extractions:
            raise ValueError(f"Dependency '{dep_schema_name}' not found in extractions")

        if filter_unique_cols:
            dep_extraction = filter_unique_columns(extractions[dep_schema_name])
        else:
            dep_extraction = extractions[dep_schema_name]

        schema_json = get_extraction_schema_model(schema_structure[dep_schema_name],
                                                  include_fields=dep_extraction.columns.tolist(),
                                                  singleton=True, description_only=True).schema()
        schema_definition = json.dumps(drop_type_def_from_schema_json(schema_json))
        prompt += (f"###{dep_schema_name}###\n"
                   f"Schema:\n"
                   f"{schema_definition}\n"
                   f"Data:\n"
                   f"{dep_extraction.to_json(orient='index')}\n\n")

    return prompt


def create_completion_prompt(
        schema: pa.DataFrameSchema, extractions: PaperExtraction, include_fields: List[str], filter_unique_cols=True) -> str:
    assert schema.name in extractions.extractions, f"Schema '{schema.name}' not found in extractions"
    prompt = create_extraction_prompt(schema, extractions, filter_unique_cols)
    existing_extraction = extractions[schema.name]

    prompt += (
        f'Please complete the extraction for the `{schema.name}` schema by filling in the {include_fields} fields '
        f'for the following {len(existing_extraction)} entries.\n'
        f'f"###{schema.name}###\n'
        f"Existing data:\n"
        f"{existing_extraction.reset_index().to_json(orient='index')}\n\n"
    )

    return prompt



