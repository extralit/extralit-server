import json
from typing import List, Optional

import pandera as pa
from llama_index.core import PromptTemplate
from llama_index.core.prompts import default_prompts

from extralit.extraction.models import PaperExtraction
from extralit.extraction.schema import get_extraction_schema_model, drop_type_def_from_schema_json
from extralit.extraction.utils import filter_unique_columns, stringify_lists

FIGURE_TABLE_EXT_PROMPT_TMPL = PromptTemplate(
    """Given the figure from a research paper, please extract only the variables and observations names of the figure/chart as columns header and rows index in an HTML table, but do not extract any numerical data values.
Figure information is below.
---------------------
{header_str}
--------------------- 
Answer:""")


DATA_EXTRACTION_SYSTEM_PROMPT_TMPL = PromptTemplate(
    """Your ability to extract and summarize this context accurately is essential for effective analysis. Pay close attention to the context's language, structure, and any cross-references to ensure a comprehensive and precise extraction of information. Do not use prior knowledge or information from outside the context to answer the questions. Only use the information provided in the context to answer the questions.\n"""
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Query: {query_str}\n"
    "Answer: "
)

# DEFAULT_SYSTEM_PROMPT_TMPL = PromptTemplate(default_prompts.DEFAULT_TEXT_QA_PROMPT_TMPL)
DEFAULT_SYSTEM_PROMPT_TMPL = DATA_EXTRACTION_SYSTEM_PROMPT_TMPL


DATA_EXTRACTION_COMPLETION_PROMPT_TMPL = PromptTemplate(
    "Your task is to extract data from a malaria research paper.\n"
    "The {schema_name} details can be split across the provided context. Respond with details by looking at the whole context always.\n"
    "If you don't find the information in the given context or you are not sure, "
    "omit the key-value in your JSON response.\n"
    "{dependencies_prompt}\n"
    "{dependencies_data}\n"
)

def create_extraction_prompt(
        schema: pa.DataFrameSchema, extractions: PaperExtraction, filter_unique_cols=False) -> str:
    prompt = (
        f"Your task is to extract data from a malaria research paper.\n"
        f"The `{schema.name}` details can be split across the provided context. Respond with details by looking at the whole context always.\n"
        f"If you don't find the information in the given context or you are not sure, "
        f"omit the key-value in your JSON response. ")
    schema_structure = extractions.schemas
    dependencies = schema_structure.upstream_dependencies[schema.name]
    if dependencies:
        prompt += (
            f"The `{schema.name}` data you're extracting needs to be conditioned on the provided "
            f"`{stringify_lists(dependencies, conjunction='and')}` tables which you need to reference, "
            f"however, there can be multiple `{schema.name}` data entries for each unique combination of these references."
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
        prompt += (
            f"###{dep_schema_name}###\n"
            f"Schema:\n"
            f"{schema_definition}\n"
            f"Data:\n"
            f"{dep_extraction.to_json(orient='index')}\n\n")

    return prompt


def create_completion_prompt(
        schema: pa.DataFrameSchema, extractions: PaperExtraction, include_fields: List[str],
        filter_unique_cols=True, extra_prompt: Optional[str]=None, ) -> str:
    assert schema.name in extractions.extractions, f"Schema '{schema.name}' not found in extractions"
    prompt = create_extraction_prompt(schema, extractions, filter_unique_cols)
    existing_extraction = extractions[schema.name]

    note = f'Note: {extra_prompt}\n' if extra_prompt else ''

    prompt += (
        f'Please complete the following `{schema.name}` table by extracting the {include_fields} fields '
        f'for the following {len(existing_extraction)} entries.\n'
        f'{note}'
        f'###{schema.name}###\n'
        f'Data:\n'
        f'{existing_extraction.reset_index().to_json(orient="index")}\n\n'
    )

    return prompt



