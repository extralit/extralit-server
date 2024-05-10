import logging
import os
import warnings
from os.path import join, exists
from typing import Tuple, Dict, Optional, List

import pandas as pd
import pandera as pa
from llama_index.core import VectorStoreIndex, PromptTemplate, Response
from llama_index.core.prompts import default_prompts
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator, )
from pydantic.v1 import BaseModel

from extralit.extraction.models.paper import PaperExtraction, SchemaStructure
from extralit.extraction.models.response import ResponseResult, ResponseResults
from extralit.extraction.schema import convert_schema_to_pydantic_model
from extralit.extraction.vector_index import create_or_load_vectorstore_index
from extralit.schema.references.assign import assign_unique_index, get_prefix


def query_rag_llm(
        prompt: str,
        index: VectorStoreIndex,
        output_cls=BaseModel,
        similarity_top_k=20,
        filters: Optional[MetadataFilters] = None,
        response_mode="compact",
        text_qa_template=PromptTemplate(default_prompts.DEFAULT_TEXT_QA_PROMPT_TMPL),
        **kwargs,
) -> Response:
    warnings.filterwarnings('ignore', module='pydantic')

    query_engine = index.as_query_engine(
        output_cls=output_cls,
        response_mode=response_mode,
        similarity_top_k=similarity_top_k,
        filters=filters,
        text_qa_template=text_qa_template,
        **kwargs
    )

    obs_response = query_engine.query(prompt)

    return obs_response


def convert_response_to_dataframe(response: Response) -> pd.DataFrame:
    try:
        df: pd.DataFrame = response.response.to_df()
    except AttributeError:
        logging.error(f"Failed to convert response to DataFrame: {response}")
        df = pd.DataFrame()
    return df


def generate_reference_columns(df: pd.DataFrame, schema: pa.DataFrameSchema):
    index_names = [index.name.lower() for index in schema.index.indexes] \
        if hasattr(schema.index, 'indexes') else []
    for index_name in index_names:
        if index_name not in df.columns:
            df[index_name] = 'NOTMATCHED'
    if index_names:
        df = df.set_index(index_names, verify_integrity=False)
    return df


def extract_entity(
        index: VectorStoreIndex,
        extractions: PaperExtraction,
        schema: pa.DataFrameSchema,
        schema_structure: SchemaStructure,
        similarity_top_k=20,
        verbose=None, **kwargs) -> Tuple[pd.DataFrame, ResponseResult]:
    prompt = f"""
You are a highly detailed-oriented data extractor with research domain expertise.
Use Chain-of-Thought to create a mapping of the table fields from the context to the `{schema.name}` schema fields.
It is not necessary to include "NA" values in the JSON in your response. 
"""

    dep_schemas = schema_structure.upstream_dependencies[schema.name]
    if dep_schemas:
        prompt += \
            f"The `{schema.name}` table to extract should be conditioned on the provided `{', '.join(dep_schemas)}` " \
            f"entries, however, each combination of references may have multiple `{schema.name}` data entries. " \
            f"Here are the {dep_schemas} already extracted from the paper:\n\n"

    # Inject prior extraction data into the query
    for dep_schema in dep_schemas:
        if dep_schema not in extractions.extractions:
            raise ValueError(f"Dependency '{dep_schema}' not found in extractions")

        dep_extraction = filter_unique_columns(extractions[dep_schema])
        prompt += f"###{dep_schema}###\n`\n{dep_extraction.to_json(orient='index')}\n`\n"

    if verbose:
        logging.info(f'\nSCHEMA: {schema.name}\nPROMPT: {prompt}')

    # Call the call_rag_llm function
    output_cls = convert_schema_to_pydantic_model(
        schema, top_class=schema.name + 's', lower_class=schema.name, skip_validators=True)

    filters = MetadataFilters(
        filters=[MetadataFilter(key="reference", value=extractions.reference, operator=FilterOperator.EQ)]
    )

    response = query_rag_llm(
        prompt, index=index, output_cls=output_cls,
        similarity_top_k=similarity_top_k, filters=filters, **kwargs)

    df = convert_response_to_dataframe(response)
    df = generate_reference_columns(df, schema)
    return df, ResponseResult(**response.__dict__)


def extract_with_fallback(
        index: VectorStoreIndex,
        extractions: PaperExtraction,
        responses: ResponseResults,
        schema: pa.DataFrameSchema,
        schema_structure: SchemaStructure,
        models: List[str],
        verbose: Optional[int]) -> pd.DataFrame:
    for model in models:
        try:
            index.service_context.llm.model = model
            df, responses[schema.name] = extract_entity(index, extractions=extractions, schema=schema,
                                                        schema_structure=schema_structure, verbose=verbose)
            return df
        except Exception as e:
            logging.log(logging.WARNING, f"Error {schema.name} ({model}): {e}")
            if verbose >= 2:
                raise e

    return pd.DataFrame()


def extract_from_schemas(
        paper: pd.Series,
        schema_structure: SchemaStructure,
        index: VectorStoreIndex = None,
        llm_models=["gpt-3.5-turbo", 'gpt-4-turbo'],
        embed_model='text-embedding-ada-002',
        index_kwargs: Dict = None,
        interim_path='data/interim/',
        load_only=False,
        verbose: int = 0,
) -> Tuple[PaperExtraction, ResponseResults]:
    reference = paper.name
    if isinstance(llm_models, str):
        llm_models = [llm_models]

    ### Load interim results ###
    interim_save_dir = join(interim_path, llm_models[0], reference)
    if load_only:
        if not exists(interim_save_dir):
            raise FileNotFoundError(f"Interim save directory does not exist: {interim_save_dir}")
        with open(join(interim_save_dir, 'responses.json'), 'r') as file:
            responses = ResponseResults.parse_raw(file.read())

        extractions = PaperExtraction(
            extractions={k: v.response.to_df() for k, v in responses.items.items()},
            schemas=schema_structure,
        )
        return extractions, responses

    ### Create or load the index ###
    if index is None:
        index = create_or_load_vectorstore_index(paper, llm_model=llm_models[0], embed_model=embed_model,
                                                 **(index_kwargs or {}))
    assert index.service_context.llm.model == llm_models[0], \
        f"LLM model mismatch: {index.service_context.llm.model} != {llm_models[0]}"

    extractions = PaperExtraction(extractions={}, schemas=schema_structure, reference=reference)
    responses = ResponseResults(items={}, docs_metadata={id: doc.metadata for id, doc in index.docstore.docs.items()})

    ### Extract entities ###
    for schema_name in schema_structure.ordering:
        schema = schema_structure[schema_name]

        df = extract_with_fallback(
            index, extractions=extractions, responses=responses, schema=schema,
            schema_structure=schema_structure, models=llm_models, verbose=verbose)

        if schema.index.name:
            df = assign_unique_index(df, schema, index_name=schema.index.name, prefix=get_prefix(schema), n_digits=2)
        df = df.drop_duplicates()

        extractions.extractions[schema_name] = df

    ### Save interim results ###
    try:
        os.makedirs(interim_save_dir, exist_ok=True)
        with open(join(interim_save_dir, 'responses.json'), 'w') as file:
            file.write(responses.json())
    except Exception as e:
        logging.error(f"Interim save responses: {e}")

    return extractions, responses


def filter_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that have the same value in all rows.
    """
    if len(df) > 1:
        return df.dropna(axis='columns', how='all').loc[:, (df.astype(str).nunique() > 1)]
    else:
        return df
