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
from extralit.extraction.schema import build_extraction_model
from extralit.extraction.vector_index import create_or_load_vectorstore_index
from extralit.schema.references.assign import assign_unique_index, get_prefix
from extralit.extraction.utils import convert_response_to_dataframe, generate_reference_columns, filter_unique_columns


def query_index(
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


def extract_schema(schema: pa.DataFrameSchema, extractions: PaperExtraction, index: VectorStoreIndex,
                   subset: Optional[List[str]] = None, headers: Optional[List[str]] = None, similarity_top_k=20,
                   text_qa_template=PromptTemplate(default_prompts.DEFAULT_TEXT_QA_PROMPT_TMPL), verbose=None,
                   **kwargs) -> Tuple[pd.DataFrame, ResponseResult]:
    """
    Extract a schema from a paper using the RAG LLM.
    Args:
        schema (pa.DataFrameSchema): The schema to extract.
        extractions (PaperExtraction): The extractions from the paper.
        index (VectorStoreIndex): The index to use for the extraction.
        similarity_top_k (int): The number of similar documents to retrieve. Defaults to 20.
        subset (Optional[List[str]]): A list of column names to include in the Pydantic model. Defaults to None.
        headers (Optional[List[str]]): The headers to filter the documents by. Defaults to None.
        text_qa_template (PromptTemplate): The text QA template to use. Defaults to the default text QA template.
        verbose (Optional[int]): The verbosity level. Defaults to None.
        **kwargs (Dict): Additional keyword arguments to pass to the `query_rag_llm` and `as_query_engine` function.
            text_qa_template (PromptTemplate): The text QA template to use. Defaults to the default text QA template.

    Returns:
        Tuple[pd.DataFrame, ResponseResult]: The extracted DataFrame and the ResponseResult.
    """
    prompt = f"""
You are a highly detailed-oriented data extractor with research domain expertise.
Use Chain-of-Thought to create a mapping of the table fields from the context to the `{schema.name}` schema fields.
It is not necessary to include "NA" values in the JSON in your response. 
"""
    schema_structure = extractions.schemas
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
    output_cls = build_extraction_model(
        schema, subset=subset, top_class=schema.name + 's', lower_class=schema.name, validate_assignment=False)

    filters = MetadataFilters(
        filters=[MetadataFilter(key="reference", value=extractions.reference, operator=FilterOperator.EQ)]
    )
    if headers:
        filters.filters.append(MetadataFilter(key="header", value=headers, operator=FilterOperator.IN))

    response = query_index(
        prompt, index=index, output_cls=output_cls,
        similarity_top_k=similarity_top_k, filters=filters,
        text_qa_template=text_qa_template, response_mode="compact", **kwargs)

    df = convert_response_to_dataframe(response)
    df = generate_reference_columns(df, schema)
    return df, ResponseResult(**response.__dict__)


def extract_paper(
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

    extractions = PaperExtraction(
        extractions={}, schemas=schema_structure, reference=reference)
    responses = ResponseResults(
        items={}, docs_metadata={id: doc.metadata for id, doc in index.docstore.docs.items()})

    ### Extract entities ###
    for schema_name in extractions.schemas.ordering:
        schema = extractions.schemas[schema_name]

        df = extract_schema_with_fallback(schema=schema, extractions=extractions, index=index, responses=responses,
                                          models=llm_models, verbose=verbose)

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


def extract_schema_with_fallback(schema: pa.DataFrameSchema, extractions: PaperExtraction, index: VectorStoreIndex,
                                 responses: ResponseResults, models: List[str], verbose: Optional[int]=0) -> pd.DataFrame:
    for model in models:
        try:
            index.service_context.llm.model = model
            df, responses[schema.name] = extract_schema(schema=schema, extractions=extractions, index=index,
                                                        verbose=verbose)
            return df
        except Exception as e:
            logging.log(logging.WARNING, f"Error {schema.name} ({model}): {e}")
            if verbose >= 2:
                raise e

    return pd.DataFrame()


