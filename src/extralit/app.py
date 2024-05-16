from typing import Optional, Union, Dict, List
from uuid import UUID

import pandas as pd
from fastapi import FastAPI, Depends, Body, Query, status, HTTPException
from fastapi.responses import StreamingResponse

import argilla as rg
from langfuse.llama_index import LlamaIndexCallbackHandler
from llama_index.core import BaseCallbackHandler

from extralit.convert.json_table import json_to_df
from extralit.extraction.extraction import extract_schema
from extralit.extraction.models.paper import PaperExtraction
from extralit.extraction.models.schema import SchemaStructure
from extralit.extraction.vector_index import create_or_load_vectorstore_index
from extralit.server.context.files import get_minio_client
from extralit.server.context.llamaindex import get_langfuse_global
from extralit.server.context.vectordb import get_weaviate_client
from extralit.server.context.datasets import get_argilla_dataset
from extralit.server.models.extraction import ExtractionRequest, ExtractionResponse
from extralit.server.utils import astreamer

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/index")
async def load_vectorstore_index(
        dataset=Depends(get_argilla_dataset, use_cache=True),
        client=Depends(get_weaviate_client, use_cache=True),
):
    return {"message": "Hello World", 'client': str(client), 'argilla_client': str(dataset)}


@app.get("/schemas/{workspace}")
async def get_schemas(
        workspace: str = 'itn-recalibration',
        minio_client=Depends(get_minio_client, use_cache=True),
):
    ss = SchemaStructure.from_s3(minio_client=minio_client, bucket_name=workspace)
    return ss.ordering


@app.get("/question/{reference}/{query}")
async def completion(
        reference: str,
        query: str,
        dataset=Depends(get_argilla_dataset, use_cache=True),
        weaviate_client=Depends(get_weaviate_client, use_cache=True),
):
    index = create_or_load_vectorstore_index(
        paper=pd.Series(name=reference),
        weaviate_client=weaviate_client,
        preprocessing_dataset=dataset,
        llm_model="gpt-3.5-turbo",
        embed_model='text-embedding-3-small',
        reindex=False,
        index_name="LlamaIndexDocumentSections",
    )

    query_engine = index.as_query_engine(
        streaming=True,
    )

    response = query_engine.query(query)
    return StreamingResponse(astreamer(response.response_gen), media_type="text/event-stream")


@app.post("/completion", status_code=status.HTTP_201_CREATED,
          response_model=ExtractionResponse)
async def completion(
        *,
        extraction_request: ExtractionRequest = Body(...),
        workspace: str = Query(...),
        username: Optional[Union[str, UUID]] = None,
        model: str = "gpt-4o",
        similarity_top_k=3,
        weaviate_client=Depends(get_weaviate_client, use_cache=True),
        minio_client=Depends(get_minio_client, use_cache=True),
        global_handler: Optional[LlamaIndexCallbackHandler] = Depends(get_langfuse_global, use_cache=True),
):
    try:
        schemas = SchemaStructure.from_s3(minio_client=minio_client, bucket_name=workspace)
        schema = schemas[extraction_request.schema_name]

        extraction_dfs = {}
        for schema_name, extraction_dict in extraction_request.extractions.items():
            schema = schemas[schema_name]
            extraction_dfs[schema.name] = json_to_df(extraction_dict, schema=schema)

        extractions = PaperExtraction(
            reference=extraction_request.reference,
            extractions=extraction_dfs,
            schemas=schemas)

        ### Create or load the index ###
        if global_handler is not None and hasattr(global_handler, 'set_trace_params'):
            global_handler.set_trace_params(
                name=f"extract-{extraction_request.reference}",
                user_id=username,
                session_id=extraction_request.reference,
                tags=[workspace, extraction_request.reference, extraction_request.schema_name],
            )

        index = create_or_load_vectorstore_index(
            paper=pd.Series(name=extraction_request.reference),
            weaviate_client=weaviate_client,
            llm_model=model,
            embed_model='text-embedding-3-small',
            index_name="LlamaIndexDocumentSections",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to create an extraction request: {e}')

    if extraction_request.headers and len(extraction_request.headers) > similarity_top_k:
        similarity_top_k = len(extraction_request.headers)

    try:
        ### Extract entities ###
        df, rag_response = extract_schema(
            schema=schema, extractions=extractions, index=index, schema_structure=schemas,
            similarity_top_k=similarity_top_k,
            include_fields=extraction_request.columns,
            headers=extraction_request.headers,
            types=extraction_request.types,
            verbose=1, )

        if df.empty:
            if rag_response.source_nodes is None or len(rag_response.source_nodes) == 0:
                raise HTTPException(status_code=404,
                                    detail='There were no context selected due to stringent filters. Please modify your filters.')
            raise HTTPException(status_code=404, detail='No entities found in the extraction')

        response = ExtractionResponse.parse_raw(df.to_json(orient='table'))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return response
