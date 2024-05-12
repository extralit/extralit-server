import pandas as pd
from fastapi import FastAPI, Depends, Body
from fastapi.responses import StreamingResponse

import argilla as rg

from extralit.convert.json_table import json_to_df
from extralit.extraction.extraction import extract_schema
from extralit.extraction.models.paper import PaperExtraction
from extralit.extraction.models.schema import SchemaStructure
from extralit.extraction.vector_index import create_or_load_vectorstore_index
from extralit.server.context.files import get_minio_client
from extralit.server.context.llamaindex import get_langfuse_global
from extralit.server.context.vectordb import get_weaviate_client
from extralit.server.context.datasets import get_argilla_dataset
from extralit.server.models.extraction import ExtractionRequest
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
        workspace: str='itn-recalibration',
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


@app.get("/extract/{reference}/{schema_name}")
async def extract(
        reference: str,
        schema_name: str,
        request: ExtractionRequest = Body(...),
        dataset=Depends(get_argilla_dataset, use_cache=True),
        weaviate_client=Depends(get_weaviate_client, use_cache=True),
        minio_client=Depends(get_minio_client, use_cache=True),
        global_handler=Depends(get_langfuse_global, use_cache=True),
    ):

    schemas = SchemaStructure.from_s3(minio_client=minio_client, bucket_name=dataset.workspace.name)
    schema = schemas[schema_name]

    previous_extractions = {}
    for schema_name, json_str in request.previous_extractions.items():
        previous_extractions[schema_name] = json_to_df(json_str, schema=schemas[schema_name])

    extractions = PaperExtraction(
        reference=reference,
        extractions=previous_extractions,
        schemas=schemas)

    ### Create or load the index ###
    if hasattr(global_handler, 'set_trace_params'):
        global_handler.set_trace_params(
            name=f"extract-{reference}",
            tags=[reference],
        )

    index = create_or_load_vectorstore_index(
        paper=pd.Series(name=reference),
        weaviate_client=weaviate_client,
        llm_model="gpt-3.5-turbo",
        embed_model='text-embedding-3-small',
        reindex=False,
        index_name="LlamaIndexDocumentSections",
    )

    ### Extract entities ###
    df, response = extract_schema(schema=schema, extractions=extractions, index=index, verbose=1,
                                  schema_structure=schemas)

    return df.to_json(orient='table')
