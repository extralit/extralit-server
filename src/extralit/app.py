import pandas as pd
from fastapi import FastAPI, Depends
from fastapi.responses import StreamingResponse

from extralit.extraction.vector_index import create_or_load_vectorstore_index
from extralit.server.context.vectordb import get_weaviate_client
from extralit.server.context.datasets import get_argilla_dataset
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


index = create_or_load_vectorstore_index(
    paper=pd.Series(name="pennetier2013efficacy"),
    weaviate_client=get_weaviate_client(),
    preprocessing_dataset=get_argilla_dataset(),
    llm_model="gpt-3.5-turbo",
    embed_model='text-embedding-3-small',
    reindex=False,
    index_name="LlamaIndexDocumentSections",
)

@app.get("/question/{input}")
async def create_item(input: str):
    query_engine = index.as_query_engine(
        streaming=True,
    )
    response = query_engine.query(input)
    return StreamingResponse(astreamer(response.response_gen), media_type="text/event-stream")
