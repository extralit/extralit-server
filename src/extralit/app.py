import os
import argilla as rg

from fastapi import FastAPI, Depends

import extralit
from extralit.context.vectordb import get_weaviate_client
from extralit.context.datasets import get_argilla_client


app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/index")
async def load_vectorstore_index(
    client=Depends(get_weaviate_client),
    argilla_client=Depends(get_argilla_client),
    ):
    return {"message": "Hello World", 'client': str(client), 'argilla_client': str(argilla_client)}

