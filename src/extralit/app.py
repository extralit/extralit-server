from fastapi import FastAPI
import extralit

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}
