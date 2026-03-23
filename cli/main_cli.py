import os

import typer

from pipelines.answering_pipeline import AnsweringPipeline
from pipelines.rag_pipeline import RagPipeline
from utils.chromadb_client import chromadb_client

app = typer.Typer()

@app.command()
def answer(query: str):
    answering_pipeline = AnsweringPipeline()
    response = answering_pipeline.answer(query)
    print(response)

@app.command()
def rag_update():
    rag_pipeline = RagPipeline()
    collection = chromadb_client.chroma_client.get_or_create_collection(name="identity")
    rag_pipeline.update_directory(os.environ["DATA_PATH"], collection, chromadb_client)

@app.command()
def rag_get():
    collection = chromadb_client.chroma_client.get_or_create_collection(name="identity")
    chromadb_client.get_chunks_where(["source"], [""], collection)
