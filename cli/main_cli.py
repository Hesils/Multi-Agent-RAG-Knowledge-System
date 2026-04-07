import os

import typer

from pipelines.answering_pipeline import AnsweringPipeline
from pipelines.rag_pipeline import RagPipeline
from utils.chromadb_client import chromadb_client
from rag.watchers.data_files_watcher import watch_datafiles

app = typer.Typer()

@app.command()
def answer(query: str):
    answering_pipeline = AnsweringPipeline()
    response = answering_pipeline.answer(query)
    print(response)

@app.command()
def rag_update_all():
    rag_pipeline = RagPipeline()
    collection = chromadb_client.chroma_client.get_or_create_collection(name="chunks")
    rag_pipeline.update_directory(os.environ["DATA_PATH"], collection, chromadb_client)

@app.command()
def rag_update():
    rag_pipeline = RagPipeline()
    collection = chromadb_client.chroma_client.get_or_create_collection(name="chunks")
    rag_pipeline.update_doc(r"G:\Mon Drive\test.gdoc", collection, chromadb_client)

@app.command()
def rag_get():
    collection = chromadb_client.chroma_client.get_or_create_collection(name="chunks")
    chromadb_client.get_chunks_where(["source"], [f"{os.environ['DATA_PATH']}\\CV_Desvignes_Quentin.pdf"], collection)

@app.command()
def watch_datafiles_dir():
    watch_datafiles()
