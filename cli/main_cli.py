import typer

from pipelines.answering_pipeline import AnsweringPipeline
from pipelines.rag_pipeline import RagPipeline
from rag.loaders.pdf_loader import PdfLoader
from utils.chromadb_client import chromadb_client

app = typer.Typer()

@app.command()
def answer(query: str):
    answering_pipeline = AnsweringPipeline()
    response = answering_pipeline.answer(query)
    print(response)

@app.command()
def rag_update():
    # TODO attention: sur un update complet, plusieur loader a instancier et a directory_load
    # TODO: faire un general loader qui va load en fonction du fichier rencontrer et pas un loader par type de fichier
    loader = PdfLoader()
    rag_pipeline = RagPipeline()
    collection = chromadb_client.chroma_client.get_or_create_collection(name="identity")
    rag_pipeline.update_db_collection_content(collection, chromadb_client, loader)
