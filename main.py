import os

from pipelines.answering_pipeline import AnsweringPipeline
from pipelines.rag_pipeline import RagPipeline
from utils.chromadb_client import chromadb_client


def init():
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError("OPENAI_API_KEY must be valued in environment")
    if "CHROMADB_PATH" not in os.environ:
        raise ValueError("CHROMADB_PATH must be valued in environment")
    if "DATA_PATH" not in os.environ:
        raise ValueError("DATA_PATH must be valued in environment")


def main():
    init()
    # rag_pipeline = RagPipeline()
    # collection = chromadb_client.chroma_client.get_or_create_collection(name="identity")
    # rag_pipeline.update_db_collection_content(collection, chromadb_client)
    user_input = "Quelles librairies python a utilisé Mr Desvignes lors de son emploi à la MAIF ?"
    answering_pipeline = AnsweringPipeline()
    response = answering_pipeline.answer(user_input)
    print(response)



if __name__ == "__main__":
    main()