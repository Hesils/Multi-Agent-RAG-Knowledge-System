import os

from pipelines.answering_pipeline import AnsweringPipeline
# from pipelines.rag_pipeline import RagPipeline


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
    # rag_pipeline.update_db_collection_content(collection)
    user_input = ""
    answering_pipeline = AnsweringPipeline()
    response = answering_pipeline.answer(user_input)
    print(response)



if __name__ == "__main__":
    main()