import os

from pipelines.rag_pipeline import RagPipeline
from agents.answer_agent import AnswerAgent

def init():
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError("OPENAI_API_KEY must be valued in environment")
    if "CHROMADB_PATH" not in os.environ:
        raise ValueError("CHROMADB_PATH must be valued in environment")
    if "DATA_PATH" not in os.environ:
        raise ValueError("DATA_PATH must be valued in environment")


def main():
    init()
    rag_pipeline = RagPipeline()
    collection = rag_pipeline.chroma_client.get_or_create_collection(name="identity")
    # rag_pipeline.update_db_collection_content(collection)
    agent = AnswerAgent(name="AnswerAgent", description="AI assistant")
    user_input = "Combien d'années d'experience Monsieur Desvignes a ?"
    query_results = rag_pipeline.get_chunks_for_collection(collection, [user_input])
    response = agent.execute(user_input, rag_pipeline.build_data(query_results))
    print(response)



if __name__ == "__main__":
    main()