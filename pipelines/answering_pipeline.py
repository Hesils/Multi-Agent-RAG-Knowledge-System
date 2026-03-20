import json

from agents.query_agent import QueryAgent
from agents.answer_agent import AnswerAgent
from utils.chromadb_client import chromadb_client

class AnsweringPipeline:
    def __init__(self):
        self.query_agent = self.init_query_agent()
        self.answer_agent = self.init_answer_agent()
        self.collection = chromadb_client.chroma_client.get_or_create_collection(name="identity")


    def answer(self, user_input: str) -> str:
        reworked_query = self.query_agent.execute(user_input)
        sub_queries_chunks = chromadb_client.get_chunks_for_collection(self.collection, reworked_query.sub_queries)
        query_chunks = chromadb_client.get_chunks_for_collection(self.collection, reworked_query.optimized_query)
        ids_set, content_set, metadata_set = set(), set(), set()
        for batch in zip(
                query_chunks["ids"] + sub_queries_chunks["ids"],
                query_chunks["documents"] + sub_queries_chunks["documents"],
                query_chunks["metadatas"] + sub_queries_chunks["metadatas"]
        ):
            records = zip(batch[0], batch[1], batch[2])
            for rec_id, content, metadata in records:
                if rec_id in ids_set:
                    continue
                ids_set.add(rec_id)
                content_set.add(content)
                metadata_set.add(content)

        formated_data = self.format_data(ids_set, content_set, metadata_set)
        answer = self.answer_agent.execute(user_input, formated_data)
        return answer

    @staticmethod
    def format_data(ids: set[str], contents: set[str], metadatas: set[str]) -> str:
        formated_data = {}
        if len(ids) != len(contents) or len(ids) != len(metadatas):
            raise ValueError(f"ids({len(ids)}), contents({len(contents)}) and metadatas({len(metadatas)}) must contain same number of elements.")
        for rec_id, content, metadata in zip(ids, contents, metadatas):
            formated_data[rec_id] = {"content": content, "metadata": metadata}
        return json.dumps(formated_data)

    @staticmethod
    def init_query_agent():
        return QueryAgent(name="QueryAgent", description="Optimizer of RAG queries")

    @staticmethod
    def init_answer_agent():
        return AnswerAgent(name="AnswerAgent", description="AI assistant")

