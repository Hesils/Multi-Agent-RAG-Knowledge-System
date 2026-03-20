import json

from agents.ChunkRankerAgent import ChunkRankerAgent
from agents.query_agent import QueryAgent
from agents.answer_agent import AnswerAgent
from utils.chromadb_client import chromadb_client

class AnsweringPipeline:
    def __init__(self):
        self.query_agent = self.init_query_agent()
        self.answer_agent = self.init_answer_agent()
        self.chunk_ranker_agent = self.init_chunk_ranker_agent()
        self.collection = chromadb_client.chroma_client.get_or_create_collection(name="identity")


    def answer(self, user_input: str) -> str:
        reworked_query = self.query_agent.execute(user_input)
        sub_queries_chunks = chromadb_client.get_chunks_for_collection(self.collection, reworked_query.sub_queries)
        query_chunks = chromadb_client.get_chunks_for_collection(self.collection, reworked_query.optimized_query)
        ids_list, content_list, metadata_list = self.make_chunks_unique(
            query_chunks["ids"] + sub_queries_chunks["ids"],
            query_chunks["documents"] + sub_queries_chunks["documents"],
            query_chunks["metadatas"] + sub_queries_chunks["metadatas"]
        )
        formated_data = self.format_data(ids_list, content_list, metadata_list)
        ranker_output = self.chunk_ranker_agent.execute(json.dumps({
            "user_query": user_input,
            "chunks": formated_data
        }))
        ids_list, content_list, metadata_list = self.make_chunks_unique(
            query_chunks["ids"] + sub_queries_chunks["ids"],
            query_chunks["documents"] + sub_queries_chunks["documents"],
            query_chunks["metadatas"] + sub_queries_chunks["metadatas"],
            ids_to_exclude=[chunk.chunk_id for chunk in ranker_output.chunks_rank if not chunk.relevance > 0.7]
        )
        answer = self.answer_agent.execute(user_input, self.format_data(ids_list, content_list, metadata_list))
        return answer

    @staticmethod
    def make_chunks_unique(
            ids: list[str],
            contents: list[str],
            metadatas: list[dict],
            ids_to_exclude=None
    ) -> tuple[list[str], list[str], list[dict]]:

        ids_list, content_list, metadata_list = list(), list(), list()
        if ids_to_exclude is None:
            ids_to_exclude = []
        for batch in zip(ids, contents, metadatas):
            records = zip(batch[0], batch[1], batch[2])
            for rec_id, content, metadata in records:
                if rec_id in ids_list or rec_id in ids_to_exclude:
                    continue
                ids_list.append(rec_id)
                content_list.append(content)
                metadata_list.append(metadata)
        return ids_list, content_list, metadata_list

    @staticmethod
    def format_data(ids: list[str], contents: list[str], metadatas: list[dict]) -> str:
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
        return AnswerAgent(
            name="AnswerAgent",
            description="AI assistant",
            system_prompt_version="0.1.1"
        )

    @staticmethod
    def init_chunk_ranker_agent():
        return ChunkRankerAgent(
            name="ChunkRankerAgent",
            description="Chunk relevance ranker",
            system_prompt_version="0.1.2"
        )

