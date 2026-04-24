import json
import os
from time import time

from agents.ChunkRankerAgent import ChunkRankerAgent
from agents.query_agent import QueryAgent
from agents.answer_agent import AnswerAgent
from agents.answer_critic_agent import AnswerCriticAgent
from utils.chromadb_client import chromadb_client
from utils.types.agent_types import AnswerSource
from utils.metrics import metrics
# from utils.trace_client import TraceClient
from utils.bm25_utils import compute_bm25_scores

BASE_URL = "http://localhost:8000"

class AnsweringPipeline:
    def __init__(self):
        self.query_agent = self.init_query_agent()
        self.chunk_ranker_agent = self.init_chunk_ranker_agent()
        self.answer_agent = self.init_answer_agent()
        self.answer_critic_agent = self.init_answer_critic_agent()
        self.collection = chromadb_client.chroma_client.get_or_create_collection(name="identity")


    def answer(self, user_input: str) -> str:
        # trace_client = TraceClient()
        # trace_id = trace_client.start(pipeline="answer_pipeline", query=user_input)
        metrics.add("request", {})
        req_start_time = time()
        # trace_client.step("start", {"query": user_input})
        # --------- Retrieve
        formated_data = self.retrieve_data(
            user_input,
            # trace_client
        )

        answer_try = 0
        # --------- Answer
        answer_agent_output = self.answer_agent.execute(
            user_input,
            "user",
            formated_data,
            # trace_client
        )
        # ---------- Critic loop
        while answer_try < 5:
            answer_critic = self.answer_critic_agent.execute(json.dumps({
                "user_query": user_input,
                "answer": answer_agent_output.answer,
                "chunks": formated_data,
                "answer_sources": [source.model_dump_json() for source in answer_agent_output.sources]
            }), role="user",
                # trace_client=trace_client
            )
            if answer_critic.is_valid:
                break
            else:
                answer_try += 1
                answer_agent_output = self.answer_agent.execute(
                    answer_critic.model_dump_json(),
                    "system",
                    # trace_client=trace_client
                )
        total_duration = time() - req_start_time
        metrics.add("request", {"duration": total_duration})

        final_answer = "\n".join([answer_agent_output.answer, self.format_sources(answer_agent_output.sources)])
        # trace_client.step("final_answer", {
        #     "length": len(final_answer),
        #     "sources_count": len(answer_agent_output.sources)
        # }, duration=total_duration)

        # trace_client.end(final_answer)
        return final_answer

    def retrieve_data(
            self,
            user_input: str,
            # trace_client: TraceClient
    ) -> str:
        retrieval_start_time = time()
        # -------- QUERY REWRITE --------
        reworked_query = self.query_agent.execute(
            user_input,
            role="user",
            # trace_client=trace_client
        )

        # trace_client.step("query_rewrite", {
        #     "optimized_query": reworked_query.optimized_query,
        #     "sub_queries": reworked_query.sub_queries
        # })
        # -------- VECTOR SEARCH --------
        sub_queries_chunks = chromadb_client.get_chunks_for_collection(self.collection, reworked_query.sub_queries)
        query_chunks = chromadb_client.get_chunks_for_collection(self.collection, reworked_query.optimized_query)
        ids_list, content_list, metadata_list, distance_list = self.make_chunks_unique(
            query_chunks["ids"] + sub_queries_chunks["ids"],
            query_chunks["documents"] + sub_queries_chunks["documents"],
            query_chunks["metadatas"] + sub_queries_chunks["metadatas"],
            query_chunks["distances"] + sub_queries_chunks["distances"]
        )
        # -------- VECTOR SCORES --------
        vector_scores = self.normalize_vector_scores(distance_list)
        # --------- BM25 ---------
        bm25_scores = compute_bm25_scores(
            reworked_query.optimized_query,
            list(content_list)
        )
        # -------- HYBRID SCORING --------
        hybrid_scores = [
            0.7 * v + 0.3 * b
            for v, b in zip(vector_scores, bm25_scores)
        ]

        # Sort
        ranked = sorted(
            zip(ids_list, content_list, metadata_list, hybrid_scores),
            key=lambda x: x[3],
            reverse=True
        )
        ids_list, content_list, metadata_list, _ = zip(*ranked)
        metrics.add("documents", {"count": len(set(metadata["source"] for metadata in metadata_list))})
        # trace_client.step("retrieval", {
        #     "nb_chunks": len(ids_list)
        # })

        formated_data = self.format_data(ids_list, content_list, metadata_list)
        metrics.add("retrieval_time", {"duration": time() - retrieval_start_time})
        # -------- RERANK --------
        rerank_stime = time()
        ranker_output = self.chunk_ranker_agent.execute(json.dumps({
            "user_query": user_input,
            "chunks": formated_data
        }),
            role="user",
            # trace_client=trace_client
        )
        ids_list, content_list, metadata_list, distance_list = self.make_chunks_unique(
            query_chunks["ids"] + sub_queries_chunks["ids"],
            query_chunks["documents"] + sub_queries_chunks["documents"],
            query_chunks["metadatas"] + sub_queries_chunks["metadatas"],
            query_chunks["distances"] + sub_queries_chunks["distances"],
            ids_to_exclude=[chunk.chunk_id for chunk in ranker_output.chunks_rank if not chunk.relevance > 0.7]
        )
        formated_data = self.format_data(ids_list, content_list, metadata_list)
        rerank_time = time() - rerank_stime
        metrics.add("rerank_time", {"duration": rerank_time})
        # trace_client.step("rerank", {
        #     "kept_chunks": len(ids_list)
        # }, duration=rerank_time)
        return formated_data

    @staticmethod
    def make_chunks_unique(
            ids: list[list[str]],
            contents: list[list[str]],
            metadatas: list[list[dict]],
            distances: list[list[float]],
            ids_to_exclude=None
    ) -> tuple[list[str], list[str], list[dict], list[float]]:

        ids_list, content_list, metadata_list, distance_list = list(), list(), list(), list()
        if ids_to_exclude is None:
            ids_to_exclude = []
        for batch in zip(ids, contents, metadatas, distances):
            records = zip(batch[0], batch[1], batch[2], batch[3])
            for rec_id, content, metadata, distance in records:
                if rec_id in ids_list or rec_id in ids_to_exclude:
                    continue
                ids_list.append(rec_id)
                content_list.append(content)
                metadata_list.append(metadata)
                distance_list.append(distance)
        return ids_list, content_list, metadata_list, distance_list

    @staticmethod
    def format_sources(sources: list[AnswerSource]):
        if not sources:
            return ""
        formated_sources =set(f"{source.source.replace(os.environ['DATA_PATH'], '')[1:]} page {source.page}" for source in sources)
        return "\n".join(["Sources:", "\n".join(formated_sources)])

    @staticmethod
    def format_data(ids: list[str], contents: list[str], metadatas: list[dict]) -> str:
        formated_data = {}
        if len(ids) != len(contents) or len(ids) != len(metadatas):
            raise ValueError(f"ids({len(ids)}), contents({len(contents)}) and metadatas({len(metadatas)}) must contain same number of elements.")
        for rec_id, content, metadata in zip(ids, contents, metadatas):
            clear_metadata = {"source": metadata["source"], "page": metadata["page"]}
            formated_data[rec_id] = {"content": content, "metadata": clear_metadata}
        return json.dumps(formated_data)

    @staticmethod
    def normalize_vector_scores(distances: list[float]) -> list[float]:
        sims = [1 / (1 + d) for d in distances]  # why: invert distance

        max_sim = max(sims) if sims else 1.0
        return [s / max_sim for s in sims]

    @staticmethod
    def init_query_agent():
        return QueryAgent(name="QueryAgent", description="Optimizer of RAG queries")

    @staticmethod
    def init_answer_agent():
        return AnswerAgent(
            name="AnswerAgent",
            description="AI assistant",
            system_prompt_version="0.1.2"
        )

    @staticmethod
    def init_chunk_ranker_agent():
        return ChunkRankerAgent(
            name="ChunkRankerAgent",
            description="Chunk relevance ranker",
            system_prompt_version="0.1.2"
        )

    @staticmethod
    def init_answer_critic_agent():
        return AnswerCriticAgent(
            name="AnswerCriticAgent",
            description="Answer ranker"
        )