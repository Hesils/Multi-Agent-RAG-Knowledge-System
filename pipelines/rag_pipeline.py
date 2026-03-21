import os
import json

import chromadb
from langchain.agents.structured_output import StructuredOutputValidationError
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agents.ChunkContextualizerAgent import ChunkContextualizerAgent
from utils.chromadb_client import ChromadbClient


class RagPipeline:
    # TODO Adding a "Document" folder recursive RAG loader and adding a new collection by subdirectory
    # TODO Implementing list_collection and smarter id generation
    # TODO Do AnswerPipeline make the collection choice

    def __init__(self):
        self.pdf_loader = PyPDFDirectoryLoader(os.environ["DATA_PATH"])
        self.chunk_contextualizer_agent = self.init_contextualizer_agent()

    def update_db_collection_content(self, collection: chromadb.Collection, chroma_client: ChromadbClient):
        raw_documents = self.pdf_loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=100,
            length_function=len,
            is_separator_regex=False,
        )
        chunks = text_splitter.split_documents(raw_documents)

        # preparing to be added in chromadb
        documents = []
        metadata = []
        ids = []
        i = 0
        for chunk in chunks:
            context_pages_content = self.get_chunk_around_pages(chunk.metadata["source"], chunk.metadata["page"], raw_documents)
            call_try = 0
            while call_try < 5:
                try:
                    contextualized_content = self.chunk_contextualizer_agent.execute(json.dumps({
                        "chunk": chunk.page_content,
                        "around_pages_content": "\n".join([page.page_content for page in context_pages_content])
                    }), role="user")
                    documents.append(contextualized_content.content)
                    break
                except StructuredOutputValidationError as e:
                    print(f"Structured Agent Output error : {e}")
                    call_try += 1
            ids.append("ID"+str(i))
            metadata.append(chunk.metadata)
            i += 1

        # adding/update to chromadb
        chroma_client.db_upsert(
            collection=collection,
            documents=documents,
            metadatas=metadata,
            ids=ids
        )


    @staticmethod
    def get_chunk_around_pages(chunk_source_doc: str, chunk_page: int, documents_pages: list) -> list:
        context_pages_content = []
        for page in documents_pages:
            if page.metadata["source"] == chunk_source_doc and chunk_page - 1 <= page.metadata["page"] <= chunk_page + 1:
                context_pages_content.append(page)
        return context_pages_content


    @staticmethod
    def build_data(query_results: chromadb.QueryResult) -> str:
        data_dict = {}
        batches = zip(query_results["ids"], query_results["documents"], query_results["metadatas"])
        for batch in batches:
            records = zip(batch[0], batch[1], batch[2])
            for chunk_id, document, metadata in records:
                data_dict[chunk_id] = {"content": document, "metadata": metadata}
        return json.dumps(data_dict)

    @staticmethod
    def init_contextualizer_agent():
        agent = ChunkContextualizerAgent(
            name="ChunkContextualizerAgent",
            description="Agent that enriches doncument chunks"
        )
        return agent
