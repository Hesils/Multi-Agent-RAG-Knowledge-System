import os

import chromadb

from agents.ChunkContextualizerAgent import ChunkContextualizerAgent
from rag.loaders.base_loader import BaseLoader
from utils.chromadb_client import ChromadbClient


class RagPipeline:
    def __init__(self):
        self.chunk_contextualizer_agent = self.init_contextualizer_agent()

    def update_db_collection_content(self, collection: chromadb.Collection, chroma_client: ChromadbClient, loader: BaseLoader):
        # TODO faire une version qui n'update qu'un doc en VDB et une qui update plusieur docs
        documents = []
        metadata = []
        ids = []
        raw_documents = loader.directory_load(os.environ["DATA_PATH"])
        for doc_path, doc in raw_documents.items():
            chunks = loader.document_chunking([doc])
            for chunk in chunks:
                context_chuck_pages = loader.get_chunk_around_pages(chunk.metadata["source"], chunk.metadata["page"], chunks)
                chunk.page_content = loader.contextualize_chunk(chunk, context_chuck_pages, self.chunk_contextualizer_agent)
                documents.append(chunk.page_content)
                metadata.append(chunk.metadata)
                ids.append(chunk.metadata["uuid"])
        chroma_client.db_upsert(
            collection=collection,
            documents=documents,
            metadatas=metadata,
            ids=ids
        )

    @staticmethod
    def init_contextualizer_agent():
        agent = ChunkContextualizerAgent(
            name="ChunkContextualizerAgent",
            description="Agent that enriches doncument chunks"
        )
        return agent
