from pathlib import Path
from typing import Union

import chromadb

from agents.ChunkContextualizerAgent import ChunkContextualizerAgent
from rag.loaders.base_loader import BaseLoader
from rag.loaders.pdf_loader import PdfLoader
from rag.chunks_managers.base_chunks_manager import BaseChunksManager
from rag.chunks_managers.pdf_chunks_manager import PdfChunksManager
from utils.chromadb_client import ChromadbClient


class RagPipeline:
    def __init__(self):
        self.chunk_contextualizer_agent = self.init_contextualizer_agent()

    def update_doc(self, doc_path: str, collection: chromadb.Collection, chroma_client: ChromadbClient):
        loader = self.chose_loader(doc_path)
        chunk_manager = self.chose_chunker(doc_path)
        if not loader or not chunk_manager:
            # Type de document non géré
            return
        documents = []
        metadata = []
        ids = []
        raw_doc = loader.load(doc_path)
        for doc in raw_doc:
            chunks = chunk_manager.document_chunking([doc])
            for chunk in chunks:
                context_chuck_pages = chunk_manager.get_chunk_around_pages(chunk.metadata["source"], chunk.metadata["page"], chunks)
                chunk.page_content = chunk_manager.contextualize_chunk(chunk, context_chuck_pages, self.chunk_contextualizer_agent)
                documents.append(chunk.page_content)
                metadata.append(chunk.metadata)
                ids.append(chunk.metadata["uuid"])
        chroma_client.db_upsert(
            collection=collection,
            documents=documents,
            metadatas=metadata,
            ids=ids
        )

    def update_docs(self, docs_path: list[str], collection: chromadb.Collection, chroma_client: ChromadbClient):
        for doc in docs_path:
            self.update_doc(doc, collection, chroma_client)

    def update_directory(self, dir_path: str, collection: chromadb.Collection, chroma_client: ChromadbClient):
        dir_path = Path(dir_path)
        for file in dir_path.glob("*"):
            if file.is_dir():
                self.update_directory(str(dir_path), collection, chroma_client)
            elif file.is_file():
                self.update_doc(str(file), collection, chroma_client)

    @staticmethod
    def chose_loader(file_path: str) -> Union[BaseLoader,None]:
        ext = Path(file_path).suffix
        if ext == ".pdf":
            return PdfLoader()
        return None

    @staticmethod
    def chose_chunker(file_path: str) -> Union[BaseChunksManager,None]:
        ext = Path(file_path).suffix
        if ext == ".pdf":
            return PdfChunksManager()
        return None

    @staticmethod
    def init_contextualizer_agent():
        agent = ChunkContextualizerAgent(
            name="ChunkContextualizerAgent",
            description="Agent that enriches doncument chunks"
        )
        return agent
