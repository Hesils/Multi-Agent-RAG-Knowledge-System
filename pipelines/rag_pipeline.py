import time
from pathlib import Path
from typing import Union
from concurrent.futures import ThreadPoolExecutor, as_completed

import chromadb
from langchain_core.documents import Document

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
            return
        existing_hash_map, existing_hashes = self._get_existing_chunks(
            doc_path, collection, chroma_client
        )
        raw_doc = loader.load(doc_path)
        tasks, new_hashes = self._collect_new_tasks(
            raw_doc, chunk_manager, existing_hashes
        )
        documents, metadata, ids = self._execute_parallel(tasks, chunk_manager)
        ids_to_delete = self._compute_deletions(
            existing_hash_map, existing_hashes, new_hashes
        )
        self._apply_changes(
            collection,
            chroma_client,
            documents,
            metadata,
            ids,
            ids_to_delete
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

    def insert_doc(self, doc_path: str, collection: chromadb.Collection, chroma_client: ChromadbClient):
        loader = self.chose_loader(doc_path)
        chunk_manager = self.chose_chunker(doc_path)

        if not loader or not chunk_manager:
            return

        raw_doc = loader.load(doc_path)

        tasks = self._collect_insert_tasks(raw_doc, chunk_manager)

        documents, metadata, ids = self._execute_parallel(tasks, chunk_manager)

        self._apply_insert(
            collection,
            chroma_client,
            documents,
            metadata,
            ids
        )

    @staticmethod
    def update_doc_path(src_path: str, new_path: str, collection: chromadb.Collection, chroma_client: ChromadbClient):
        result = chroma_client.get_chunks_where(["source"], [src_path], collection)
        chunks = list(zip(
            result["ids"],
            result["metadatas"],
            result["documents"]
        ))
        ids = []
        metadatas = []
        documents = []

        for cid, metadata, doc in chunks:
            metadata["source"] = new_path

            ids.append(cid)
            metadatas.append(metadata)
            documents.append(doc)
        if ids:
            chroma_client.db_upsert(
                collection=collection,
                ids=ids,
                metadatas=metadatas,
                documents=documents
            )

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

    @staticmethod
    def _get_existing_chunks(doc_path, collection, chroma_client):
        result = chroma_client.get_chunks_where(["source"], [doc_path], collection)

        chunks = [
            Document(id=cid, metadata=meta, page_content=doc)
            for cid, meta, doc in zip(
                result["ids"], result["metadatas"], result["documents"]
            )
        ]

        hash_map = {
            c.metadata["chunk_hash"]: c
            for c in chunks
            if "chunk_hash" in c.metadata
        }

        return hash_map, set(hash_map.keys())

    @staticmethod
    def _collect_new_tasks(raw_doc, chunk_manager, existing_hashes):
        tasks = []
        new_hashes = set()

        for doc in raw_doc:
            chunks = chunk_manager.document_chunking([doc])

            for chunk in chunks:
                chunk_hash = chunk.metadata.get("chunk_hash")
                if not chunk_hash:
                    continue

                new_hashes.add(chunk_hash)

                if chunk_hash in existing_hashes:
                    continue

                tasks.append((chunk, chunks))

        return tasks, new_hashes

    def _process_chunk(self, chunk, chunks, chunk_manager):
        context_pages = chunk_manager.get_chunk_around_pages(
            chunk.metadata["source"],
            chunk.metadata["page"],
            chunks
        )

        content = chunk_manager.contextualize_chunk(
            chunk,
            context_pages,
            self.chunk_contextualizer_agent
        )
        # To avoid rate limit reach
        time.sleep(30)
        return {
            "content": content,
            "metadata": chunk.metadata,
            "id": chunk.metadata["uuid"]
        }

    def _execute_parallel(self, tasks, chunk_manager):
        documents, metadata, ids = [], [], []

        if not tasks:
            return documents, metadata, ids

        max_workers = min(8, len(tasks))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._process_chunk, chunk, chunks, chunk_manager)
                for chunk, chunks in tasks
            ]

            for future in as_completed(futures):
                result = future.result()
                documents.append(result["content"])
                metadata.append(result["metadata"])
                ids.append(result["id"])

        return documents, metadata, ids

    @staticmethod
    def _compute_deletions(existing_hash_map, existing_hashes, new_hashes):
        hashes_to_delete = existing_hashes - new_hashes
        return [existing_hash_map[h].id for h in hashes_to_delete]

    @staticmethod
    def _apply_changes(collection: chromadb.Collection, chroma_client: ChromadbClient, documents, metadata, ids, ids_to_delete):
        if ids_to_delete:
            chroma_client.db_delete_with_id(
                collection=collection,
                chunks_id=ids_to_delete
            )

        if documents:
            chroma_client.db_upsert(
                collection=collection,
                documents=documents,
                metadatas=metadata,
                ids=ids
            )

    @staticmethod
    def delete_doc(doc_path: str, collection: chromadb.Collection, chroma_client: ChromadbClient):
        get_result = chroma_client.get_chunks_where(
            ["source"],
            [doc_path],
            collection,
        )
        if not get_result["ids"]:
            return
        chroma_client.db_delete_with_id(get_result["ids"], collection)

    @staticmethod
    def _collect_insert_tasks(raw_doc, chunk_manager):
        tasks = []

        for doc in raw_doc:
            chunks = chunk_manager.document_chunking([doc])
            for chunk in chunks:
                if "chunk_hash" not in chunk.metadata:
                    continue
                tasks.append((chunk, chunks))
        return tasks

    @staticmethod
    def _apply_insert(collection, chroma_client, documents, metadata, ids):
        if not documents:
            return

        chroma_client.db_upsert(
            collection=collection,
            documents=documents,
            metadatas=metadata,
            ids=ids
        )