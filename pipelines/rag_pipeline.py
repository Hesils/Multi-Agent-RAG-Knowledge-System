import os
import time
from pathlib import Path
from typing import Union
from concurrent.futures import ThreadPoolExecutor, as_completed

import chromadb
from langchain_core.documents import Document

from agents.ChunkContextualizerAgent import ChunkContextualizerAgent
from rag.loaders import (
    PdfLoader,
    OpenDocLoader,
    TextFileLoader,
    MarkdownLoader,
    BaseLoader,
    GDocsLoader
)
from rag.chunks_managers import (
    PdfChunksManager,
    OpenDocsChunksManager,
    TextChunksManager,
    MarkdownChunksManager,
    BaseChunksManager,
    GDocsChunksManager
)
from utils.chromadb_client import ChromadbClient
from utils.metrics import metrics
from utils.trace_client import TraceClient


class RagPipeline:
    def __init__(self):
        self.chunk_contextualizer_agent = self.init_contextualizer_agent()

    def update_doc(self, doc_path: str, collection: chromadb.Collection, chroma_client: ChromadbClient):
        trace_client = TraceClient()
        trace_id = trace_client.start(pipeline="rag_ingestion", query=f"Update {Path(doc_path).name}")
        print(f"trace_id: {trace_id}")
        trace_client.step("updated_file_detected", {"file": doc_path})
        process_start_time = time.time()
        loader = self.chose_loader(doc_path)
        chunk_manager = self.chose_chunker(doc_path)

        if not loader or not chunk_manager:
            metrics.add("ignored_file", {"file_type" : doc_path.split(".")[-1]})
            trace_client.step("Unsupported_file_type", {"file_type": doc_path.split(".")[-1]})
            trace_client.end("File Ignored")
            return

        existing_hash_map, existing_hashes = self._get_existing_chunks(
            doc_path, collection, chroma_client
        )
        load_start_time = time.time()
        raw_doc = loader.load(doc_path)
        tasks, new_hashes = self._collect_new_tasks(
            raw_doc, chunk_manager, existing_hashes
        )
        trace_client.step(
            "loading_document",
            {"nb_pages": len(raw_doc), "updated_chunks": len(new_hashes), "not_updated_chunks": len(existing_hashes) - len(new_hashes)},
            time.time() - load_start_time
        )
        contextualisation_start_time = time.time()
        documents, metadata, ids = self._execute_parallel(tasks, chunk_manager, trace_client)
        trace_client.step(
            "contextualize_chunks",
            {},
            time.time() - contextualisation_start_time
        )
        if documents:
            metrics.add("file_size", {"size" : sum([len(doc.page_content.encode("utf-8")) for doc in raw_doc ])})
            metrics.add("chunks_created", {"count" : len(documents)})
            metrics.add("file_processed", {"status" : "success", "file_type": chunk_manager.file_type})
        ids_to_delete = self._compute_deletions(
            existing_hash_map, existing_hashes, new_hashes
        )
        self._apply_changes(
            collection,
            chroma_client,
            documents,
            metadata,
            ids,
            ids_to_delete,
            trace_client
        )
        total_time = time.time() - process_start_time
        metrics.add("processing_time", {"duration" : total_time})
        trace_client.end("File updated")

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
        trace_client = TraceClient()
        trace_id = trace_client.start(pipeline="rag_ingestion", query=f"New document {Path(doc_path).name}")
        print(f"trace_id: {trace_id}")
        trace_client.step("new_file_detected", {"file": doc_path})
        process_start_time = time.time()

        loader = self.chose_loader(doc_path)
        chunk_manager = self.chose_chunker(doc_path)

        if not loader or not chunk_manager:
            metrics.add("ignored_file", {"file_type" : doc_path.split(".")[-1]})
            trace_client.step("Unsupported_file_type", {"file_type": doc_path.split(".")[-1]})
            trace_client.end("File Ignored")
            return

        load_start_time = time.time()
        raw_doc = loader.load(doc_path)

        tasks = self._collect_insert_tasks(raw_doc, chunk_manager)
        trace_client.step(
            "loading_document",
            {"nb_pages": len(raw_doc), "new_chunks": len(tasks)},
            time.time() - load_start_time
        )
        contextualisation_start_time = time.time()
        documents, metadata, ids = self._execute_parallel(tasks, chunk_manager, trace_client)
        trace_client.step(
            "contextualize_chunks",
            {},
            time.time() - contextualisation_start_time
        )
        if documents:
            metrics.add("file_size", {"size" : sum([len(doc.page_content.encode("utf-8")) for doc in raw_doc ])})
            metrics.add("chunks_created", {"count" : len(documents)})
            metrics.add("file_processed", {"status" : "success", "file_type": chunk_manager.file_type})

        self._apply_insert(
            collection,
            chroma_client,
            documents,
            metadata,
            ids,
            trace_client
        )
        metrics.add("processing_time", {"duration" : time.time() - process_start_time})
        trace_client.end("File inserted")


    @staticmethod
    def update_doc_path(src_path: str, new_path: str, collection: chromadb.Collection, chroma_client: ChromadbClient):
        trace_client = TraceClient()
        trace_id = trace_client.start(pipeline="rag_ingestion", query=f"Moved document {Path(src_path).name}")
        print(f"trace_id: {trace_id}")
        trace_client.step("file_moved_detected", {"src_file_path": src_path, "dest_file_path": new_path})
        fetch_chunks_start_time = time.time()
        result = chroma_client.get_chunks_where(["source"], [src_path], collection)
        chunks = list(zip(
            result["ids"],
            result["metadatas"],
            result["documents"]
        ))
        trace_client.step("fetching_chunks", {"nb_chunks": len(chunks)}, time.time() - fetch_chunks_start_time)

        ids = []
        metadatas = []
        documents = []

        for cid, metadata, doc in chunks:
            metadata["source"] = new_path

            ids.append(cid)
            metadatas.append(metadata)
            documents.append(doc)
        if ids:
            upsert_start_time = time.time()
            chroma_client.db_upsert(
                collection=collection,
                ids=ids,
                metadatas=metadatas,
                documents=documents
            )
            trace_client.step("upserting_chunks", {"nb_chunks": len(chunks)}, time.time() - upsert_start_time)
        trace_client.end("Doc path updated")


    @staticmethod
    def chose_loader(file_path: str) -> Union[BaseLoader,None]:
        ext = Path(file_path).suffix
        if ext == ".pdf":
            return PdfLoader()
        elif ext == ".md":
            return MarkdownLoader()
        elif ext == ".txt":
            return TextFileLoader()
        elif ext == ".odt":
            return OpenDocLoader()
        # elif ext == ".gdoc" and "GDRIVE_CRED_PATH" in os.environ:
        #     return GDocsLoader(os.environ["GDRIVE_CRED_PATH"])
        return None

    @staticmethod
    def chose_chunker(file_path: str) -> Union[BaseChunksManager,None]:
        ext = Path(file_path).suffix
        if ext == ".pdf":
            return PdfChunksManager()
        elif ext == ".md":
            return MarkdownChunksManager()
        elif ext == ".txt":
            return TextChunksManager()
        elif ext == ".odt":
            return OpenDocsChunksManager()
        # elif ext == ".gdoc" and "GDRIVE_CRED_PATH" in os.environ:
        #     return GDocsChunksManager()
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

    def _process_chunk(self, chunk, chunks, chunk_manager, trace_client: TraceClient):
        context_pages = chunk_manager.get_chunk_around_pages(
            chunk.metadata["source"],
            chunk.metadata["page"],
            chunks
        )

        content = chunk_manager.contextualize_chunk(
            chunk,
            context_pages,
            self.chunk_contextualizer_agent,
            trace_client
        )
        # To avoid rate limit reach
        time.sleep(30)
        return {
            "content": content,
            "metadata": chunk.metadata,
            "id": chunk.metadata["uuid"]
        }

    def _execute_parallel(self, tasks, chunk_manager: BaseChunksManager, trace_client: TraceClient):
        documents, metadata, ids = [], [], []

        if not tasks:
            return documents, metadata, ids

        max_workers = min(8, len(tasks))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._process_chunk, chunk, chunks, chunk_manager, trace_client)
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
    def _apply_changes(collection: chromadb.Collection, chroma_client: ChromadbClient, documents, metadata, ids, ids_to_delete, trace_client: TraceClient):
        if ids_to_delete:
            deletion_start_time = time.time()
            chroma_client.db_delete_with_id(
                collection=collection,
                chunks_id=ids_to_delete
            )
            trace_client.step(
                "deleting_old_chunks",
                {"nb_to_delete": len(ids_to_delete)},
                time.time() - deletion_start_time
            )


        if documents:
            upsert_start_time = time.time()
            chroma_client.db_upsert(
                collection=collection,
                documents=documents,
                metadatas=metadata,
                ids=ids
            )
            metrics.add("db_upsert", {"collection": "default","duration" : time.time() - upsert_start_time})
            trace_client.step(
                "upserting_chunks",
                {"nb_to_upsert": len(ids)},
                time.time() - upsert_start_time
            )

    @staticmethod
    def delete_doc(doc_path: str, collection: chromadb.Collection, chroma_client: ChromadbClient):
        trace_client = TraceClient()
        trace_id = trace_client.start(pipeline="rag_ingestion", query=f"Delete document {Path(doc_path).name}")
        print(f"trace_id: {trace_id}")
        trace_client.step("file_deleted_detected", {"file_path": doc_path})
        fetch_chunks_start_time = time.time()
        get_result = chroma_client.get_chunks_where(
            ["source"],
            [doc_path],
            collection,
        )

        if not get_result["ids"]:
            trace_client.step("Nothing_to_delete", {})
            trace_client.end("No Chunks to delete")
            return
        trace_client.step("fetching_chunks", {"nb_chunks": len(get_result["ids"])}, time.time() - fetch_chunks_start_time)
        delete_start_time = time.time()
        chroma_client.db_delete_with_id(get_result["ids"], collection)
        trace_client.step("deleting_chunks", {"nb_chunks": len(get_result["ids"])}, time.time() - delete_start_time)
        trace_client.end("Doc chunks deleted")

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
    def _apply_insert(collection, chroma_client, documents, metadata, ids, trace_client: TraceClient):
        if not documents:
            return
        insert_start_time = time.time()
        chroma_client.db_upsert(
            collection=collection,
            documents=documents,
            metadatas=metadata,
            ids=ids
        )
        metrics.add("db_insert", {"collection": "default","duration" : time.time() - insert_start_time})
        trace_client.step(
            "inserting_chunks",
            {"nb_to_insert": len(ids)},
            time.time() - insert_start_time
        )