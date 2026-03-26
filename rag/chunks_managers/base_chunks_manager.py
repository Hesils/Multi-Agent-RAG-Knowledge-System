import hashlib
import json
from abc import ABC
from pathlib import Path

from langchain_core.documents import Document
from langchain.agents.structured_output import StructuredOutputValidationError
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agents.base_agent import BaseAgent
from utils.trace_client import TraceClient


class BaseChunksManager(ABC):
    def __init__(self, file_type: str):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=100,
            length_function=len,
            is_separator_regex=False,
        )
        self.file_type = file_type

    def document_chunking(self, documents: list[Document]) -> list[Document]:
        chunks = []
        docs_full_content = {}
        for document in documents:
            if "source" not in document.metadata:
                raise ValueError(f"Source not present in metadata.")
            if "page" not in document.metadata:
                raise ValueError(f"Page not present in metadata.")
            doc_path = Path(document.metadata["source"])
            new_metadata = {
                "source": str(doc_path),
                "folder": doc_path.parent.name,
                "type": self.file_type,
                "page": document.metadata["page"]
            }
            if doc_path in docs_full_content:
                docs_full_content[doc_path] += document.page_content
            else:
                docs_full_content[doc_path] = document.page_content
            chunks += self.text_splitter.create_documents([document.page_content], [new_metadata])
        for i, chunk in enumerate(chunks):
            chunk.metadata["file_hash"] = hashlib.md5(docs_full_content[Path(chunk.metadata["source"])].encode()).hexdigest()
            chunk.metadata["chunk_hash"] = hashlib.md5(chunk.page_content.encode()).hexdigest()
            chunk.metadata["uuid"] = hashlib.md5(f"{chunk.metadata['source']}_{i}_{chunk.metadata['file_hash']}".encode()).hexdigest()
        return chunks

    @staticmethod
    def contextualize_chunk(chunk: Document, context_pages_content: list[Document], contextualizer: BaseAgent, trace_client: TraceClient) -> str:
        call_try = 0
        while call_try < 5:
            try:
                contextualized_content = contextualizer.execute(json.dumps({
                    "chunk": chunk.page_content,
                    "around_pages_content": "\n".join([page.page_content for page in context_pages_content])
                }), role="user", trace_client=trace_client)
                return contextualized_content.content
            except StructuredOutputValidationError as e:
                print(f"Structured Agent Output error : {e}")
                call_try += 1
        raise ValueError("Structured Agent Output error")

    @staticmethod
    def get_chunk_around_pages(chunk_source_doc: str, chunk_page: int, documents_pages: list) -> list:
        context_pages_content = []
        for page in documents_pages:
            if page.metadata["source"] == chunk_source_doc and chunk_page - 1 <= page.metadata["page"] <= chunk_page + 1:
                context_pages_content.append(page)
        return context_pages_content