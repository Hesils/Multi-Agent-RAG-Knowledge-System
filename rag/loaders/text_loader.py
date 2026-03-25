from pathlib import Path
from abc import ABC, abstractmethod

from langchain_community import document_loaders as dl
from langchain_core.documents import Document

from rag.loaders.base_loader import BaseLoader

class BaseTextLoader(BaseLoader, ABC):
    def __init__(self, file_type: str):
        super().__init__(file_type)

    @abstractmethod
    def load(self, file_path: str) -> list[Document]:
        ...

    def directory_load(self, dir_path: str, recursive: bool = False) -> dict[str, Document]:
        raw_documents = {}
        dir_path = Path(dir_path)
        for file in dir_path.glob("*"):
            if recursive and file.is_dir():
                raw_documents.update(self.directory_load(str(file), recursive))
            elif file.suffix == self.file_type:
                raw_documents[str(file)] = self.load(str(file))
        return raw_documents

class TextFileLoader(BaseTextLoader):
    def __init__(self):
        super().__init__("txt")

    def load(self, file_path: str) -> list[Document]:
        loader = dl.TextLoader(file_path)
        raw_document = loader.load()
        for doc in raw_document:
            doc.metadata["page"] = 1
        return raw_document

class MarkdownLoader(BaseTextLoader):
    def __init__(self):
        super().__init__("md")

    def load(self, file_path: str) -> list[Document]:
        loader = dl.UnstructuredMarkdownLoader(file_path)
        raw_document = loader.load()
        return raw_document

