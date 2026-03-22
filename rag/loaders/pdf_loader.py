from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from rag.loaders.base_loader import BaseLoader

class PdfLoader(BaseLoader):
    def __init__(self):
        super().__init__("pdf")

    def load(self, file_path: str) -> list[Document]:
        loader = PyPDFLoader(file_path)
        raw_document = loader.load()
        return raw_document

    def directory_load(self, dir_path: str, recursive: bool = False) -> dict[str, Document]:
        raw_documents = {}
        dir_path = Path(dir_path)
        for file in dir_path.glob("*"):
            if recursive and file.is_dir():
                raw_documents.update(self.directory_load(str(file), recursive))
            elif file.suffix == self.file_type:
                raw_documents[str(file)] = self.load(str(file))
        return raw_documents
