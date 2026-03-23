from abc import ABC, abstractmethod

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter



class BaseLoader(ABC):
    def __init__(self, file_type: str):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=100,
            length_function=len,
            is_separator_regex=False,
        )
        self.file_type = file_type

    @abstractmethod
    def load(self, file_path: str) -> list[Document]:
        ...

    @abstractmethod
    def directory_load(self, dir_path: str) -> dict[str, Document]:
        ...