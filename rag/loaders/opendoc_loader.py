from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET

from langchain_core.documents import Document
from rag.loaders.base_loader import BaseLoader


ODT_NAMESPACE = {
    "text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0"
}


class OpenDocLoader(BaseLoader):
    def __init__(self):
        super().__init__(".odt")

    @staticmethod
    def _extract_text(file_path: str) -> list[str]:
        with ZipFile(file_path, "r") as z:
            with z.open("content.xml") as f:
                tree = ET.parse(f)

        root = tree.getroot()

        paragraphs = []
        for p in root.findall(".//text:p", ODT_NAMESPACE):
            text = "".join(p.itertext()).strip()
            if text:
                paragraphs.append(text)

        return paragraphs

    @staticmethod
    def _split_into_pages(paragraphs: list[str], max_chars: int = 1000) -> list[str]:
        pages = []
        current_page = []

        current_len = 0

        for para in paragraphs:
            if current_len + len(para) > max_chars and current_page:
                pages.append("\n".join(current_page))
                current_page = []
                current_len = 0

            current_page.append(para)
            current_len += len(para)

        if current_page:
            pages.append("\n".join(current_page))

        return pages

    def load(self, file_path: str) -> list[Document]:
        paragraphs = self._extract_text(file_path)
        pages = self._split_into_pages(paragraphs)

        documents = []
        for i, page in enumerate(pages):
            documents.append(
                Document(
                    page_content=page,
                    metadata={
                        "source": file_path,
                        "page": i
                    }
                )
            )

        return documents

    def directory_load(self, dir_path: str, recursive: bool = False) -> dict[str, list[Document]]:
        raw_documents = {}
        dir_path = Path(dir_path)

        for file in dir_path.glob("*"):
            if recursive and file.is_dir():
                raw_documents.update(self.directory_load(str(file), recursive))
            elif file.suffix == self.file_type:
                raw_documents[str(file)] = self.load(str(file))

        return raw_documents