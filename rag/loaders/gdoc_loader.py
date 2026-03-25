
from pathlib import Path
import json
import re

from langchain_core.documents import Document
from rag.loaders.base_loader import BaseLoader

from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials


class GDocsLoader(BaseLoader):
    def __init__(self, credentials_path: str):
        super().__init__(".gdocs")

        scopes = ["https://www.googleapis.com/auth/documents.readonly"]
        creds = Credentials.from_service_account_file(credentials_path, scopes=scopes)
        self.service = build("docs", "v1", credentials=creds)

    # -------------------- HELPERS --------------------
    @staticmethod
    def _extract_doc_id(file_path: str) -> str:
        with open(file_path, "r") as f:
            data = json.load(f)

        url = data.get("url", "")
        match = re.search(r"/document/d/([a-zA-Z0-9-_]+)", url)
        if not match:
            raise ValueError(f"Invalid Google Docs URL in {file_path}")

        return match.group(1)

    def _fetch_document(self, doc_id: str) -> dict:
        return self.service.documents().get(documentId=doc_id).execute()

    @staticmethod
    def _extract_paragraphs(doc: dict) -> list[str]:
        content = doc.get("body", {}).get("content", [])

        paragraphs = []

        for element in content:
            if "paragraph" in element:
                texts = []
                for el in element["paragraph"].get("elements", []):
                    if "textRun" in el:
                        texts.append(el["textRun"].get("content", ""))
                text = "".join(texts).strip()
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

    # -------------------- MAIN --------------------

    def load(self, file_path: str) -> list[Document]:
        doc_id = self._extract_doc_id(file_path)
        doc = self._fetch_document(doc_id)

        paragraphs = self._extract_paragraphs(doc)
        pages = self._split_into_pages(paragraphs)

        documents = []
        for i, page in enumerate(pages):
            documents.append(
                Document(
                    page_content=page,
                    metadata={
                        "source": file_path,
                        "page": i,
                        "gdoc_id": doc_id
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