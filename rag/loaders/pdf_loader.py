from collections import defaultdict
from pathlib import Path

from langchain_core.documents import Document
from rag.loaders.base_loader import BaseLoader

from docling.document_converter import DocumentConverter


class PdfLoader(BaseLoader):
    def __init__(self):
        super().__init__("pdf")
        self.converter = DocumentConverter()


    def load(self, file_path: str) -> list[Document]:
        result = self.converter.convert(file_path)
        doc = result.document

        # --- group texts by page ---
        pages_content = defaultdict(list)
        pages_metadata = defaultdict(lambda: {
            "labels": set(),
            "bboxes": []
        })

        for text_item in doc.texts:
            if not text_item.text:
                continue

            prov = text_item.prov[0] if text_item.prov else None
            page_no = prov.page_no if prov else 0

            pages_content[page_no].append(text_item.text)

            # enrich metadata
            pages_metadata[page_no]["labels"].add(text_item.label)

            if prov and prov.bbox:
                pages_metadata[page_no]["bboxes"].append({
                    "l": prov.bbox.l,
                    "t": prov.bbox.t,
                    "r": prov.bbox.r,
                    "b": prov.bbox.b
                })

        documents = []

        for page_no, texts in pages_content.items():
            content = "\n".join(texts).strip()

            if not content:
                continue  # why: avoid empty pages

            metadata = {
                "source": file_path,
                "page": page_no,
                "filename": doc.origin.filename,
                "filetype": doc.origin.mimetype,
                "doc_hash": doc.origin.binary_hash,
                "labels": list(pages_metadata[page_no]["labels"]),
                "bbox_count": len(pages_metadata[page_no]["bboxes"]),
                "page_width": doc.pages[str(page_no)].size.width if str(page_no) in doc.pages else None,
                "page_height": doc.pages[str(page_no)].size.height if str(page_no) in doc.pages else None
            }

            documents.append(
                Document(
                    page_content=content,
                    metadata=metadata
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