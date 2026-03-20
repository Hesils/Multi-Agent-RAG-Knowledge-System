import os
import json

import chromadb
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class RagPipeline:
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(os.environ["CHROMADB_PATH"])
        self.pdf_loader = PyPDFDirectoryLoader(os.environ["DATA_PATH"])

    def update_db_collection_content(self, collection: chromadb.Collection):
        raw_documents = self.pdf_loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=100,
            length_function=len,
            is_separator_regex=False,
        )
        chunks = text_splitter.split_documents(raw_documents)

        # preparing to be added in chromadb
        documents = []
        metadata = []
        ids = []
        i = 0
        for chunk in chunks:
            documents.append(chunk.page_content)
            ids.append("ID"+str(i))
            metadata.append(chunk.metadata)
            i += 1

        # adding/update to chromadb
        collection.upsert(
            documents=documents,
            metadatas=metadata,
            ids=ids
        )

    @staticmethod
    def get_chunks_for_collection(collection: chromadb.Collection, requests: list[str]) -> chromadb.QueryResult:
        results = collection.query(
            query_texts=requests,
            n_results=10
        )
        return results

    @staticmethod
    def build_data(query_results: chromadb.QueryResult) -> str:
        data_dict = {}
        batches = zip(query_results["ids"], query_results["documents"], query_results["metadatas"])
        for batch in batches:
            records = zip(batch[0], batch[1], batch[2])
            for id, document, metadata in records:
                data_dict[id] = {"content": document, "metadata": metadata}
        return json.dumps(data_dict)
