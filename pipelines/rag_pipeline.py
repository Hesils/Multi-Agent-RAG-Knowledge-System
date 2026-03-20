import os
import json

import chromadb
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.chromadb_client import ChromadbClient


class RagPipeline:
    def __init__(self):
        self.pdf_loader = PyPDFDirectoryLoader(os.environ["DATA_PATH"])

    def update_db_collection_content(self, collection: chromadb.Collection, chroma_client: ChromadbClient):
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
        chroma_client.db_upsert(
            collection=collection,
            documents=documents,
            metadatas=metadata,
            ids=ids
        )



    @staticmethod
    def build_data(query_results: chromadb.QueryResult) -> str:
        data_dict = {}
        batches = zip(query_results["ids"], query_results["documents"], query_results["metadatas"])
        for batch in batches:
            records = zip(batch[0], batch[1], batch[2])
            for chunk_id, document, metadata in records:
                data_dict[chunk_id] = {"content": document, "metadata": metadata}
        return json.dumps(data_dict)
