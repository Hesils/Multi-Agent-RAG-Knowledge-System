import json
import os

import chromadb
from chromadb import Metadata
from chromadb.api.types import OneOrMany, Document, ID


class ChromadbClient:
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(os.environ["CHROMADB_PATH"])


    @staticmethod
    def db_upsert(
            collection: chromadb.Collection,
            documents: OneOrMany[Document],
            metadatas: OneOrMany[Metadata],
            ids: OneOrMany[ID]
    ) -> bool:
        try:
            collection.upsert(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
        except Exception as e:
            print(f"Error while upserting in db: {e}")
            return False
        return True

    @staticmethod
    def get_chunks_for_collection(
            collection: chromadb.Collection,
            requests: list[str],
            n_results: int = 10
    ) -> chromadb.QueryResult:
        results = collection.query(
            query_texts=requests,
            n_results=n_results
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

chromadb_client = ChromadbClient()