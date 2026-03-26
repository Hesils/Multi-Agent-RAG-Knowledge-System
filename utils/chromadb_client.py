import json
import os

import chromadb
from chromadb import Metadata
from chromadb.api.types import OneOrMany, Document, ID

from utils.metrics import metrics


class ChromadbClient:
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(os.environ["CHROMADB_PATH"])

    def list_collections(self) -> list[str]:
        return [collection.name for collection in self.chroma_client.list_collections()]

    @staticmethod
    def db_delete_with_id(chunks_id: list[str], collection: chromadb.Collection):
        collection.delete(chunks_id)

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
            metrics.add("error", {"type": "DB Upsert Error"})
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
    def get_chunks_where(
            metadata_keys: list[str],
            metadata_values: list[str],
            collection: chromadb.Collection,
    ) -> chromadb.GetResult:
        if len(metadata_keys) != len(metadata_values):
            metrics.add("error", {"type": "DB Get Error"})
            raise ValueError(f"metadata_keys({len(metadata_keys)}) must have same number of element than metadata_values({len(metadata_values)})")
        results = collection.get(
            where={key: value for key, value in zip(metadata_keys, metadata_values)},
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