import os
import time

from watchdog.events import FileSystemEvent, FileSystemEventHandler, FileSystemMovedEvent, FileDeletedEvent, FileModifiedEvent, FileCreatedEvent, FileMovedEvent
from watchdog.observers import Observer

from pipelines.rag_pipeline import RagPipeline
from utils.chromadb_client import chromadb_client


class DataFilesEventHandler(FileSystemEventHandler):
    def __init__(self):
        super().__init__()
        self.rag_pipeline = RagPipeline()
        self.collection = chromadb_client.chroma_client.get_or_create_collection(name="identity")

    def on_any_event(self, event: FileSystemEvent) -> None:
        if event.event_type == "moved":
            self.rag_pipeline.update_doc_path(event.src_path, event.dest_path, self.collection, chromadb_client)
        elif event.event_type == "deleted":
            self.rag_pipeline.delete_doc(event.src_path, self.collection, chromadb_client)
        elif event.event_type == "created":
            self.rag_pipeline.insert_doc(event.src_path, self.collection, chromadb_client)
        elif event.event_type == "modified":
            self.rag_pipeline.update_doc(event.src_path, self.collection, chromadb_client)
        print(event)


def watch_datafiles():
    event_handler = DataFilesEventHandler()
    observer = Observer()
    observer.schedule(
        event_handler,
        os.environ["DATA_PATH"],
        recursive=True,
        event_filter=[FileMovedEvent, FileSystemMovedEvent, FileDeletedEvent, FileModifiedEvent, FileCreatedEvent]
    )
    observer.start()
    try:
        while True:
            print("Watching")
            time.sleep(10)
    finally:
        observer.stop()
        observer.join()