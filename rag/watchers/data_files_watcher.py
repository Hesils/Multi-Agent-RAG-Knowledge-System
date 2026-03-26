import os
import time
import threading
import hashlib

from watchdog.events import FileSystemEvent, FileSystemEventHandler, FileSystemMovedEvent, FileDeletedEvent, FileModifiedEvent, FileCreatedEvent, FileMovedEvent
from watchdog.observers import Observer

from pipelines.rag_pipeline import RagPipeline
from utils.chromadb_client import chromadb_client

from utils.metrics import metrics

DELETE_TIMEOUT = 2.0  # temps avant considérer un delete comme réel
DEBOUNCE_SECONDS = 0.5  # debounce pour les bursts


class DataFilesEventHandler(FileSystemEventHandler):
    def __init__(self):
        super().__init__()
        self.rag_pipeline = RagPipeline()
        self.collection = chromadb_client.chroma_client.get_or_create_collection(name="identity")

        self.event_buffer = []
        self.lock = threading.Lock()
        self.hash_cache = {}  # path -> hash
        self.deleted_events_buffer = []  # [(path, hash, timestamp)]

        # Thread pour traiter les events
        self.worker = threading.Thread(target=self._process_events_loop, daemon=True)
        self.worker.start()
        metrics.add("queue", {"action": "set","size": 0})

    # -------------------- FS EVENT --------------------
    def on_any_event(self, event: FileSystemEvent):
        with self.lock:
            metrics.add("queue", {"action": "inc","size": 1})
            self.event_buffer.append({"event": event, "time": time.time()})

    # -------------------- MAIN LOOP --------------------
    def _process_events_loop(self):
        while True:
            time.sleep(DEBOUNCE_SECONDS)
            self._flush_events()
            self._flush_pending_deletes()

    # -------------------- FLUSH EVENTS --------------------
    def _flush_events(self):
        with self.lock:
            events = self.event_buffer
            self.event_buffer = []

        for e in events:
            self._handle_event(e["event"])

    # -------------------- FLUSH VRAIS DELETES --------------------
    def _flush_pending_deletes(self):
        now = time.time()
        to_delete = [e for e in self.deleted_events_buffer if now - e[2] >= DELETE_TIMEOUT]
        for path, _, _ in to_delete:
            self.rag_pipeline.delete_doc(path, self.collection, chromadb_client)
            self.hash_cache.pop(path, None)
            metrics.add("queue", {"action": "set","size": 0})
        self.deleted_events_buffer = [e for e in self.deleted_events_buffer if now - e[2] < DELETE_TIMEOUT]

    # -------------------- HANDLE SINGLE EVENT --------------------
    def _handle_event(self, event: FileSystemEvent):
        path = getattr(event, "src_path", None)
        if not path:
            return

        if event.event_type == "deleted":
            file_hash = self.hash_cache.get(path)
            self.deleted_events_buffer.append((path, file_hash, time.time()))

        elif event.event_type == "created":
            # try to match a buffered delete (move/rename)
            match = self._find_delete_match(path)
            if match:
                del_path, _, _ = match
                self.rag_pipeline.update_doc_path(del_path, path, self.collection, chromadb_client)
                self.deleted_events_buffer.remove(match)
                self.hash_cache[path] = self.hash_cache.get(del_path)  # move hash to new path
                self.hash_cache.pop(del_path, None)
                metrics.add("queue", {"action": "dec","size": 2})
            else:
                self.rag_pipeline.insert_doc(path, self.collection, chromadb_client)
                self.hash_cache[path] = self._compute_file_hash(path)
                metrics.add("queue", {"action": "dec","size": 1})

        elif event.event_type == "modified":
            self.rag_pipeline.update_doc(path, self.collection, chromadb_client)
            self.hash_cache[path] = self._compute_file_hash(path)
            metrics.add("queue", {"action": "dec","size": 1})
        elif event.event_type == "moved":
            self.rag_pipeline.update_doc_path(path, event.dest_path, self.collection, chromadb_client)
            self.hash_cache[event.dest_path] = self.hash_cache[path]
            self.hash_cache.pop(path, None)
            metrics.add("queue", {"action": "dec","size": 1})

    # -------------------- MATCH DELETE --------------------
    def _find_delete_match(self, create_path: str):
        now = time.time()
        create_hash = self._compute_file_hash(create_path)

        best_match = None

        for entry in self.deleted_events_buffer:
            del_path, del_hash, ts = entry
            if now - ts > DELETE_TIMEOUT:
                continue

            # hash match
            if create_hash and del_hash and create_hash == del_hash:
                return entry

            # filename match
            if os.path.basename(del_path) == os.path.basename(create_path):
                best_match = entry

            # fallback: time proximity
            if not best_match:
                best_match = entry

        return best_match

    # -------------------- COMPUTE HASH --------------------
    @staticmethod
    def _compute_file_hash(path):
        if not path or not os.path.exists(path):
            return None
        try:
            hasher = hashlib.md5()
            with open(path, "rb") as f:
                while chunk := f.read(8192):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception:
            return None


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
    print("Watching")
    try:
        while True:
            time.sleep(10)
    finally:
        observer.stop()
        observer.join()