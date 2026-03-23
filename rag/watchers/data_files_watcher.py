import os
import time

from watchdog.events import FileSystemEvent, FileSystemEventHandler, FileSystemMovedEvent, FileDeletedEvent, FileModifiedEvent, FileCreatedEvent, FileMovedEvent
from watchdog.observers import Observer

class DataFilesEventHandler(FileSystemEventHandler):
    def on_any_event(self, event: FileSystemEvent) -> None:
        if event.event_type == "moved":
            ...
        elif event.event_type == "deleted":
            ...
        elif event.event_type == "created":
            ...
        elif event.event_type == "modified":
            ...
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
    try:
        while True:
            # Watch toutes les 10 minutes
            time.sleep(600)
    finally:
        observer.stop()
        observer.join()