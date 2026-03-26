import requests
import threading
import time

class MetricsBuffer:
    def __init__(self, url, flush_interval=5, max_size=50):
        self.url = url
        self.buffer = []
        self.lock = threading.Lock()
        self.flush_interval = flush_interval
        self.max_size = max_size

        threading.Thread(target=self._auto_flush, daemon=True).start()

    def add(self, event_type, data):
        with self.lock:
            self.buffer.append({"type": event_type, "data": data})

            if len(self.buffer) >= self.max_size:
                self.flush()

    def flush(self):
        with self.lock:
            if not self.buffer:
                return

            batch = self.buffer
            self.buffer = []

        try:
            requests.post(self.url, json={"events": batch}, timeout=2)
        except Exception:
            print(f"Error metrics flush: {batch}")
            pass  # éviter de casser ta pipeline

    def _auto_flush(self):
        while True:
            time.sleep(self.flush_interval)
            self.flush()

metrics = MetricsBuffer("http://localhost:8000/metrics/batch")