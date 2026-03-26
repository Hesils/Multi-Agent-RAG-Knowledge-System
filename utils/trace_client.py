import requests

class TraceClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.trace_id = None

    def start(self, pipeline: str, query: str):
        res = requests.post(f"{self.base_url}/trace/start", json={
            "pipeline": pipeline,
            "query": query
        })
        self.trace_id = res.json()["trace_id"]
        return self.trace_id

    def step(self, step, data=None, duration=None):
        if not self.trace_id:
            return
        requests.post(f"{self.base_url}/trace/step", json={
            "trace_id": self.trace_id,
            "step": step,
            "data": data or {},
            "duration": duration
        })

    def end(self, response):
        if not self.trace_id:
            return
        requests.post(f"{self.base_url}/trace/end", params={
            "trace_id": self.trace_id,
            "response": response[:1000]
        })
