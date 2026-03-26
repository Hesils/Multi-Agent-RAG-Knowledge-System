import uuid
from typing import Dict, List, Any

from fastapi import FastAPI
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
import time

from app.observability import (
    FILES_PROCESSED,
    CHUNKS_CREATED,
    CHUNKS_PER_UPSERT,
    ERRORS,
    PROCESSING_TIME,
    IGNORED_FILES,
    FILE_SIZE,
    FILES_IN_QUEUE,
    EMPTY_RESPONSE,
    DB_UPSERT_TIME,
    REQUEST_COUNT,
    REQUEST_LATENCY,
    RERANK_TIME,
    RETRIEVAL_TIME,
    DOCUMENTS_RETRIEVED,
    TOKENS_USED,
    AGENT_CALLS,
    LLM_TIME
)
from pydantic import BaseModel


app = FastAPI()
app.mount("/ui", StaticFiles(directory="app/static", html=True), name="static")

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# endpoint simulant ingestion
@app.post("/ingest_test")
def ingest_test():
    import random
    import time

    start = time.time()

    chunks = random.randint(1, 20)
    CHUNKS_CREATED.inc(chunks)
    FILES_PROCESSED.labels(status="success", file_type="pdf").inc()

    PROCESSING_TIME.observe(time.time() - start)

    return {"chunks": chunks}

# endpoint simulant answer
@app.post("/answer_test")
def answer_test():
    import time

    start = time.time()
    REQUEST_COUNT.inc()
    AGENT_CALLS.labels(agent_name="QueryOptimyzer").inc()
    LLM_TIME.observe(time.time() - start)
    TOKENS_USED.labels(type="input").inc(10)
    TOKENS_USED.labels(type="output").inc(15)
    RETRIEVAL_TIME.observe(time.time() - start)
    DOCUMENTS_RETRIEVED.observe(3)
    RERANK_TIME.observe(time.time() - start)
    AGENT_CALLS.labels(agent_name="AnswerAgent").inc()
    LLM_TIME.observe(time.time() - start)
    TOKENS_USED.labels(type="input").inc(10)
    TOKENS_USED.labels(type="output").inc(15)
    REQUEST_LATENCY.observe(start - time.time())

class Event(BaseModel):
    type: str
    data: Dict[str, Any]

class BatchPayload(BaseModel):
    events: List[Event]


@app.post("/metrics/batch")
def metrics_batch(payload: BatchPayload):
    for event in payload.events:
        t = event.type
        d = event.data

        try:
            # -------- RAG --------
            if t == "file_processed":
                FILES_PROCESSED.labels(
                    status=d["status"],
                    file_type=d["file_type"]
                ).inc()

            elif t == "chunks_created":
                CHUNKS_CREATED.inc(d["count"])

            elif t == "error":
                ERRORS.labels(type=d["type"]).inc()

            elif t == "ignored_file":
                IGNORED_FILES.labels(file_type=d["file_type"]).inc()

            elif t == "processing_time":
                PROCESSING_TIME.observe(d["duration"])

            elif t == "db_upsert":
                DB_UPSERT_TIME.labels(
                    collection=d["collection"]
                ).observe(d["duration"])

                if "nb_chunks" in d:
                    CHUNKS_PER_UPSERT.observe(d["nb_chunks"])

            elif t == "file_size":
                FILE_SIZE.observe(d["size"])

            elif t == "queue":
                if d["action"] == "set":
                    FILES_IN_QUEUE.set(d["size"])
                elif d["action"] == "inc":
                    FILES_IN_QUEUE.inc(d["size"])
                elif d["action"] == "dec":
                    FILES_IN_QUEUE.dec(d["size"])

            # -------- LLM --------
            elif t == "request":
                REQUEST_COUNT.inc()

            elif t == "request_latency":
                REQUEST_LATENCY.observe(d["duration"])

            elif t == "empty_response":
                EMPTY_RESPONSE.inc()

            elif t == "tokens":
                TOKENS_USED.labels(type=d["type"]).inc(d["count"])

            elif t == "agent_call":
                AGENT_CALLS.labels(agent_name=d["agent_name"]).inc()

            elif t == "retrieval_time":
                RETRIEVAL_TIME.observe(d["duration"])

            elif t == "llm_time":
                LLM_TIME.observe(d["duration"])

            elif t == "rerank_time":
                RERANK_TIME.observe(d["duration"])

            elif t == "documents":
                DOCUMENTS_RETRIEVED.observe(d["count"])

        except Exception as e:
            # optionnel : log erreur mais ne pas casser tout le batch
            print(f"Metric error: {t} -> {e}")

    return {"status": "ok", "events_processed": len(payload.events)}


# ------------ Trace Viewer ------------
traces: Dict[str, Dict] = {}

class StartTracePayload(BaseModel):
    pipeline: str
    query: str | None = None

@app.post("/trace/start")
def start_trace(payload: StartTracePayload):
    trace_id = str(uuid.uuid4())

    traces[trace_id] = {
        "pipeline": payload.pipeline,
        "query": payload.query,
        "steps": [],
        "start_time": time.time()
    }

    return {"trace_id": trace_id}

class StepPayload(BaseModel):
    trace_id: str
    step: str
    data: dict
    duration: float | None = None


@app.post("/trace/step")
def add_step(payload: StepPayload):
    trace = traces.get(payload.trace_id)

    if not trace:
        return {"error": "trace not found"}

    trace["steps"].append({
        "step": payload.step,
        "data": payload.data,
        "duration": payload.duration,
        "timestamp": time.time()
    })

    return {"status": "ok"}

@app.post("/trace/end")
def end_trace(trace_id: str, response: str):
    trace = traces.get(trace_id)

    trace["response"] = response
    trace["end_time"] = time.time()

    return {"status": "ok"}


# endpoint pour récupérer une trace
@app.get("/trace/{trace_id}")
def get_trace(trace_id: str):
    return traces.get(trace_id, {"error": "Trace not found"})

# endpoint pour lister toutes les traces
@app.get("/traces")
def list_traces(pipeline: str | None = None):
    results = []
    for k, v in traces.items():
        if pipeline and v.get("pipeline") != pipeline:
            continue

        results.append({
            "trace_id": k,
            "pipeline": v.get("pipeline"),
            "query": v.get("query"),
            "start_time": v.get("start_time")
        })
    return results