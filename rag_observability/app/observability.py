from prometheus_client import Counter, Histogram, Gauge

# -------------- RAG --------------

# Counters
FILES_PROCESSED = Counter(
    "files_processed_total",
    "Total number of processed files",
    ["status", "file_type"]
)

CHUNKS_CREATED = Counter(
    "chunks_created_total",
    "Total number of chunks created"
)

ERRORS = Counter(
    "pipeline_errors_total",
    "Total number of errors",
    ["type"]
)

IGNORED_FILES = Counter(
    "ignored_files_total",
    "Total number of files ignored",
    ["file_type"]
)


# Histograms (timing)
PROCESSING_TIME = Histogram(
    "file_processing_seconds",
    "Time spent processing a file"
)

DB_UPSERT_TIME = Histogram(
    "db_upsert_time_seconds",
    "Time spent upserting chunks in db",
    ["collection"]
)

FILE_SIZE = Histogram(
    "file_size_bytes",
    "File size"
)

CHUNKS_PER_UPSERT = Histogram(
    "chunks_per_upsert",
    "Number of chunks per DB upsert"
)

# Gauge


FILES_IN_QUEUE = Gauge(
    "files_in_queue",
    "Files waiting processing"
)

# -------------- LLMs --------------

# Counters
REQUEST_COUNT = Counter(
    "rag_requests_total",
    "Total number of user queries"
)

EMPTY_RESPONSE = Counter(
    "empty_responses_total",
    "Number of empty answers"
)
TOKENS_USED = Counter(
    "tokens_used_total",
    "Total tokens used",
    ["type"]  # prompt / completion
)
AGENT_CALLS = Counter(
    "agent_calls_total",
    "Number of calls per agent",
    ["agent_name"]
)
# Histograms
REQUEST_LATENCY = Histogram(
    "rag_request_duration_seconds",
    "Time to answer a query"
)
RETRIEVAL_TIME = Histogram("retrieval_time_seconds", "Retrieval time")
LLM_TIME = Histogram("llm_time_seconds", "LLM response time")
RERANK_TIME = Histogram("rerank_time_seconds", "Reranking time")
DOCUMENTS_RETRIEVED = Histogram(
    "documents_retrieved",
    "Number of retrieved documents"
)

# Gauges