from BM25 import BM25


def tokenize(text: str) -> list[str]:
    return text.lower().split()


def compute_bm25_scores(query: str, documents: list[str]) -> list[float]:
    tokenized_docs = [tokenize(doc) for doc in documents]
    bm25 = BM25(tokenized_docs)

    tokenized_query = tokenize(query)
    scores = bm25.simall(tokenized_query)

    if not scores:
        return [0.0] * len(documents)

    max_score = max(scores) or 1.0
    return [s / max_score for s in scores]  # why: normalize for hybrid