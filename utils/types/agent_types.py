from typing import Optional

from pydantic import BaseModel


class AgentInput(BaseModel):
    messages: list[dict]

class QueryAgentOutput(BaseModel):
    optimized_query: str
    sub_queries: list[str]

class ChunkRank(BaseModel):
    chunk_id: str
    relevance: float
    justification: str

class ChunkRankerAgentOutput(BaseModel):
    chunks_rank: list[ChunkRank]

class ChunkContextualizerOutput(BaseModel):
    content: str

class AnswerSource(BaseModel):
    chunk_id: str
    source: str
    page: int

class AnswerOutput(BaseModel):
    answer: str
    sources: list[AnswerSource]

class AnswerCriticAgentOutput(BaseModel):
    is_valid: bool
    confidence_score: float
    issues: Optional[list[str]]