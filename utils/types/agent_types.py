from pydantic import BaseModel


class AgentInput(BaseModel):
    messages: list[dict]

class QueryAgentOutput(BaseModel):
    optimized_query: str
    sub_queries: list[str]