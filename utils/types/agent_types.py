from pydantic import BaseModel


class AgentInput(BaseModel):
    messages: list[dict]