import time
from typing import Union, Optional
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from utils.prompt_utils import get_prompt
from utils.types.agent_types import AgentInput
from utils.metrics import metrics
from utils.trace_client import TraceClient

class BaseAgent:
    def __init__(
            self,
            name: str,
            description: str,
            model: ChatOpenAI,
            tools: Optional[list|None] = None,
            system_prompt_version: Union[str | None] = "0.1.0",
            output_type: type[BaseModel] = None
    ):
        if tools is None:
            tools = []
        self.structured_output = True if output_type else False
        self.model: ChatOpenAI = model
        self.name = name
        self.description = description
        self.system_prompt = get_prompt(self.name, system_prompt_version)
        self.agent = create_agent(
            name=self.name,
            model=model,
            system_prompt=self.system_prompt,
            tools=tools,
            response_format=output_type,
        )
        self.agent_input = AgentInput(
            messages = [
                {"role":"system", "content":self.system_prompt}
            ]
        )

    def execute(self, request: str, role: str, trace_client: Optional[TraceClient] = None):
        print(f"Lancement de l'agent {self.name}")
        self.agent_input.messages.append({
            "role": role,
            "content": request
        })
        metrics.add("agent_call", {"agent_name": self.name})
        agent_call_start_time = time.time()
        agent_response = self.agent.invoke(self.agent_input)
        llm_time = agent_call_start_time - time.time()
        metrics.add("llm_time", {"duration": llm_time})
        response_content = agent_response["messages"][-1].content if not self.structured_output else agent_response["structured_response"].model_dump_json()
        metrics.add("tokens", {"type": "input", "count": agent_response["messages"][-1].usage_metadata["input_tokens"]})
        metrics.add("tokens", {"type": "output", "count": agent_response["messages"][-1].usage_metadata["output_tokens"]})
        self.agent_input.messages.append({
            "role": "ai",
            "content": response_content
        })
        if trace_client:
            trace_client.step(f"{self.name} call", {
                "preview": self.agent_input.messages[-1]["content"][:200]
            }, duration=llm_time)
        return agent_response["messages"][-1].content if not self.structured_output else agent_response["structured_response"]