from typing import Union, Optional, Any
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from utils.prompt_utils import get_prompt
from utils.types.agent_types import AgentInput


class BaseAgent:
    def __init__(
            self,
            name: str,
            description: str,
            model: ChatOpenAI,
            tools: Optional[list|None] = None,
            system_prompt_version: Union[str | None] = "0.1.0",
            output_type: type[Any] = None
    ):
        if tools is None:
            tools = []
        self.model: ChatOpenAI = model
        self.name = name
        self.description = description
        self.system_prompt = get_prompt(self.name, system_prompt_version)
        self.agent = create_agent(
            name=self.name,
            model=model,
            system_prompt=self.system_prompt,
            tools=tools,
            response_format=output_type
        )
        self.response_history = []

    def execute(self, request: str):
        print(f"Lancement de l'agent {self.name}")
        agent_input = AgentInput(
            messages = [
                {"role":"system", "content":self.system_prompt},
                {"role":"user","content":request}
            ]
        )
        self.response_history.append(self.agent.invoke(agent_input))
        return self.response_history[-1]["structured_response"]