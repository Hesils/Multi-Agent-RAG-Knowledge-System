from typing import Union, Optional

from langchain_openai import ChatOpenAI

from agents.base_agent import BaseAgent
from utils.types.agent_types import AgentInput


class AnswerAgent(BaseAgent):
    def __init__(self,
                 name: str,
                 description: str,
                 model: Optional[ChatOpenAI] = None,
                 system_prompt_version: Union[str | None] = "0.1.0"
                 ):
        super().__init__(
            name,
            description,
            model=model if model else ChatOpenAI(
                model="gpt-5-nano",
                temperature=0,
            ),
            system_prompt_version=system_prompt_version,
        )

    def execute(self, request: str, data: str = ""):
        print(f"Lancement de l'agent {self.name}")
        agent_input = AgentInput(
            messages = [
                {"role":"system", "content":self.system_prompt},
                {"role":"system", "content":f"There is the provided reference data: {data}"},
                {"role":"user","content":request}
            ]
        )
        self.response_history.append(self.agent.invoke(agent_input))
        return self.response_history[-1]["messages"][-1].content