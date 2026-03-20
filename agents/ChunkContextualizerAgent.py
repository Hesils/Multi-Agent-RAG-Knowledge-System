from typing import Union, Optional

from langchain_openai import ChatOpenAI

from agents.base_agent import BaseAgent
from utils.types.agent_types import ChunkContextualizerOutput


class ChunkContextualizerAgent(BaseAgent):
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
            output_type=ChunkContextualizerOutput
        )
