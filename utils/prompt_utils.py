import semver
from pathlib import Path
from typing import Union


PROMPT_PATH = Path(__file__).parent.parent / "agents/prompts"

def get_prompt(agent_name: str, version: Union[str | None]) -> str:
    prompts = PROMPT_PATH.glob(f"{agent_name}*")
    if version:
        version = semver.Version.parse(version)
    higher = None
    wanted_prompt = None
    for prompt in prompts:
        prompt_version = semver.Version.parse(".".join(prompt.name.split(".")[1:]))
        if version:
            wanted_prompt = prompt
            break
        elif (higher and prompt_version > higher) or not higher:
            higher = prompt_version
            wanted_prompt = prompt
    with open(wanted_prompt, "r") as f:
        return f.read()