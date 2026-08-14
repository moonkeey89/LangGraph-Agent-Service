from typing import TypedDict, Any

class AgentState(TypedDict):
    task: str
    plan: Any
    results: list
    final_answer: str