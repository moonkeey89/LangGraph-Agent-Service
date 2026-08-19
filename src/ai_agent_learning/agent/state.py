from typing import Annotated, Literal, NotRequired, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


ErrorType = Literal[
    "transient",
    "invalid_arguments",
    "permission",
    "permanent",
    "side_effect_unknown",
]

AgentStatus = Literal[
    "running",
    "completed",
    "retry",
    "agent_correction",
    "human_review",
    "success",
    "failed",
    "cancelled",
]


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    status: NotRequired[AgentStatus]
    error: NotRequired[str | None]
    error_type: NotRequired[ErrorType | None]
    failed_node: NotRequired[str | None]
    retry_count: NotRequired[int]
    max_retries: NotRequired[int]
