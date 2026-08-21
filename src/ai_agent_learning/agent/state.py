from typing import Annotated, Literal

from typing_extensions import NotRequired, TypedDict

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

MemoryManagerStatus = Literal[
    "skipped",
    "decided",
    "applied",
    "rejected",
    "failed",
]


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    status: NotRequired[AgentStatus]
    error: NotRequired[str | None]
    error_type: NotRequired[ErrorType | None]
    failed_node: NotRequired[str | None]
    retry_count: NotRequired[int]
    max_retries: NotRequired[int]
    memory_decision: NotRequired[dict[str, object] | None]
    memory_candidate_ids: NotRequired[list[str]]
    memory_manager_status: NotRequired[MemoryManagerStatus]
    memory_manager_error: NotRequired[str | None]
