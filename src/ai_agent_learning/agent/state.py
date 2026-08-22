import operator
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

MemoryRecallStatus = Literal["skipped", "completed", "failed"]
CriticStatus = Literal[
    "pending",
    "passed",
    "revision_required",
    "failed",
]


class SubagentCallRecord(TypedDict):
    call_id: str
    turn_id: str
    agent_name: str
    task: str
    signature: str
    status: str


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    session_user_id: NotRequired[str]
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
    recalled_memories: NotRequired[list[dict[str, object]]]
    memory_recall_status: NotRequired[MemoryRecallStatus]
    memory_recall_error: NotRequired[str | None]
    subagent_calls: NotRequired[
        Annotated[list[SubagentCallRecord], operator.add]
    ]
    draft_answer: NotRequired[str | None]
    critic_decision: NotRequired[dict[str, object] | None]
    critic_status: NotRequired[CriticStatus]
    critic_error: NotRequired[str | None]
    revision_count: NotRequired[int]
    max_revisions: NotRequired[int]
    revised_answer: NotRequired[str | None]
    revision_error: NotRequired[str | None]
    final_answer: NotRequired[str | None]
    unresolved_critic_issues: NotRequired[list[str]]
