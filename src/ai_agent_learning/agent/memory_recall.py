import logging
import re

from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime

from ai_agent_learning.agent.context import AgentContext
from ai_agent_learning.agent.state import AgentState
from ai_agent_learning.skills.memory import MemoryPolicyError, search_memory


logger = logging.getLogger(__name__)

_PERSONAL_MEMORY_QUERY = re.compile(
    r"(?:"
    r"我是谁|我叫什么|我的名字|我有什么|你(?:还)?记得我|关于我|"
    r"我(?:喜欢|爱|不喜欢).*(?:什么|哪些|谁)|"
    r"我的.*(?:是什么|有哪些|什么|哪些)|"
    r"我.*(?:使用|用).*(?:什么|哪些|哪一个)|"
    r"我.*(?:目标|偏好).*(?:什么|哪些)"
    r")",
    re.IGNORECASE,
)


def is_memory_recall_query(user_message: str) -> bool:
    """Return whether the current question may need user-scoped memory."""
    return bool(_PERSONAL_MEMORY_QUERY.search(user_message.strip()))


def _latest_human_text(state: AgentState) -> str:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            return str(message.content).strip()
    return ""


def _trusted_user_id(runtime: Runtime[AgentContext]) -> str:
    context = runtime.context
    user_id = context.user_id if isinstance(context, AgentContext) else ""
    if not user_id.strip():
        raise MemoryPolicyError("运行时上下文缺少可信 user_id")
    return user_id.strip()


class MemoryRecallNode:
    """Retrieve bounded user memories before AgentNode answers."""

    def run(
        self,
        state: AgentState,
        runtime: Runtime[AgentContext],
    ) -> dict[str, object]:
        query = _latest_human_text(state)
        if not query or not is_memory_recall_query(query):
            return {
                "recalled_memories": [],
                "memory_recall_status": "skipped",
                "memory_recall_error": None,
            }
        if runtime.store is None:
            return {
                "recalled_memories": [],
                "memory_recall_status": "skipped",
                "memory_recall_error": None,
            }

        try:
            memories = search_memory(
                runtime.store,
                user_id=_trusted_user_id(runtime),
                query=query,
            )
            recalled = [
                {
                    "memory_id": memory["memory_id"],
                    "content": memory["content"],
                    "memory_type": memory["memory_type"],
                    "source": memory["source"],
                }
                for memory in memories
            ]
            return {
                "recalled_memories": recalled,
                "memory_recall_status": "completed",
                "memory_recall_error": None,
            }
        except Exception as error:
            # Recall is optional context and must not block the main answer.
            logger.exception("Memory Recall failed")
            return {
                "recalled_memories": [],
                "memory_recall_status": "failed",
                "memory_recall_error": f"{type(error).__name__}: {error}",
            }
