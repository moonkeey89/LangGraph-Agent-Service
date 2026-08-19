import logging
from collections.abc import Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool

from ai_agent_learning.agent.error_recovery import DEFAULT_MAX_RETRIES
from ai_agent_learning.agent.state import AgentState


logger = logging.getLogger(__name__)


class AgentNode:
    def __init__(self, llm: BaseChatModel, tools: Sequence[BaseTool]):
        self.llm = llm.bind_tools(tools)

    def run(self, state: AgentState) -> dict[str, object]:
        logger.debug("Invoking agent model")
        response = self.llm.invoke(state["messages"])
        logger.debug("Agent model invocation completed")

        return {
            "messages": [response],
            "status": "running" if response.tool_calls else "completed",
            "error": None,
            "error_type": None,
            "failed_node": None,
            "retry_count": 0,
            "max_retries": state.get("max_retries", DEFAULT_MAX_RETRIES),
        }
