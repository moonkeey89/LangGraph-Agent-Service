import logging
from collections.abc import Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool

from ai_agent_learning.agent.state import AgentState


logger = logging.getLogger(__name__)


class AgentNode:
    def __init__(self, llm: BaseChatModel, tools: Sequence[BaseTool]):
        self.llm = llm.bind_tools(tools)

    def run(self, state: AgentState) -> dict[str, list[BaseMessage]]:
        logger.debug("Invoking agent model")
        response = self.llm.invoke(state["messages"])
        logger.debug("Agent model invocation completed")

        return {"messages": [response]}
