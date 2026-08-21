import json
import logging
from hashlib import sha256
from typing import Protocol

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command

from ai_agent_learning.agent.context import AgentContext
from ai_agent_learning.agent.state import AgentState, SubagentCallRecord
from ai_agent_learning.agents.subagent import SubagentResult


logger = logging.getLogger(__name__)
DEFAULT_MAX_SUBAGENT_CALLS = 4


class SubagentInvoker(Protocol):
    agent_name: str

    def invoke(self, task: str) -> SubagentResult: ...


def _latest_turn_id(state: AgentState) -> str:
    messages = state.get("messages", [])
    for reverse_index, message in enumerate(reversed(messages)):
        if isinstance(message, HumanMessage):
            if message.id:
                return str(message.id)
            digest = sha256(
                f"{len(messages) - reverse_index}\0{message.content}".encode()
            ).hexdigest()
            return f"turn-{digest[:20]}"
    return "turn-unknown"


def _call_signature(agent_name: str, task: str) -> str:
    normalized = " ".join(task.casefold().split())
    digest = sha256(f"{agent_name}\0{normalized}".encode()).hexdigest()
    return digest[:24]


def _summary(value: str, limit: int = 160) -> str:
    compact = " ".join(value.split())
    return compact if len(compact) <= limit else f"{compact[:limit]}..."


class SubagentDispatcher:
    """Enforce handoff boundaries before invoking a stateless specialist."""

    def __init__(self, max_subagent_calls: int = DEFAULT_MAX_SUBAGENT_CALLS):
        if max_subagent_calls <= 0:
            raise ValueError("max_subagent_calls 必须是正整数")
        self.max_subagent_calls = max_subagent_calls

    def invoke(
        self,
        subagent: SubagentInvoker,
        task: str,
        runtime: ToolRuntime[AgentContext, AgentState],
    ) -> Command:
        normalized_task = task.strip()
        state = runtime.state if isinstance(runtime.state, dict) else {}
        turn_id = _latest_turn_id(state)
        signature = _call_signature(subagent.agent_name, normalized_task)
        current_turn_calls = [
            record
            for record in state.get("subagent_calls", [])
            if record.get("turn_id") == turn_id
        ]
        call_id = str(runtime.tool_call_id or f"handoff-{signature}")

        if len(current_turn_calls) >= self.max_subagent_calls:
            return self._blocked_command(
                subagent=subagent,
                task=normalized_task,
                call_id=call_id,
                turn_id=turn_id,
                signature=signature,
                reason=(
                    "本轮 Subagent 调用已达到上限 "
                    f"{self.max_subagent_calls}，停止继续协调"
                ),
            )

        if any(
            record.get("signature") == signature
            for record in current_turn_calls
        ):
            return self._blocked_command(
                subagent=subagent,
                task=normalized_task,
                call_id=call_id,
                turn_id=turn_id,
                signature=signature,
                reason="同一轮不能用相同任务重复调用同一个 Subagent",
            )

        logger.info(
            "Supervisor handoff #%s/%s: agent=%s task=%s",
            len(current_turn_calls) + 1,
            self.max_subagent_calls,
            subagent.agent_name,
            _summary(normalized_task),
        )
        try:
            result = subagent.invoke(normalized_task)
        except Exception as error:
            logger.exception("Subagent boundary failed: %s", subagent.agent_name)
            result = SubagentResult(
                agent_name=subagent.agent_name,
                status="failed",
                result=None,
                error=f"{type(error).__name__}: {error}",
                retry_recommended=False,
            )

        logger.info(
            "Subagent completed: agent=%s status=%s result=%s error=%s",
            subagent.agent_name,
            result.status,
            _summary(result.result or ""),
            _summary(result.error or ""),
        )
        record = self._record(
            call_id=call_id,
            turn_id=turn_id,
            subagent=subagent,
            task=normalized_task,
            signature=signature,
            status=result.status,
        )
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=json.dumps(
                            result.to_dict(),
                            ensure_ascii=False,
                        ),
                        name=self._tool_name(subagent.agent_name),
                        tool_call_id=call_id,
                        id=f"subagent-result:{call_id}",
                    )
                ],
                "subagent_calls": [record],
            }
        )

    def _blocked_command(
        self,
        *,
        subagent: SubagentInvoker,
        task: str,
        call_id: str,
        turn_id: str,
        signature: str,
        reason: str,
    ) -> Command:
        logger.warning(
            "Supervisor handoff blocked: agent=%s reason=%s task=%s",
            subagent.agent_name,
            reason,
            _summary(task),
        )
        record = self._record(
            call_id=call_id,
            turn_id=turn_id,
            subagent=subagent,
            task=task,
            signature=signature,
            status="blocked",
        )
        payload = {
            "agent_name": subagent.agent_name,
            "status": "blocked",
            "result": None,
            "error": reason,
            "retry_recommended": False,
        }
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=json.dumps(payload, ensure_ascii=False),
                        name=self._tool_name(subagent.agent_name),
                        tool_call_id=call_id,
                        id=f"subagent-result:{call_id}",
                        status="error",
                    )
                ],
                "subagent_calls": [record],
                "status": "failed",
                "error": reason,
                "error_type": "permanent",
                "failed_node": "tools",
            }
        )

    @staticmethod
    def _record(
        *,
        call_id: str,
        turn_id: str,
        subagent: SubagentInvoker,
        task: str,
        signature: str,
        status: str,
    ) -> SubagentCallRecord:
        return {
            "call_id": call_id,
            "turn_id": turn_id,
            "agent_name": subagent.agent_name,
            "task": task,
            "signature": signature,
            "status": status,
        }

    @staticmethod
    def _tool_name(agent_name: str) -> str:
        return f"ask_{agent_name}"


def create_subagent_tools(
    travel_agent: SubagentInvoker,
    math_agent: SubagentInvoker,
    *,
    max_subagent_calls: int = DEFAULT_MAX_SUBAGENT_CALLS,
) -> list[BaseTool]:
    dispatcher = SubagentDispatcher(max_subagent_calls)

    @tool
    def ask_travel_agent(
        task: str,
        runtime: ToolRuntime[AgentContext, AgentState],
    ) -> Command:
        """委派天气、景点和基础旅游信息任务；不能进行预算或数学计算。"""
        return dispatcher.invoke(travel_agent, task, runtime)

    @tool
    def ask_math_agent(
        task: str,
        runtime: ToolRuntime[AgentContext, AgentState],
    ) -> Command:
        """委派数学、算术和预算计算任务；不能查询天气或旅游景点。"""
        return dispatcher.invoke(math_agent, task, runtime)

    return [ask_travel_agent, ask_math_agent]
