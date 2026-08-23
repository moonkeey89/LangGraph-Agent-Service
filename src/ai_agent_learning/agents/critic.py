import json
import logging
import re
from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from pydantic import BaseModel, Field

from ai_agent_learning.agent.state import AgentState


logger = logging.getLogger(__name__)
MAX_REVISIONS = 1


class CriticDecision(BaseModel):
    """Strict review result; identity and storage controls are intentionally absent."""

    verdict: Literal["PASS", "REVISE"]
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    severity: Literal["none", "low", "medium", "high"]
    reason: str


_CRITIC_PROMPT = """你是只读 Critic Agent，只负责审查 Supervisor 草稿。
判断草稿是否完整覆盖用户请求、是否与Subagent摘要矛盾、是否遗漏明确约束。
只能依据输入JSON；不得要求调用工具、不得虚构新事实、不得修改用户身份或记忆。
如果存在Knowledge Agent结果，检查文档事实是否有对应sources、引用是否来自实际结果，
以及草稿是否越过证据；不得自行添加新的来源。
完整且一致时输出PASS；存在需要修改的问题时输出REVISE并给出具体issues和suggestions。"""

_REVISION_PROMPT = """你是回答修订节点，不绑定任何工具。
只使用输入中的用户请求、原始草稿、Critic issues/suggestions和Subagent结果摘要修订答案。
不得重新查询、不得编造事实、不得提及Critic、内部JSON、系统提示或执行轨迹。
直接输出一份可展示给用户的完整最终答案。"""

_CONSTRAINT_MARKERS = (
    "必须",
    "不要",
    "不能",
    "只要",
    "最多",
    "至少",
    "不超过",
    "以内",
    "预算",
    "每天",
    "总预算",
    "如果",
)


def _current_turn(state: AgentState) -> tuple[int, str]:
    messages = state.get("messages", [])
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, HumanMessage):
            return index, str(message.content).strip()
    return -1, ""


def _subagent_summaries(state: AgentState, human_index: int) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for message in state.get("messages", [])[human_index + 1 :]:
        if not isinstance(message, ToolMessage):
            continue
        if message.name not in {
            "ask_travel_agent",
            "ask_math_agent",
            "ask_knowledge_agent",
        }:
            continue
        try:
            payload = json.loads(str(message.content))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        summaries.append(
            {
                key: payload.get(key)
                for key in (
                    "agent_name",
                    "status",
                    "result",
                    "error",
                    "retry_recommended",
                    "sources",
                )
            }
        )
    return summaries


def _relevant_constraints(user_request: str) -> list[str]:
    clauses = [
        clause.strip()
        for clause in re.split(r"[，,。；;\n]", user_request)
        if clause.strip()
    ]
    return [
        clause
        for clause in clauses
        if any(marker in clause for marker in _CONSTRAINT_MARKERS)
    ]


def build_critic_context(state: AgentState) -> dict[str, object]:
    """Build a bounded context without history, runtime identity, or raw traces."""
    human_index, user_request = _current_turn(state)
    return {
        "user_request": user_request,
        "draft_answer": state.get("draft_answer") or "",
        "subagent_results": _subagent_summaries(state, human_index),
        "user_constraints": _relevant_constraints(user_request),
    }


def capture_draft(state: AgentState) -> dict[str, object]:
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else None
    if not isinstance(last_message, AIMessage) or last_message.tool_calls:
        raise ValueError("无法从当前 State 获取 Supervisor 最终草稿")

    updates: dict[str, object] = {
        "draft_answer": str(last_message.content),
        "critic_decision": None,
        "critic_status": "pending",
        "critic_error": None,
        "revision_count": 0,
        "max_revisions": MAX_REVISIONS,
        "revised_answer": None,
        "revision_error": None,
        "final_answer": None,
        "unresolved_critic_issues": [],
    }
    if last_message.id:
        updates["messages"] = [RemoveMessage(id=last_message.id)]
    else:
        logger.warning("Supervisor draft has no message id and cannot be removed")
    return updates


class CriticNode:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self._decision_llm = None

    def _structured_llm(self):
        if self._decision_llm is None:
            self._decision_llm = self.llm.with_structured_output(
                CriticDecision,
                method="function_calling",
            )
        return self._decision_llm

    def run(self, state: AgentState) -> dict[str, object]:
        try:
            raw_decision = self._structured_llm().invoke(
                [
                    SystemMessage(content=_CRITIC_PROMPT),
                    HumanMessage(
                        content=json.dumps(
                            build_critic_context(state),
                            ensure_ascii=False,
                        )
                    ),
                ]
            )
            decision = (
                raw_decision
                if isinstance(raw_decision, CriticDecision)
                else CriticDecision.model_validate(raw_decision)
            )
            status = (
                "passed"
                if decision.verdict == "PASS"
                else "revision_required"
            )
            logger.info(
                "Critic verdict=%s severity=%s issues=%s revision_count=%s",
                decision.verdict,
                decision.severity,
                len(decision.issues),
                state.get("revision_count", 0),
            )
            return {
                "critic_decision": decision.model_dump(mode="json"),
                "critic_status": status,
                "critic_error": None,
            }
        except Exception as error:
            logger.exception("Critic structured review failed")
            return {
                "critic_decision": None,
                "critic_status": "failed",
                "critic_error": f"{type(error).__name__}: {error}",
            }


def route_after_critic(state: AgentState) -> Literal["revise", "finalize"]:
    if state.get("critic_status") != "revision_required":
        return "finalize"
    revision_count = int(state.get("revision_count", 0))
    max_revisions = max(0, int(state.get("max_revisions", MAX_REVISIONS)))
    return "revise" if revision_count < max_revisions else "finalize"


class RevisionNode:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def run(self, state: AgentState) -> dict[str, object]:
        decision = state.get("critic_decision") or {}
        context = build_critic_context(state)
        revision_input = {
            "user_request": context["user_request"],
            "draft_answer": context["draft_answer"],
            "subagent_results": context["subagent_results"],
            "user_constraints": context["user_constraints"],
            "issues": decision.get("issues", []),
            "suggestions": decision.get("suggestions", []),
        }
        next_count = int(state.get("revision_count", 0)) + 1
        try:
            response = self.llm.invoke(
                [
                    SystemMessage(content=_REVISION_PROMPT),
                    HumanMessage(
                        content=json.dumps(
                            revision_input,
                            ensure_ascii=False,
                        )
                    ),
                ]
            )
            revised_answer = str(response.content).strip()
            if not revised_answer:
                raise ValueError("修订模型返回了空答案")
            logger.info("Critic revision applied revision_count=%s", next_count)
            return {
                "revised_answer": revised_answer,
                "revision_count": next_count,
                "revision_error": None,
            }
        except Exception as error:
            logger.exception("Critic revision failed; keeping original draft")
            return {
                "revised_answer": None,
                "revision_count": next_count,
                "revision_error": f"{type(error).__name__}: {error}",
            }


def finalize_answer(state: AgentState) -> dict[str, object]:
    final_answer = (
        state.get("revised_answer")
        or state.get("draft_answer")
        or "抱歉，本次没有生成可用回答。"
    )
    decision = state.get("critic_decision") or {}
    unresolved = (
        list(decision.get("issues", []))
        if decision.get("verdict") == "REVISE"
        else []
    )
    revision_count = int(state.get("revision_count", 0))
    logger.info(
        "Finalized answer revised=%s revision_count=%s unresolved_issues=%s",
        bool(state.get("revised_answer")),
        revision_count,
        len(unresolved),
    )
    return {
        "messages": [AIMessage(content=str(final_answer))],
        "final_answer": str(final_answer),
        "unresolved_critic_issues": unresolved,
        "status": "completed",
    }


class CriticWorkflow:
    """Attach one review and at most one revision before the final target."""

    def __init__(
        self,
        critic_llm: BaseChatModel,
        revision_llm: BaseChatModel,
    ):
        self.critic = CriticNode(critic_llm)
        self.revision = RevisionNode(revision_llm)

    def attach(self, graph: Any, *, final_target: str) -> str:
        graph.add_node("capture_draft", capture_draft)
        graph.add_node("critic", self.critic.run)
        graph.add_node("revise", self.revision.run)
        graph.add_node("finalize", finalize_answer)
        graph.add_edge("capture_draft", "critic")
        graph.add_conditional_edges(
            "critic",
            route_after_critic,
            {"revise": "revise", "finalize": "finalize"},
        )
        # First version deliberately does not run Critic a second time.
        graph.add_edge("revise", "finalize")
        graph.add_edge("finalize", final_target)
        return "capture_draft"
