import json
import logging

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import Command, Interrupt
from pydantic import ValidationError

from ai_agent_learning.agent import (
    build_graph,
    show_current_state,
    show_state_history,
)
from ai_agent_learning.checkpoint import CHECKPOINT_DB_PATH, open_sqlite_checkpointer
from ai_agent_learning.config import Settings
from ai_agent_learning.llm import create_llm
from ai_agent_learning.logging_config import configure_logging
from ai_agent_learning.tools import TOOLS


logger = logging.getLogger(__name__)
DEFAULT_THREAD_ID = "default"


def create_agent_app(
    settings: Settings,
    checkpointer: BaseCheckpointSaver,
):
    llm = create_llm(settings)
    return build_graph(llm, TOOLS, checkpointer=checkpointer)


def prompt_thread_id() -> str | None:
    try:
        thread_id = input(
            f"请输入会话 ID（直接回车使用 {DEFAULT_THREAD_ID}）:"
        ).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None

    return thread_id or DEFAULT_THREAD_ID


def _extract_interrupts(result) -> tuple[Interrupt, ...]:
    if isinstance(result, dict):
        interrupts = result.get("__interrupt__", ())
    else:
        interrupts = getattr(result, "interrupts", ())

    return tuple(interrupts or ())


def _pending_interrupts(app, config: dict) -> tuple[Interrupt, ...]:
    interrupts = getattr(app.get_state(config), "interrupts", ())
    if not isinstance(interrupts, (list, tuple)):
        return ()
    return tuple(interrupts)


def prompt_approval(interrupt_info: Interrupt) -> dict[str, object] | None:
    print("\n检测到需要人工审批的敏感操作：")
    print(json.dumps(interrupt_info.value, ensure_ascii=False, indent=2))

    while True:
        try:
            decision = input(
                "请输入 approve 批准、reject 拒绝，或 exit 保持暂停并退出："
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return None

        if decision == "approve":
            return {"approved": True}
        if decision == "reject":
            return {"approved": False, "reason": "用户拒绝"}
        if decision in {"exit", "quit"}:
            return None

        print("无法识别，请输入 approve、reject 或 exit。")


def resume_interrupted_graph(
    app,
    config: dict,
    interrupts: tuple[Interrupt, ...],
):
    current_interrupts = interrupts

    while current_interrupts:
        resume_value = prompt_approval(current_interrupts[0])
        if resume_value is None:
            print("审批状态已保留，可使用相同 thread_id 重新启动后继续。")
            return None

        result = app.invoke(Command(resume=resume_value), config=config)
        current_interrupts = _extract_interrupts(result)

    return result


def run_cli(app, thread_id: str) -> None:
    config = {"configurable": {"thread_id": thread_id}}

    try:
        pending = _pending_interrupts(app, config)
        if pending:
            print("检测到该会话存在尚未处理的审批请求。")
            resumed_result = resume_interrupted_graph(app, config, pending)
            if resumed_result is None:
                return
            print(resumed_result["messages"][-1].content)
    except Exception:
        logger.exception("Failed to resume pending approval")
        print("恢复待审批操作失败，请检查 Checkpoint 状态。")
        return

    while True:
        try:
            user_input = input("请输入（输入 exit 退出）:").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if user_input.lower() in {"exit", "quit"}:
            return

        if not user_input:
            continue

        if user_input.lower() in {"/state", "/history"}:
            try:
                if user_input.lower() == "/state":
                    show_current_state(app, thread_id)
                else:
                    show_state_history(app, thread_id)
            except Exception:
                logger.exception("Checkpoint inspection failed")
                print("状态查看失败，请检查当前 Graph 是否配置了 Checkpointer。")
            continue

        try:
            result = app.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
            )
            interrupts = _extract_interrupts(result)
            if interrupts:
                result = resume_interrupted_graph(app, config, interrupts)
                if result is None:
                    return
        except Exception:
            logger.exception("Agent request failed")
            print("抱歉，本次请求执行失败，请稍后重试。")
            continue

        print(result["messages"][-1].content)


def main() -> int:
    try:
        settings = Settings()
    except ValidationError:
        configure_logging("ERROR")
        logger.error("配置无效，请检查 .env 中的 DeepSeek 配置")
        return 1

    configure_logging(settings.log_level)

    try:
        with open_sqlite_checkpointer() as checkpointer:
            app = create_agent_app(settings, checkpointer)

            thread_id = prompt_thread_id()
            if thread_id is None:
                return 0

            logger.info(
                "AI Agent started with model %s, thread %s, checkpoint database %s",
                settings.deepseek_model,
                thread_id,
                CHECKPOINT_DB_PATH,
            )
            run_cli(app, thread_id)
    except Exception:
        logger.exception("Agent startup failed")
        return 1

    return 0
