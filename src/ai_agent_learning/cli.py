import json
import logging

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import Command, Interrupt
from pydantic import ValidationError

from ai_agent_learning.agent import (
    build_graph,
    checkpoint_id,
    describe_replay_nodes,
    fork_calculation_result,
    replay_checkpoint,
    select_checkpoint,
    show_current_state,
    show_state_history,
    TimeTravelError,
    validate_fork_checkpoint,
    validate_replay_checkpoint,
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
    print("\n检测到需要人工处理的操作：")
    print(json.dumps(interrupt_info.value, ensure_ascii=False, indent=2))
    is_failure_review = (
        isinstance(interrupt_info.value, dict)
        and interrupt_info.value.get("action") == "tool_failure_review"
    )

    while True:
        try:
            prompt = (
                "请输入 retry 再试一次、cancel 取消，或 exit 保持暂停并退出："
                if is_failure_review
                else "请输入 approve 批准、reject 拒绝，或 exit 保持暂停并退出："
            )
            decision = input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return None

        if is_failure_review:
            if decision == "retry":
                return {"action": "retry"}
            if decision == "cancel":
                return {"action": "cancel", "reason": "用户取消重试"}
        else:
            if decision == "approve":
                return {"approved": True}
            if decision == "reject":
                return {"approved": False, "reason": "用户拒绝"}
        if decision in {"exit", "quit"}:
            return None

        expected = (
            "retry、cancel 或 exit"
            if is_failure_review
            else "approve、reject 或 exit"
        )
        print(f"无法识别，请输入 {expected}。")


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


def prompt_checkpoint_selection(snapshots, action: str):
    """让用户按 /history 展示的、从 1 开始的序号选择 Checkpoint。"""
    if not snapshots:
        return None

    while True:
        try:
            raw_sequence = input(
                f"请输入要{action}的 Checkpoint 序号（输入 cancel 取消）："
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None

        if raw_sequence.lower() in {"cancel", "exit", "quit"}:
            return None

        try:
            return select_checkpoint(snapshots, int(raw_sequence))
        except (ValueError, TimeTravelError) as exc:
            print(f"选择无效：{exc}")


def _print_time_travel_result(result) -> None:
    interrupts = _extract_interrupts(result)
    if interrupts:
        print(
            "Time Travel 分支遇到新的 interrupt，已保持暂停；"
            "系统不会自动批准敏感操作。"
        )
        return

    if isinstance(result, dict) and result.get("messages"):
        print(result["messages"][-1].content)


def run_replay_command(app, thread_id: str) -> None:
    snapshots = show_state_history(app, thread_id)
    selected = prompt_checkpoint_selection(snapshots, "Replay")
    if selected is None:
        return

    try:
        validate_replay_checkpoint(selected, thread_id)
        nodes = " → ".join(describe_replay_nodes(selected))
        print(f"将使用完整 config Replay：checkpoint_id={checkpoint_id(selected)}")
        print(f"将从 next 开始重新执行：{nodes}")
        print("该 Checkpoint 之前的节点不会重新执行。")
        result = replay_checkpoint(app, selected, thread_id)
    except TimeTravelError as exc:
        print(f"Replay 已拒绝：{exc}")
        return

    _print_time_travel_result(result)


def run_fork_command(app, thread_id: str) -> None:
    snapshots = show_state_history(app, thread_id)
    selected = prompt_checkpoint_selection(snapshots, "Fork")
    if selected is None:
        return

    try:
        original_message = validate_fork_checkpoint(selected, thread_id)
    except TimeTravelError as exc:
        print(f"Fork 已拒绝：{exc}")
        return

    try:
        replacement = input(
            f"原 calculate 结果为 {original_message.content!s}，请输入分支中的新结果："
        ).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return

    try:
        original_checkpoint_id = checkpoint_id(selected)
        fork_config, result = fork_calculation_result(
            app,
            selected,
            thread_id,
            replacement,
        )
    except TimeTravelError as exc:
        print(f"Fork 已拒绝：{exc}")
        return

    new_checkpoint_id = fork_config["configurable"]["checkpoint_id"]
    print(f"原 checkpoint_id: {original_checkpoint_id}")
    print(f"update_state() 创建的新 checkpoint_id: {new_checkpoint_id}")
    print("本次更新按 tools 节点输出处理，后继路径为 tool_success → agent。")
    _print_time_travel_result(result)


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

        if user_input.lower() in {"/state", "/history", "/replay", "/fork"}:
            try:
                if user_input.lower() == "/state":
                    show_current_state(app, thread_id)
                elif user_input.lower() == "/history":
                    show_state_history(app, thread_id)
                elif user_input.lower() == "/replay":
                    run_replay_command(app, thread_id)
                else:
                    run_fork_command(app, thread_id)
            except Exception:
                logger.exception("Checkpoint or Time Travel command failed")
                print("Checkpoint 操作失败，请检查日志和当前会话状态。")
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
