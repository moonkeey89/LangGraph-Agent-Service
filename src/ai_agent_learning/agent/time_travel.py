from typing import Any

from langchain_core.messages import AIMessage, ToolMessage


SAFE_REPLAY_TOOL_NAMES = frozenset({"calculate"})


class TimeTravelError(ValueError):
    """Base error for invalid or unsafe checkpoint operations."""


class UnsafeTimeTravelError(TimeTravelError):
    """Raised when replaying a checkpoint may repeat a side effect."""


class ThreadIsolationError(TimeTravelError):
    """Raised when a checkpoint belongs to another thread."""


def select_checkpoint(snapshots: list[Any], sequence: int):
    """Select a snapshot by the one-based sequence printed by /history."""
    if sequence < 1 or sequence > len(snapshots):
        raise TimeTravelError(
            f"Checkpoint 序号必须在 1 到 {len(snapshots)} 之间。"
        )
    return snapshots[sequence - 1]


def checkpoint_id(snapshot) -> str:
    configurable = snapshot.config.get("configurable", {})
    value = configurable.get("checkpoint_id")
    if not value:
        raise TimeTravelError("所选快照缺少 checkpoint_id，无法执行 Time Travel。")
    return str(value)


def _snapshot_thread_id(snapshot) -> str | None:
    configurable = snapshot.config.get("configurable", {})
    value = configurable.get("thread_id")
    return str(value) if value is not None else None


def _ensure_same_thread(snapshot, thread_id: str) -> None:
    snapshot_thread_id = _snapshot_thread_id(snapshot)
    if snapshot_thread_id != thread_id:
        raise ThreadIsolationError(
            "所选 Checkpoint 不属于当前 thread_id，已拒绝跨线程重放。"
        )


def _messages(snapshot) -> list[Any]:
    values = snapshot.values if isinstance(snapshot.values, dict) else {}
    messages = values.get("messages", [])
    return list(messages) if isinstance(messages, (list, tuple)) else []


def _last_message(snapshot):
    messages = _messages(snapshot)
    return messages[-1] if messages else None


def _pending_tool_names(snapshot) -> set[str]:
    if "tools" not in snapshot.next:
        return set()

    for message in reversed(_messages(snapshot)):
        if isinstance(message, AIMessage) and message.tool_calls:
            return {
                str(tool_call.get("name", ""))
                for tool_call in message.tool_calls
            }
    return set()


def describe_replay_nodes(snapshot) -> tuple[str, ...]:
    """Describe the known node prefix that will run from this checkpoint."""
    if snapshot.next == ("tools",):
        return ("tools", "agent", "memory_manager", "memory_executor")
    if snapshot.next == ("agent",):
        return ("agent", "memory_manager", "memory_executor")
    return tuple(snapshot.next)


def validate_replay_checkpoint(snapshot, thread_id: str) -> None:
    """Allow only an unambiguous, side-effect-free calculate continuation."""
    _ensure_same_thread(snapshot, thread_id)
    checkpoint_id(snapshot)

    if snapshot.interrupts:
        raise UnsafeTimeTravelError(
            "该 Checkpoint 含有待处理 interrupt；请使用 Command(resume=...)，"
            "不能把 Resume 当作 Replay。"
        )

    if not snapshot.next:
        raise TimeTravelError("该 Checkpoint 已到达 END，没有可重新执行的节点。")

    if snapshot.next == ("tools",):
        tool_names = _pending_tool_names(snapshot)
        if not tool_names:
            raise UnsafeTimeTravelError(
                "无法识别 ToolNode 即将执行的工具，已按不安全路径拒绝 Replay。"
            )
        unsafe_tools = tool_names - SAFE_REPLAY_TOOL_NAMES
        if unsafe_tools:
            names = ", ".join(sorted(unsafe_tools))
            raise UnsafeTimeTravelError(
                f"该 Checkpoint 将执行非白名单工具：{names}；"
                "第一版只允许 Replay calculate。"
            )
        return

    if snapshot.next == ("agent",):
        last_message = _last_message(snapshot)
        if isinstance(last_message, ToolMessage) and last_message.name == "calculate":
            return
        raise UnsafeTimeTravelError(
            "只有 calculate 结果返回后的 AgentNode Checkpoint 可以 Replay；"
            "当前后续行为无法证明无副作用。"
        )

    raise UnsafeTimeTravelError(
        f"第一版不允许从节点 {snapshot.next} 执行 Replay。"
    )


def replay_checkpoint(graph, snapshot, thread_id: str, *, context=None):
    """Replay from the selected historical checkpoint's complete config."""
    validate_replay_checkpoint(snapshot, thread_id)
    return graph.invoke(None, snapshot.config, context=context)


def validate_fork_checkpoint(snapshot, thread_id: str) -> ToolMessage:
    """Require the unambiguous checkpoint after calculate and before agent."""
    validate_replay_checkpoint(snapshot, thread_id)
    last_message = _last_message(snapshot)
    if snapshot.next != ("agent",) or not isinstance(last_message, ToolMessage):
        raise TimeTravelError(
            "Fork 请选择 calculate 已返回 ToolMessage、下一步为 agent 的 Checkpoint。"
        )
    if last_message.name != "calculate":
        raise UnsafeTimeTravelError("第一版只允许修改 calculate 的 ToolMessage。")
    if not last_message.id:
        raise TimeTravelError("calculate ToolMessage 缺少消息 ID，无法安全替换。")
    return last_message


def fork_calculation_result(
    graph,
    snapshot,
    thread_id: str,
    replacement_content: str,
    *,
    context=None,
):
    """Replace one calculate ToolMessage, create a checkpoint, then continue."""
    original_message = validate_fork_checkpoint(snapshot, thread_id)
    if not replacement_content.strip():
        raise TimeTravelError("新的 calculate 结果不能为空。")

    replacement_message = original_message.model_copy(
        update={"content": replacement_content.strip()}
    )
    fork_config = graph.update_state(
        snapshot.config,
        {"messages": [replacement_message]},
        as_node="tools",
    )
    result = graph.invoke(None, fork_config, context=context)
    return fork_config, result
