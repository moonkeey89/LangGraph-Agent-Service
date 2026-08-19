from pprint import pformat


def _thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _format_message_content(message) -> str:
    content = getattr(message, "content", "")
    if content:
        return str(content)

    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        return f"<空内容；tool_calls={tool_calls}>"

    return "<空内容>"


def _format_next(next_nodes: tuple[str, ...]) -> str:
    return ", ".join(next_nodes) if next_nodes else "END"


def show_current_state(graph, thread_id: str):
    """打印并返回指定线程的最新 StateSnapshot。"""
    snapshot = graph.get_state(_thread_config(thread_id))

    print(f"\n=== 当前 StateSnapshot：thread_id={thread_id} ===")
    print("snapshot.values:")
    print(pformat(snapshot.values, sort_dicts=False))
    print(f"snapshot.next: {snapshot.next}")
    print("snapshot.config:")
    print(pformat(snapshot.config, sort_dicts=False))
    print("snapshot.metadata:")
    print(pformat(snapshot.metadata, sort_dicts=False))
    print(f"snapshot.created_at: {snapshot.created_at}")
    print("snapshot.parent_config:")
    print(pformat(snapshot.parent_config, sort_dicts=False))

    return snapshot


def show_state_history(graph, thread_id: str):
    """按从新到旧的顺序打印并返回指定线程的历史快照。"""
    snapshots = list(graph.get_state_history(_thread_config(thread_id)))

    print(f"\n=== StateSnapshot 历史：thread_id={thread_id}（从新到旧） ===")
    if not snapshots:
        print("未找到 Checkpoint 历史。")
        return snapshots

    for index, snapshot in enumerate(snapshots, start=1):
        values = snapshot.values if isinstance(snapshot.values, dict) else {}
        messages = values.get("messages", [])
        last_message = messages[-1] if messages else None
        configurable = snapshot.config.get("configurable", {})

        print(f"\nCheckpoint #{index}{'（最新）' if index == 1 else ''}")
        print(f"  创建时间: {snapshot.created_at}")
        print(f"  消息数量: {len(messages)}")
        if last_message is None:
            print("  最后一条消息: <无>")
        else:
            print(f"  最后一条消息类型: {type(last_message).__name__}")
            print(f"  最后一条消息内容: {_format_message_content(last_message)}")
        print(f"  下一步执行节点: {_format_next(snapshot.next)}")
        print(f"  checkpoint_id: {configurable.get('checkpoint_id', '<无>')}")

    return snapshots
