import json
import unittest
from contextlib import contextmanager
from types import SimpleNamespace

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, AIMessageChunk
from langgraph.types import Interrupt

from ai_agent_learning.api.app import create_app
from ai_agent_learning.api.service import AgentService


class StreamingFakeGraph:
    """Deterministic v2 stream double; never calls DeepSeek."""

    def __init__(self):
        self.states: dict[str, dict[str, object]] = {}
        self.stream_calls = 0
        self.invoke_calls = 0
        self.memory_writes = 0

    @staticmethod
    def _thread(config) -> str:
        return config["configurable"]["thread_id"]

    def get_state(self, config):
        state = self.states.get(self._thread(config), {})
        return SimpleNamespace(
            values=dict(state.get("values", {})),
            interrupts=tuple(state.get("interrupts", ())),
        )

    def invoke(self, *_args, **_kwargs):
        self.invoke_calls += 1
        raise AssertionError("SSE must not invoke the graph a second time")

    def stream(
        self,
        graph_input,
        *,
        config,
        context,
        stream_mode,
        subgraphs,
        version,
    ):
        self.stream_calls += 1
        self.assert_stream_options(stream_mode, subgraphs, version)
        thread_id = self._thread(config)
        state = self.states.setdefault(
            thread_id,
            {"values": {}, "interrupts": ()},
        )
        message = str(graph_input["messages"][0].content)
        state["values"]["session_user_id"] = context.user_id

        if message == "raise-stream-secret":
            raise RuntimeError(
                "API_KEY=secret; E:/private/checkpoints.sqlite; traceback"
            )

        yield self.update("memory_recall")
        yield self.update("agent")
        if message == "interrupt":
            interrupt = Interrupt(
                value={
                    "action": "save_memory",
                    "tool_name": "save_memory",
                    "arguments": {"content": "安全内容"},
                    "message": "是否批准？",
                },
                id=f"interrupt-{thread_id}",
            )
            state["interrupts"] = (interrupt,)
            yield {
                "type": "updates",
                "ns": (),
                "data": {"__interrupt__": (interrupt,)},
            }
            return

        answer = "最终答案"
        for fragment in ("最终", "答案"):
            yield {
                "type": "messages",
                "ns": (),
                "data": (
                    AIMessageChunk(content=fragment),
                    {"langgraph_node": "agent"},
                ),
            }
        # These internal chunks must never become public token events.
        yield {
            "type": "messages",
            "ns": (),
            "data": (
                AIMessageChunk(content='{"verdict":"PASS"}'),
                {"langgraph_node": "critic"},
            ),
        }
        for node in (
            "capture_draft",
            "critic",
            "finalize",
            "memory_manager",
            "memory_executor",
        ):
            yield self.update(node)
        if message == "save-once":
            self.memory_writes += 1
        state["values"].update(
            {
                "messages": [AIMessage(content=answer)],
                "final_answer": answer,
            }
        )

    @staticmethod
    def assert_stream_options(stream_mode, subgraphs, version):
        if stream_mode != ["updates", "messages"]:
            raise AssertionError(stream_mode)
        if subgraphs is not False:
            raise AssertionError(subgraphs)
        if version != "v2":
            raise AssertionError(version)

    @staticmethod
    def update(node: str):
        return {"type": "updates", "ns": (), "data": {node: {}}}


def parse_sse(body: str) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    normalized = body.replace("\r\n", "\n")
    for block in normalized.split("\n\n"):
        if not block.strip() or block.startswith(":"):
            continue
        event_name = "message"
        data_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith("event: "):
                event_name = line.removeprefix("event: ")
            elif line.startswith("data: "):
                data_lines.append(line.removeprefix("data: "))
        events.append(
            {
                "event": event_name,
                "data": json.loads("\n".join(data_lines)),
            }
        )
    return events


class SseApiTests(unittest.TestCase):
    def setUp(self):
        self.graph = StreamingFakeGraph()

        @contextmanager
        def service_factory():
            yield AgentService(self.graph)

        self.client_context = TestClient(
            create_app(service_factory),
            raise_server_exceptions=False,
        )
        self.client = self.client_context.__enter__()

    def tearDown(self):
        self.client_context.__exit__(None, None, None)

    @staticmethod
    def headers(user_id: str = "stream-user") -> dict[str, str]:
        return {"X-User-ID": user_id}

    def stream(self, message: str, thread_id: str, user_id="stream-user"):
        with self.client.stream(
            "POST",
            "/api/v1/agent/stream",
            headers=self.headers(user_id),
            json={"message": message, "thread_id": thread_id},
        ) as response:
            content_type = response.headers.get("content-type", "")
            cache_control = response.headers.get("cache-control")
            buffering = response.headers.get("x-accel-buffering")
            body = "".join(response.iter_text())
        return response.status_code, content_type, cache_control, buffering, parse_sse(body)

    def test_started_progress_tokens_and_single_completed(self):
        status, content_type, cache, buffering, events = self.stream(
            "normal", "stream-normal"
        )

        self.assertEqual(status, 200)
        self.assertTrue(content_type.startswith("text/event-stream"))
        self.assertEqual(cache, "no-cache")
        self.assertEqual(buffering, "no")
        self.assertEqual(events[0]["event"], "started")
        progress = [
            item["data"]["node"]
            for item in events
            if item["event"] == "progress"
        ]
        self.assertEqual(
            progress,
            [
                "memory_recall",
                "agent",
                "capture_draft",
                "critic",
                "finalize",
                "memory_manager",
                "memory_executor",
            ],
        )
        tokens = [
            item["data"]["content"]
            for item in events
            if item["event"] == "token"
        ]
        self.assertEqual(tokens, ["最终", "答案"])
        self.assertNotIn('{"verdict":"PASS"}', tokens)
        terminals = [
            item for item in events
            if item["event"] in {"completed", "interrupted", "error"}
        ]
        self.assertEqual(len(terminals), 1)
        self.assertEqual(terminals[0]["event"], "completed")
        self.assertEqual(terminals[0]["data"]["answer"], "最终答案")

    def test_interrupt_is_the_only_terminal_event(self):
        *_, events = self.stream("interrupt", "stream-interrupt")
        terminals = [
            item for item in events
            if item["event"] in {"completed", "interrupted", "error"}
        ]
        self.assertEqual(len(terminals), 1)
        self.assertEqual(terminals[0]["event"], "interrupted")
        self.assertEqual(
            terminals[0]["data"]["interrupts"][0]["payload"][
                "tool_name"
            ],
            "save_memory",
        )
        self.assertFalse(any(item["event"] == "completed" for item in events))

    def test_exception_is_sanitized_error_terminal(self):
        *_, events = self.stream("raise-stream-secret", "stream-error")
        terminals = [
            item for item in events
            if item["event"] in {"completed", "interrupted", "error"}
        ]
        self.assertEqual(len(terminals), 1)
        self.assertEqual(terminals[0]["event"], "error")
        self.assertEqual(terminals[0]["data"]["code"], "internal_error")
        serialized = json.dumps(terminals[0], ensure_ascii=False)
        self.assertNotIn("API_KEY", serialized)
        self.assertNotIn("checkpoints.sqlite", serialized)
        self.assertNotIn("traceback", serialized)

    def test_other_user_cannot_stream_owned_thread(self):
        self.graph.states["owned"] = {
            "values": {"session_user_id": "owner"},
            "interrupts": (),
        }
        *_, events = self.stream("normal", "owned", user_id="other")
        self.assertEqual(events[0]["event"], "started")
        self.assertEqual(events[-1]["event"], "error")
        self.assertEqual(events[-1]["data"]["code"], "thread_forbidden")
        self.assertEqual(self.graph.stream_calls, 0)

    def test_legacy_thread_returns_conflict_event(self):
        self.graph.states["legacy"] = {
            "values": {"messages": [AIMessage(content="private")]},
            "interrupts": (),
        }
        *_, events = self.stream("normal", "legacy")
        self.assertEqual(events[-1]["event"], "error")
        self.assertEqual(
            events[-1]["data"]["code"], "legacy_thread_conflict"
        )
        self.assertNotIn("private", json.dumps(events, ensure_ascii=False))
        self.assertEqual(self.graph.stream_calls, 0)

    def test_graph_and_memory_side_effect_execute_once(self):
        *_, events = self.stream("save-once", "stream-once")
        self.assertEqual(events[-1]["event"], "completed")
        self.assertEqual(self.graph.stream_calls, 1)
        self.assertEqual(self.graph.invoke_calls, 0)
        self.assertEqual(self.graph.memory_writes, 1)


if __name__ == "__main__":
    unittest.main()
