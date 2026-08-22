import unittest
from contextlib import contextmanager
from types import SimpleNamespace

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langgraph.types import Command, Interrupt

from ai_agent_learning.api.app import create_app
from ai_agent_learning.api.service import AgentService


class StatefulFakeGraph:
    """Small deterministic graph double; no DeepSeek or local model download."""

    def __init__(self):
        self.states: dict[str, dict[str, object]] = {}
        self.memories: dict[str, str] = {}
        self.calls: list[dict[str, object]] = []

    @staticmethod
    def _thread(config) -> str:
        return config["configurable"]["thread_id"]

    def get_state(self, config):
        state = self.states.get(self._thread(config), {})
        return SimpleNamespace(
            values=dict(state.get("values", {})),
            interrupts=tuple(state.get("interrupts", ())),
        )

    def invoke(self, graph_input, *, config, context):
        thread_id = self._thread(config)
        user_id = context.user_id
        state = self.states.setdefault(
            thread_id,
            {"values": {}, "interrupts": (), "facts": {}},
        )
        self.calls.append(
            {
                "input": graph_input,
                "thread_id": thread_id,
                "user_id": user_id,
            }
        )

        if isinstance(graph_input, Command):
            resume = graph_input.resume
            state["interrupts"] = ()
            if resume.get("approved") is True:
                pending = str(state.pop("pending_memory"))
                self.memories[user_id] = pending
                answer = "长期记忆保存成功"
            else:
                answer = "操作已取消"
            return {
                "messages": [AIMessage(content=answer)],
                "final_answer": answer,
            }

        message = str(graph_input["messages"][0].content)
        state["values"]["session_user_id"] = graph_input[
            "session_user_id"
        ]
        if message == "raise-internal-secret":
            raise RuntimeError(
                "API_KEY=secret; E:/private/checkpoints.sqlite; traceback"
            )
        if message.startswith("请记住"):
            content = message.removeprefix("请记住").strip("：:，,。 ")
            interrupt = Interrupt(
                value={
                    "action": "save_memory",
                    "tool_name": "save_memory",
                    "arguments": {"content": content},
                    "message": "是否批准保存长期记忆？",
                },
                id=f"interrupt-{thread_id}",
            )
            state["pending_memory"] = content
            state["interrupts"] = (interrupt,)
            return {"__interrupt__": (interrupt,)}
        if message.startswith("我的名字是"):
            state["facts"]["name"] = message.removeprefix("我的名字是")
            answer = "好的"
        elif message == "我叫什么名字？":
            answer = str(state["facts"].get("name", "不知道"))
        elif message == "我主要使用什么语言？":
            answer = self.memories.get(user_id, "不知道")
        else:
            answer = f"回答：{message}"
        return {
            "messages": [AIMessage(content=answer)],
            "final_answer": answer,
        }


class AgentApiTests(unittest.TestCase):
    def setUp(self):
        self.graph = StatefulFakeGraph()

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
    def headers(user_id: str = "user_001") -> dict[str, str]:
        return {"X-User-ID": user_id}

    def invoke(self, message: str, thread_id: str, user_id="user_001"):
        return self.client.post(
            "/api/v1/agent/invoke",
            headers=self.headers(user_id),
            json={"message": message, "thread_id": thread_id},
        )

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_normal_invoke(self):
        response = self.invoke("你好", "api-normal")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "completed")
        self.assertEqual(response.json()["answer"], "回答：你好")
        self.assertEqual(self.graph.calls[-1]["user_id"], "user_001")

    def test_missing_required_parameters_returns_422(self):
        response = self.client.post(
            "/api/v1/agent/invoke",
            headers=self.headers(),
            json={},
        )
        self.assertEqual(response.status_code, 422)

    def test_same_user_and_thread_continue_session(self):
        self.invoke("我的名字是小明", "api-session")
        response = self.invoke("我叫什么名字？", "api-session")
        self.assertEqual(response.json()["answer"], "小明")

    def test_long_memory_is_shared_across_threads_for_same_user(self):
        interrupted = self.invoke(
            "请记住：我主要使用Python", "api-memory-a"
        )
        self.assertEqual(interrupted.json()["status"], "interrupted")
        resumed = self.client.post(
            "/api/v1/agent/resume",
            headers=self.headers(),
            json={"thread_id": "api-memory-a", "decision": "approve"},
        )
        self.assertEqual(resumed.json()["status"], "completed")
        recalled = self.invoke(
            "我主要使用什么语言？", "api-memory-b"
        )
        self.assertIn("Python", recalled.json()["answer"])

    def test_long_memory_is_isolated_between_users(self):
        self.graph.memories["user_001"] = "我主要使用Python"
        response = self.invoke(
            "我主要使用什么语言？",
            "api-user-2",
            user_id="user_002",
        )
        self.assertEqual(response.json()["answer"], "不知道")

    def test_interrupt_is_a_normal_http_response(self):
        response = self.invoke("请记住：我喜欢数学", "api-interrupt")
        body = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(body["status"], "interrupted")
        self.assertEqual(body["thread_id"], "api-interrupt")
        self.assertEqual(
            body["interrupts"][0]["payload"]["tool_name"],
            "save_memory",
        )

    def test_resume_uses_command_and_original_thread(self):
        self.invoke("请记住：我喜欢数学", "api-resume")
        response = self.client.post(
            "/api/v1/agent/resume",
            headers=self.headers(),
            json={"thread_id": "api-resume", "decision": "approve"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "completed")
        last_call = self.graph.calls[-1]
        self.assertEqual(last_call["thread_id"], "api-resume")
        self.assertIsInstance(last_call["input"], Command)
        self.assertEqual(last_call["input"].resume, {"approved": True})

    def test_thread_cannot_be_reused_by_another_user(self):
        self.invoke("你好", "api-owned", user_id="user_001")
        response = self.invoke("你好", "api-owned", user_id="user_002")
        self.assertEqual(response.status_code, 403)
        self.assertNotIn("user_001", response.text)

    def test_internal_error_is_sanitized(self):
        response = self.invoke("raise-internal-secret", "api-error")
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json(), {"detail": "Internal agent error"})
        self.assertNotIn("API_KEY", response.text)
        self.assertNotIn("checkpoints.sqlite", response.text)


if __name__ == "__main__":
    unittest.main()
