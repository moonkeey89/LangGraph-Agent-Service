import unittest
from types import SimpleNamespace

from langchain_core.messages import AIMessage
from langgraph.types import Command, Interrupt

from ai_agent_learning.api.service import (
    AgentService,
    NoPendingInterruptError,
    PendingInterruptError,
    ThreadOwnershipError,
)


class CapturingGraph:
    def __init__(self):
        self.values = {}
        self.interrupts = ()
        self.invocations = []

    def get_state(self, _config):
        return SimpleNamespace(
            values=self.values,
            interrupts=self.interrupts,
        )

    def invoke(self, graph_input, **kwargs):
        self.invocations.append((graph_input, kwargs))
        return {
            "messages": [AIMessage(content="ok")],
            "final_answer": "ok",
        }


class AgentServiceTests(unittest.TestCase):
    def test_invoke_maps_thread_user_and_message(self):
        graph = CapturingGraph()
        service = AgentService(graph)

        result = service.invoke(
            message="hello",
            thread_id="thread-1",
            user_id="user-1",
        )

        graph_input, kwargs = graph.invocations[0]
        self.assertEqual(
            kwargs["config"],
            {"configurable": {"thread_id": "thread-1"}},
        )
        self.assertEqual(kwargs["context"].user_id, "user-1")
        self.assertEqual(graph_input["session_user_id"], "user-1")
        self.assertEqual(graph_input["messages"][0].content, "hello")
        self.assertEqual(result.answer, "ok")

    def test_pending_interrupt_requires_resume(self):
        graph = CapturingGraph()
        graph.values = {"session_user_id": "user-1"}
        graph.interrupts = (Interrupt(value={"action": "approval"}),)

        with self.assertRaises(PendingInterruptError):
            AgentService(graph).invoke(
                message="new message",
                thread_id="thread-1",
                user_id="user-1",
            )

    def test_resume_requires_pending_interrupt(self):
        graph = CapturingGraph()
        graph.values = {"session_user_id": "user-1"}

        with self.assertRaises(NoPendingInterruptError):
            AgentService(graph).resume(
                thread_id="thread-1",
                user_id="user-1",
                decision="approve",
            )

    def test_resume_uses_command(self):
        graph = CapturingGraph()
        graph.values = {"session_user_id": "user-1"}
        graph.interrupts = (Interrupt(value={"action": "approval"}),)

        AgentService(graph).resume(
            thread_id="thread-1",
            user_id="user-1",
            decision="reject",
            reason="no",
        )

        graph_input, kwargs = graph.invocations[0]
        self.assertIsInstance(graph_input, Command)
        self.assertEqual(
            graph_input.resume,
            {"approved": False, "reason": "no"},
        )
        self.assertEqual(
            kwargs["config"],
            {"configurable": {"thread_id": "thread-1"}},
        )

    def test_thread_owner_is_enforced(self):
        graph = CapturingGraph()
        graph.values = {"session_user_id": "owner"}
        with self.assertRaises(ThreadOwnershipError):
            AgentService(graph).invoke(
                message="hello",
                thread_id="thread-1",
                user_id="other",
            )


if __name__ == "__main__":
    unittest.main()
