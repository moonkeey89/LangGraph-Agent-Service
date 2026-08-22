import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from langchain_core.messages import AIMessage, HumanMessage

from ai_agent_learning.agent import AgentContext
from ai_agent_learning.agents import CriticDecision, build_supervisor_graph
from ai_agent_learning.checkpoint import open_sqlite_checkpointer
from tests.integration.test_multi_agent import RoleAwareModel


class StaticCriticModel:
    def __init__(
        self,
        decision: CriticDecision | None = None,
        error: Exception | None = None,
    ):
        self.decision = decision
        self.error = error
        self.calls: list[list] = []
        self.structured_schema = None

    def bind_tools(self, _tools):
        raise AssertionError("Critic must not bind business tools")

    def with_structured_output(self, schema, method=None):
        self.structured_schema = schema
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        if self.error is not None:
            raise self.error
        return self.decision


class StaticRevisionModel:
    def __init__(self, answer: str):
        self.answer = answer
        self.calls: list[list] = []

    def bind_tools(self, _tools):
        raise AssertionError("Revision must not bind business tools")

    def invoke(self, messages):
        self.calls.append(messages)
        return AIMessage(content=self.answer)


def _pass_decision() -> CriticDecision:
    return CriticDecision(
        verdict="PASS",
        issues=[],
        suggestions=[],
        severity="none",
        reason="草稿覆盖全部任务且与Subagent结果一致",
    )


def _revise_decision(issue: str, suggestion: str) -> CriticDecision:
    return CriticDecision(
        verdict="REVISE",
        issues=[issue],
        suggestions=[suggestion],
        severity="medium",
        reason="草稿需要一次修订",
    )


class CriticIntegrationTests(unittest.TestCase):
    request = "查询北京天气和主要景点，如果每天预算500元，计算3天总预算。"

    def _invoke(self, app, thread_id: str):
        return app.invoke(
            {"messages": [HumanMessage(content=self.request)]},
            config={"configurable": {"thread_id": thread_id}},
            context=AgentContext(user_id="critic-user"),
        )

    def test_critic_nodes_are_only_on_normal_supervisor_answer_path(self):
        app = build_supervisor_graph(
            RoleAwareModel(),
            critic_llm=StaticCriticModel(_pass_decision()),
            revision_llm=StaticRevisionModel("不应执行"),
        )
        graph = app.get_graph()
        nodes = set(graph.nodes)
        edges = {(edge.source, edge.target) for edge in graph.edges}

        self.assertTrue(
            {"capture_draft", "critic", "revise", "finalize"}.issubset(
                nodes
            )
        )
        self.assertTrue(
            {
                ("agent", "capture_draft"),
                ("capture_draft", "critic"),
                ("critic", "revise"),
                ("critic", "finalize"),
                ("revise", "finalize"),
                ("finalize", "memory_manager"),
            }.issubset(edges)
        )
        self.assertIn(("failure", "memory_manager"), edges)

    def test_complete_draft_passes_without_revision(self):
        critic = StaticCriticModel(_pass_decision())
        revision = StaticRevisionModel("不应执行")
        app = build_supervisor_graph(
            RoleAwareModel(),
            critic_llm=critic,
            revision_llm=revision,
        )

        result = self._invoke(app, "critic-pass")

        self.assertEqual(result["critic_decision"]["verdict"], "PASS")
        self.assertEqual(result["critic_status"], "passed")
        self.assertEqual(result["revision_count"], 0)
        self.assertEqual(result["final_answer"], result["draft_answer"])
        self.assertEqual(len(critic.calls), 1)
        self.assertEqual(len(revision.calls), 0)
        draft_occurrences = [
            message
            for message in result["messages"]
            if isinstance(message, AIMessage)
            and message.content == result["draft_answer"]
        ]
        self.assertEqual(len(draft_occurrences), 1)

    def test_missing_budget_is_revised_from_existing_math_summary(self):
        critic = StaticCriticModel(
            _revise_decision(
                "草稿遗漏3天总预算",
                "使用Math Agent的1500元结果补充预算",
            )
        )
        revision = StaticRevisionModel(
            "北京旅游信息已整理；每天500元，3天总预算为1500元。"
        )
        app = build_supervisor_graph(
            RoleAwareModel(draft_mode="omit_budget"),
            critic_llm=critic,
            revision_llm=revision,
        )

        result = self._invoke(app, "critic-revise-missing")

        self.assertNotIn("1500", result["draft_answer"])
        self.assertIn("1500", result["final_answer"])
        self.assertEqual(result["revision_count"], 1)
        self.assertEqual(len(revision.calls), 1)
        revision_payload = json.loads(revision.calls[0][1].content)
        self.assertTrue(
            any(
                "1500" in str(item.get("result"))
                for item in revision_payload["subagent_results"]
            )
        )
        self.assertNotIn(
            result["draft_answer"],
            [
                message.content
                for message in result["messages"]
                if isinstance(message, AIMessage)
            ],
        )

    def test_budget_contradiction_is_visible_to_critic_and_corrected(self):
        critic = StaticCriticModel(
            _revise_decision(
                "草稿预算999元与Math Agent的1500元矛盾",
                "以Math Agent结果1500元为准",
            )
        )
        revision = StaticRevisionModel("3天总预算应为1500元。")
        app = build_supervisor_graph(
            RoleAwareModel(draft_mode="contradict_budget"),
            critic_llm=critic,
            revision_llm=revision,
        )

        result = self._invoke(app, "critic-contradiction")
        critic_payload = json.loads(critic.calls[0][1].content)

        self.assertIn("999", critic_payload["draft_answer"])
        self.assertTrue(
            any(
                "1500" in str(item.get("result"))
                for item in critic_payload["subagent_results"]
            )
        )
        self.assertEqual(result["final_answer"], "3天总预算应为1500元。")

    def test_revise_runs_at_most_once_without_second_critic(self):
        critic = StaticCriticModel(
            _revise_decision("持续要求修订", "再次修改")
        )
        revision = StaticRevisionModel("唯一一次修订后的答案")
        app = build_supervisor_graph(
            RoleAwareModel(),
            critic_llm=critic,
            revision_llm=revision,
        )

        result = self._invoke(app, "critic-limit")

        self.assertEqual(len(critic.calls), 1)
        self.assertEqual(len(revision.calls), 1)
        self.assertEqual(result["revision_count"], 1)
        self.assertEqual(result["max_revisions"], 1)
        self.assertEqual(result["final_answer"], "唯一一次修订后的答案")
        self.assertEqual(result["unresolved_critic_issues"], ["持续要求修订"])

    def test_critic_failure_falls_back_to_original_draft(self):
        critic = StaticCriticModel(error=ValueError("invalid critic JSON"))
        revision = StaticRevisionModel("不应执行")
        app = build_supervisor_graph(
            RoleAwareModel(),
            critic_llm=critic,
            revision_llm=revision,
        )

        with self.assertLogs(
            "ai_agent_learning.agents.critic",
            level="ERROR",
        ):
            result = self._invoke(app, "critic-failure")

        self.assertEqual(result["critic_status"], "failed")
        self.assertIn("invalid critic JSON", result["critic_error"])
        self.assertEqual(result["final_answer"], result["draft_answer"])
        self.assertEqual(len(revision.calls), 0)

    def test_memory_manager_runs_once_after_finalize(self):
        critic = StaticCriticModel(_pass_decision())
        revision = StaticRevisionModel("不应执行")
        with TemporaryDirectory() as directory:
            database = Path(directory) / "checkpoints.sqlite"
            config = {"configurable": {"thread_id": "critic-history"}}
            with open_sqlite_checkpointer(database) as checkpointer:
                app = build_supervisor_graph(
                    RoleAwareModel(),
                    checkpointer=checkpointer,
                    critic_llm=critic,
                    revision_llm=revision,
                )
                result = self._invoke(app, "critic-history")
                history = list(app.get_state_history(config))

        self.assertEqual(
            len(
                [
                    snapshot
                    for snapshot in history
                    if snapshot.next == ("memory_executor",)
                ]
            ),
            1,
        )
        self.assertTrue(
            any(snapshot.next == ("critic",) for snapshot in history)
        )
        self.assertTrue(
            any(snapshot.next == ("memory_manager",) for snapshot in history)
        )
        message_text = "\n".join(
            str(message.content) for message in result["messages"]
        )
        self.assertNotIn("草稿覆盖全部任务", message_text)


if __name__ == "__main__":
    unittest.main()
