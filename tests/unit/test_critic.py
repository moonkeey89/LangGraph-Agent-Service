import json
import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ai_agent_learning.agents.critic import (
    CriticDecision,
    build_critic_context,
    route_after_critic,
)


class CriticUnitTests(unittest.TestCase):
    def test_decision_schema_has_review_fields_and_no_identity(self):
        properties = CriticDecision.model_json_schema()["properties"]

        self.assertEqual(
            set(properties),
            {"verdict", "issues", "suggestions", "severity", "reason"},
        )
        self.assertNotIn("user_id", properties)
        self.assertNotIn("thread_id", properties)

    def test_context_contains_only_current_bounded_inputs(self):
        state = {
            "messages": [
                HumanMessage(content="旧问题：我的密码是secret"),
                AIMessage(content="旧回答"),
                HumanMessage(
                    content=(
                        "查询北京天气和景点，如果每天预算500元，"
                        "计算3天总预算"
                    )
                ),
                ToolMessage(
                    content=json.dumps(
                        {
                            "agent_name": "math_agent",
                            "status": "success",
                            "result": "500*3=1500",
                            "error": None,
                            "retry_recommended": False,
                            "internal_trace": "不应传递",
                        },
                        ensure_ascii=False,
                    ),
                    name="ask_math_agent",
                    tool_call_id="math-1",
                ),
            ],
            "draft_answer": "北京旅游信息和预算草稿",
            "recalled_memories": [
                {"content": "无关长期记忆", "user_id": "user-001"}
            ],
        }

        context = build_critic_context(state)
        encoded = json.dumps(context, ensure_ascii=False)

        self.assertEqual(
            set(context),
            {
                "user_request",
                "draft_answer",
                "subagent_results",
                "user_constraints",
            },
        )
        self.assertIn("500*3=1500", encoded)
        self.assertIn("每天预算500元", encoded)
        self.assertNotIn("secret", encoded)
        self.assertNotIn("internal_trace", encoded)
        self.assertNotIn("无关长期记忆", encoded)
        self.assertNotIn("user-001", encoded)

    def test_revision_route_honors_fixed_limit(self):
        decision = CriticDecision(
            verdict="REVISE",
            issues=["遗漏预算"],
            suggestions=["补充预算"],
            severity="medium",
            reason="信息不完整",
        ).model_dump()

        self.assertEqual(
            route_after_critic(
                {
                    "messages": [],
                    "critic_status": "revision_required",
                    "critic_decision": decision,
                    "revision_count": 0,
                    "max_revisions": 1,
                }
            ),
            "revise",
        )
        self.assertEqual(
            route_after_critic(
                {
                    "messages": [],
                    "critic_status": "revision_required",
                    "critic_decision": decision,
                    "revision_count": 1,
                    "max_revisions": 1,
                }
            ),
            "finalize",
        )


if __name__ == "__main__":
    unittest.main()
