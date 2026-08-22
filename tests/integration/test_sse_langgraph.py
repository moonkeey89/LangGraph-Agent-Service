import asyncio
import unittest
from typing import Annotated

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from ai_agent_learning.agent.context import AgentContext
from ai_agent_learning.api.service import AgentService


class MinimalStreamState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    session_user_id: str


class RealLangGraphSseTests(unittest.TestCase):
    def test_current_langgraph_messages_mode_emits_real_model_chunks(self):
        model = FakeListChatModel(responses=["真实片段"])

        def agent(state: MinimalStreamState):
            return {"messages": [model.invoke(state["messages"])]}

        builder = StateGraph(MinimalStreamState, context_schema=AgentContext)
        builder.add_node("agent", agent)
        builder.set_entry_point("agent")
        builder.add_edge("agent", END)
        graph = builder.compile(checkpointer=InMemorySaver())
        service = AgentService(graph)

        async def collect():
            return [
                event
                async for event in service.stream(
                    message="测试",
                    thread_id="real-langgraph-stream",
                    user_id="real-user",
                )
            ]

        events = asyncio.run(collect())

        self.assertEqual(events[0].event, "started")
        self.assertEqual(events[-1].event, "completed")
        self.assertEqual(events[-1].data["answer"], "真实片段")
        tokens = [
            event.data["content"]
            for event in events
            if event.event == "token"
        ]
        self.assertEqual("".join(tokens), "真实片段")
        snapshot = graph.get_state(
            {"configurable": {"thread_id": "real-langgraph-stream"}}
        )
        self.assertEqual(
            snapshot.values["messages"][-1].content,
            "真实片段",
        )
        self.assertIsInstance(snapshot.values["messages"][0], HumanMessage)
        self.assertEqual(snapshot.values["messages"][0].content, "测试")


if __name__ == "__main__":
    unittest.main()
