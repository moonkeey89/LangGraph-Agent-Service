import json
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event

import anyio
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessageChunk

from ai_agent_learning.api.app import create_app
from ai_agent_learning.api.service import AgentService
from ai_agent_learning.knowledge.models import KnowledgeChunk
from ai_agent_learning.knowledge.service import KnowledgeNotFoundError
from ai_agent_learning.research import (
    ResearchCatalog,
    ResearchExecutionService,
    ResearchService,
)


TERMINALS = {
    "run_completed",
    "run_blocked",
    "run_needs_review",
    "run_failed",
}


class KnowledgeStub:
    def ensure_owned(self, knowledge_base_id, owner_user_id):
        if knowledge_base_id != "kb_owned" or owner_user_id != "user_001":
            raise KnowledgeNotFoundError

    def get_ready_chunk(self, *, knowledge_base_id, owner_user_id, chunk_id):
        self.ensure_owned(knowledge_base_id, owner_user_id)
        if chunk_id != "chunk_001":
            raise KnowledgeNotFoundError
        return KnowledgeChunk(
            content="真实科研证据 RF-STREAM-001",
            knowledge_base_id="kb_owned",
            document_id="doc_001",
            source="safe-evidence.md",
            page=3,
            chunk_id="chunk_001",
        )


class StreamingResearchGraph:
    def __init__(
        self,
        *,
        outcome="completed",
        answer="流式科研答案",
        tokens=True,
        with_evidence=True,
        error=None,
        release: Event | None = None,
    ):
        self.outcome = outcome
        self.answer = answer
        self.tokens = tokens
        self.with_evidence = with_evidence
        self.error = error
        self.release = release
        self.stream_calls = 0
        self.entered = Event()
        self.finished = Event()

    def stream(
        self,
        state,
        *,
        config,
        context,
        stream_mode,
        subgraphs,
        version,
    ):
        self.stream_calls += 1
        assert stream_mode == ["updates", "messages"]
        assert subgraphs is False
        assert version == "v2"
        assert context.user_id == state["session_user_id"]
        assert config["configurable"]["thread_id"].startswith("research-run-")
        self.entered.set()
        if self.release is not None:
            self.release.wait(timeout=5)
        try:
            if self.error is not None:
                raise self.error
            for node in (
                "research_validate_binding",
                "research_validate_binding",
                "research_supervisor",
            ):
                yield {"type": "updates", "data": {node: {}}}
            yield {
                "type": "updates",
                "data": {
                    "private_internal_node": {
                        "prompt": "SYSTEM SECRET",
                        "other_user": "private",
                    }
                },
            }
            sources = []
            if self.with_evidence:
                sources = [
                    {
                        "knowledge_base_id": "kb_owned",
                        "document_id": "doc_001",
                        "chunk_id": "chunk_001",
                        "source": "safe-evidence.md",
                        "page": 3,
                        "excerpt": "完整证据不应通过SSE发送",
                    }
                ]
                yield {
                    "type": "updates",
                    "data": {"research_evidence_agent": {"sources": sources}},
                }
            if self.tokens:
                split = max(1, len(self.answer) // 2)
                for fragment in (self.answer[:split], self.answer[split:]):
                    if fragment:
                        yield {
                            "type": "messages",
                            "data": (
                                AIMessageChunk(content=fragment),
                                {"langgraph_node": "research_synthesize"},
                            ),
                        }
                yield {
                    "type": "messages",
                    "data": (
                        AIMessageChunk(content="CRITIC INTERNAL"),
                        {"langgraph_node": "research_critic"},
                    ),
                }
            for node in (
                "research_synthesize",
                "research_critic",
                "research_finalize",
            ):
                update = {}
                if node == "research_finalize":
                    update = {
                        "outcome": self.outcome,
                        "final_answer": (
                            self.answer
                            if self.outcome in {"completed", "needs_review"}
                            else ""
                        ),
                        "sources": sources,
                        "unresolved_issues": (
                            ["需要人工确认"]
                            if self.outcome == "needs_review"
                            else []
                        ),
                        "error": (
                            "缺少证据"
                            if self.outcome == "blocked"
                            else "安全失败说明"
                            if self.outcome == "failed"
                            else None
                        ),
                    }
                yield {"type": "updates", "data": {node: update}}
        finally:
            self.finished.set()


def parse_sse(body: str) -> list[dict[str, object]]:
    events = []
    for block in body.replace("\r\n", "\n").split("\n\n"):
        if not block.strip() or block.startswith(":"):
            continue
        event_name = "message"
        data_lines = []
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


class ResearchSseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.catalog = ResearchCatalog(
            Path(self.temporary.name) / "research.sqlite"
        )
        self.research = ResearchService(self.catalog, KnowledgeStub())
        self.project = self.research.create_project(
            owner_user_id="user_001",
            name="SSE项目",
            status="active",
            default_knowledge_base_id="kb_owned",
        )

    def tearDown(self):
        self.catalog.close()
        self.temporary.cleanup()

    def create_task(self, title="SSE任务"):
        return self.research.create_task(
            self.project.project_id,
            "user_001",
            title=title,
            acceptance_criteria=["引用证据"],
        )

    def client(self, graph):
        execution = ResearchExecutionService(self.research, graph)

        @contextmanager
        def factory():
            yield AgentService(
                graph,
                research_service=self.research,
                research_execution_service=execution,
            )

        return TestClient(create_app(factory), raise_server_exceptions=False)

    def stream(self, graph, task, *, user_id="user_001"):
        path = (
            f"/api/v1/research/projects/{self.project.project_id}"
            f"/tasks/{task.task_id}/runs/stream"
        )
        with self.client(graph) as client:
            with client.stream(
                "POST",
                path,
                headers={"X-User-ID": user_id},
            ) as response:
                body = "".join(response.iter_text())
                content_type = response.headers.get("content-type", "")
                cache = response.headers.get("cache-control")
        return response.status_code, content_type, cache, parse_sse(body)

    def test_protocol_progress_evidence_tokens_and_completed(self):
        task = self.create_task()
        graph = StreamingResearchGraph()
        status, content_type, cache, events = self.stream(graph, task)
        self.assertEqual(status, 200)
        self.assertTrue(content_type.startswith("text/event-stream"))
        self.assertEqual(cache, "no-cache")
        self.assertEqual(events[0]["event"], "run_started")
        self.assertEqual(events[1]["event"], "task_status")
        progress = [item for item in events if item["event"] == "agent_progress"]
        self.assertEqual(
            [item["data"]["stage"] for item in progress],
            ["validate_binding", "planning", "evidence", "synthesize", "critic", "finalize"],
        )
        evidence = [item for item in events if item["event"] == "evidence_found"]
        self.assertEqual(evidence[0]["data"]["count"], 1)
        self.assertEqual(
            evidence[0]["data"]["sources"],
            [{"source": "safe-evidence.md", "page": 3}],
        )
        serialized = json.dumps(events, ensure_ascii=False)
        self.assertNotIn("完整证据", serialized)
        self.assertNotIn("SYSTEM SECRET", serialized)
        self.assertNotIn("CRITIC INTERNAL", serialized)
        tokens = [
            item["data"]["content"]
            for item in events
            if item["event"] == "token"
        ]
        self.assertEqual("".join(tokens), graph.answer)
        self.assertEqual(graph.stream_calls, 1)
        terminals = [item for item in events if item["event"] in TERMINALS]
        self.assertEqual(len(terminals), 1)
        self.assertEqual(terminals[0]["event"], "run_completed")
        self.assertEqual(events[-1]["event"], "run_completed")
        self.assertEqual(
            len(self.catalog.list_runs(task.task_id)),
            1,
        )
        artifacts = self.catalog.list_artifacts(self.project.project_id)
        self.assertEqual(len(artifacts), 1)
        self.assertEqual(artifacts[0].created_by, "agent")

    def test_no_real_message_chunk_means_no_token_event(self):
        task = self.create_task("无token")
        *_, events = self.stream(
            StreamingResearchGraph(tokens=False),
            task,
        )
        self.assertFalse(any(item["event"] == "token" for item in events))
        self.assertEqual(events[-1]["event"], "run_completed")

    def test_each_outcome_has_exactly_one_matching_terminal(self):
        expected = {
            "completed": "run_completed",
            "blocked": "run_blocked",
            "needs_review": "run_needs_review",
            "failed": "run_failed",
        }
        for outcome, terminal in expected.items():
            with self.subTest(outcome=outcome):
                task = self.create_task(outcome)
                *_, events = self.stream(
                    StreamingResearchGraph(
                        outcome=outcome,
                        with_evidence=False,
                    ),
                    task,
                )
                terminals = [item for item in events if item["event"] in TERMINALS]
                self.assertEqual([item["event"] for item in terminals], [terminal])
                self.assertEqual(events[-1]["event"], terminal)
                run = self.catalog.list_runs(task.task_id)[0]
                persisted_task = self.catalog.get_task(
                    self.project.project_id, task.task_id
                )
                self.assertEqual(run.outcome, outcome)
                self.assertNotEqual(persisted_task.status, "running")

    def test_non_streaming_and_streaming_share_terminal_semantics(self):
        stream_task = self.create_task("stream")
        normal_task = self.create_task("normal")
        stream_graph = StreamingResearchGraph(with_evidence=False)
        normal_graph = StreamingResearchGraph(with_evidence=False)
        *_, events = self.stream(stream_graph, stream_task)
        result = ResearchExecutionService(
            self.research, normal_graph
        ).execute_task(
            project_id=self.project.project_id,
            task_id=normal_task.task_id,
            user_id="user_001",
        )
        terminal = events[-1]["data"]
        self.assertEqual(terminal["status"], result.status)
        self.assertEqual(terminal["outcome"], result.outcome)
        self.assertEqual(stream_graph.stream_calls, 1)
        self.assertEqual(normal_graph.stream_calls, 1)

    def test_cross_user_receives_safe_rejection_and_creates_no_run(self):
        task = self.create_task("foreign")
        *_, events = self.stream(
            StreamingResearchGraph(),
            task,
            user_id="user_002",
        )
        self.assertEqual([item["event"] for item in events], ["run_failed"])
        self.assertEqual(events[0]["data"]["code"], "request_rejected")
        self.assertNotIn("run_id", events[0]["data"])
        self.assertEqual(self.catalog.list_runs(task.task_id), [])

    def test_graph_error_is_sanitized_and_durably_failed(self):
        task = self.create_task("error")
        graph = StreamingResearchGraph(
            error=RuntimeError(
                "API_KEY=secret; E:/private/checkpoints.sqlite; traceback"
            )
        )
        *_, events = self.stream(graph, task)
        terminal = events[-1]
        self.assertEqual(terminal["event"], "run_failed")
        serialized = json.dumps(terminal, ensure_ascii=False)
        self.assertNotIn("secret", serialized)
        self.assertNotIn("checkpoints.sqlite", serialized)
        run = self.catalog.list_runs(task.task_id)[0]
        self.assertEqual((run.status, run.outcome), ("failed", "failed"))

    def test_closed_consumer_does_not_leave_run_running(self):
        task = self.create_task("disconnect")
        release = Event()
        graph = StreamingResearchGraph(
            with_evidence=False,
            release=release,
        )
        service = ResearchExecutionService(self.research, graph)

        async def consume_one_then_close():
            stream = service.execute_stream(
                project_id=self.project.project_id,
                task_id=task.task_id,
                user_id="user_001",
            )
            first = await anext(stream)
            self.assertEqual(first.event, "run_started")
            await stream.aclose()

        anyio.run(consume_one_then_close)
        release.set()
        self.assertTrue(graph.finished.wait(timeout=5))
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            run = self.catalog.list_runs(task.task_id)[0]
            if run.status != "running":
                break
            time.sleep(0.02)
        run = self.catalog.list_runs(task.task_id)[0]
        persisted_task = self.catalog.get_task(self.project.project_id, task.task_id)
        self.assertEqual((run.status, run.outcome), ("completed", "completed"))
        self.assertEqual(persisted_task.status, "completed")
        self.assertEqual(len(self.catalog.list_artifacts(self.project.project_id)), 1)

    def test_consumer_serialization_failure_does_not_skip_finalization(self):
        task = self.create_task("serialization")
        release = Event()
        graph = StreamingResearchGraph(
            with_evidence=False,
            release=release,
        )
        service = ResearchExecutionService(self.research, graph)

        async def fail_while_serializing_first_event():
            stream = service.execute_stream(
                project_id=self.project.project_id,
                task_id=task.task_id,
                user_id="user_001",
            )
            try:
                event = await anext(stream)
                self.assertEqual(event.event, "run_started")
                raise TypeError("simulated SSE serialization failure")
            except TypeError:
                pass
            finally:
                await stream.aclose()

        anyio.run(fail_while_serializing_first_event)
        release.set()
        self.assertTrue(graph.finished.wait(timeout=5))
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            run = self.catalog.list_runs(task.task_id)[0]
            if run.status != "running":
                break
            time.sleep(0.02)
        run = self.catalog.list_runs(task.task_id)[0]
        self.assertEqual((run.status, run.outcome), ("completed", "completed"))
        self.assertEqual(
            len(
                self.catalog.list_artifacts(
                    self.project.project_id,
                    task_id=task.task_id,
                )
            ),
            1,
        )


if __name__ == "__main__":
    unittest.main()
