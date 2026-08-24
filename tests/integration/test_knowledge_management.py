import io
import unittest
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from ai_agent_learning.api.app import create_app
from ai_agent_learning.api.service import AgentService
from tests.helpers import install_test_identity
from ai_agent_learning.knowledge import (
    ChromaKnowledgeRepository,
    KnowledgeCatalog,
    KnowledgeIngestor,
    KnowledgeLibraryService,
    KnowledgeRetriever,
    UploadCandidate,
)
from tests.helpers import DeterministicTestEmbeddings
from tests.unit.test_knowledge import _minimal_pdf


class ManagementGraph:
    def __init__(self):
        self.contexts = []

    def get_state(self, _config):
        return SimpleNamespace(values={}, interrupts=())

    def invoke(self, graph_input, *, config, context):
        self.contexts.append(context)
        answer = f"回答：{graph_input['messages'][0].content}"
        return {"messages": [AIMessage(content=answer)], "final_answer": answer}


class KnowledgeManagementTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.catalog = KnowledgeCatalog(self.root / "catalog.sqlite")
        self.repository = ChromaKnowledgeRepository(
            persist_directory=self.root / "chroma",
            embeddings=DeterministicTestEmbeddings(),
        )
        self.ingestor = KnowledgeIngestor(
            self.repository,
            chunk_size=80,
            chunk_overlap=10,
        )
        self.service = KnowledgeLibraryService(
            catalog=self.catalog,
            repository=self.repository,
            ingestor=self.ingestor,
            source_directory=self.root / "sources",
            max_file_size_bytes=1024,
            max_files_per_upload=5,
        )
        self.graph = ManagementGraph()

        @contextmanager
        def service_factory():
            yield AgentService(
                self.graph,
                knowledge_service=self.service,
            )

        self.client_context = TestClient(
            install_test_identity(create_app(service_factory)),
            raise_server_exceptions=False,
        )
        self.client = self.client_context.__enter__()

    def tearDown(self):
        self.client_context.__exit__(None, None, None)
        self.repository.close()
        self.catalog.close()
        self.temporary.cleanup()

    @staticmethod
    def headers(user_id="user_001"):
        return {"X-User-ID": user_id}

    def create_base(self, name="项目资料", user_id="user_001"):
        response = self.client.post(
            "/api/v1/knowledge-bases",
            headers=self.headers(user_id),
            json={"name": name, "description": "测试知识库"},
        )
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()

    def upload(self, base_id, filename, content, content_type, user_id="user_001"):
        return self.client.post(
            f"/api/v1/knowledge-bases/{base_id}/documents",
            headers=self.headers(user_id),
            files={"files": (filename, content, content_type)},
        )

    def test_create_list_get_delete_and_owner_isolation(self):
        base = self.create_base()
        base_id = base["knowledge_base_id"]
        self.assertEqual(base["owner_user_id"], "user_001")
        self.assertEqual(
            self.client.get(
                "/api/v1/knowledge-bases", headers=self.headers()
            ).json()[0]["knowledge_base_id"],
            base_id,
        )
        for method, path in [
            ("get", f"/api/v1/knowledge-bases/{base_id}"),
            ("delete", f"/api/v1/knowledge-bases/{base_id}"),
        ]:
            response = getattr(self.client, method)(
                path, headers=self.headers("user_002")
            )
            self.assertEqual(response.status_code, 404)
        self.assertEqual(
            self.client.get(
                f"/api/v1/knowledge-bases/{base_id}/documents",
                headers=self.headers("user_002"),
            ).status_code,
            404,
        )
        self.assertEqual(
            self.upload(
                base_id,
                "forbidden.txt",
                b"forbidden",
                "text/plain",
                user_id="user_002",
            ).status_code,
            404,
        )
        deleted = self.client.delete(
            f"/api/v1/knowledge-bases/{base_id}", headers=self.headers()
        )
        self.assertEqual(deleted.status_code, 204)

    def test_txt_markdown_pdf_upload_and_metadata(self):
        base_id = self.create_base()["knowledge_base_id"]
        samples = [
            ("a.txt", b"TXT UNIQUE FACT", "text/plain"),
            ("b.md", b"# MARKDOWN UNIQUE FACT", "text/markdown"),
            ("c.pdf", _minimal_pdf("PDF-UNIQUE-9001"), "application/pdf"),
        ]
        for filename, content, content_type in samples:
            response = self.upload(
                base_id, filename, content, content_type
            )
            self.assertEqual(response.status_code, 200, response.text)
            document = response.json()["items"][0]["document"]
            self.assertEqual(document["status"], "ready")
            self.assertGreater(document["chunk_count"], 0)
            self.assertEqual(document["original_filename"], filename)
        documents = self.client.get(
            f"/api/v1/knowledge-bases/{base_id}/documents",
            headers=self.headers(),
        ).json()
        self.assertEqual(len(documents), 3)
        self.assertNotIn("stored_filename", documents[0])

    def test_invalid_type_signature_extension_and_size_are_rejected(self):
        base_id = self.create_base()["knowledge_base_id"]
        cases = [
            ("bad.exe", b"unsafe", "application/octet-stream"),
            ("fake.pdf", b"not-pdf", "application/pdf"),
            ("fake.txt", b"plain", "application/pdf"),
            ("large.txt", b"x" * 1025, "text/plain"),
        ]
        for filename, content, content_type in cases:
            response = self.upload(
                base_id, filename, content, content_type
            )
            self.assertEqual(response.status_code, 422, response.text)

    def test_path_traversal_name_is_only_display_metadata(self):
        base_id = self.create_base()["knowledge_base_id"]
        response = self.upload(
            base_id,
            "../../escaped.txt",
            b"safe controlled content",
            "text/plain",
        )
        self.assertEqual(response.status_code, 200, response.text)
        document = response.json()["items"][0]["document"]
        self.assertEqual(document["original_filename"], "escaped.txt")
        record = self.catalog.get_document(document["document_id"])
        stored = (self.root / "sources" / record.stored_filename).resolve()
        self.assertTrue(stored.is_relative_to((self.root / "sources").resolve()))

    def test_duplicate_upload_does_not_duplicate_chunks(self):
        base_id = self.create_base()["knowledge_base_id"]
        first = self.upload(base_id, "first.txt", b"same fact", "text/plain")
        second = self.upload(base_id, "renamed.txt", b"same fact", "text/plain")
        first_item = first.json()["items"][0]
        second_item = second.json()["items"][0]
        self.assertFalse(first_item["duplicate"])
        self.assertTrue(second_item["duplicate"])
        self.assertEqual(
            first_item["document"]["document_id"],
            second_item["document"]["document_id"],
        )
        self.assertEqual(
            self.repository.count(knowledge_base_id=base_id),
            first_item["document"]["chunk_count"],
        )
        changed = self.upload(
            base_id,
            "first.txt",
            b"different content",
            "text/plain",
        ).json()["items"][0]
        self.assertFalse(changed["duplicate"])
        self.assertNotEqual(
            changed["document"]["document_id"],
            first_item["document"]["document_id"],
        )

    def test_upload_file_count_limit_is_enforced(self):
        base_id = self.create_base()["knowledge_base_id"]
        files = [
            ("files", (f"{index}.txt", b"content", "text/plain"))
            for index in range(6)
        ]
        response = self.client.post(
            f"/api/v1/knowledge-bases/{base_id}/documents",
            headers=self.headers(),
            files=files,
        )
        self.assertEqual(response.status_code, 422)

    def test_failed_index_rolls_back_chunks_and_source_but_keeps_status(self):
        base = self.service.create_knowledge_base(
            owner_user_id="failure-user",
            name="failure",
        )

        class PartialFailureIngestor:
            def __init__(self, repository):
                self.repository = repository

            def ingest_file(self, path, *, knowledge_base_id, document_id, source_name):
                from ai_agent_learning.knowledge.models import KnowledgeChunk

                self.repository.replace_document(
                    [
                        KnowledgeChunk(
                            content="partial",
                            knowledge_base_id=knowledge_base_id,
                            document_id=document_id,
                            source=source_name,
                            page=None,
                            chunk_id="chunk-partial",
                        )
                    ]
                )
                raise RuntimeError("private path C:/secret")

        failing = KnowledgeLibraryService(
            catalog=self.catalog,
            repository=self.repository,
            ingestor=PartialFailureIngestor(self.repository),
            source_directory=self.root / "sources",
            max_file_size_bytes=1024,
            max_files_per_upload=5,
        )
        result = failing.upload_documents(
            knowledge_base_id=base.knowledge_base_id,
            owner_user_id="failure-user",
            uploads=[UploadCandidate("bad.txt", "text/plain", io.BytesIO(b"content"))],
        )[0].document
        self.assertEqual(result.status, "failed")
        self.assertNotIn("C:/secret", result.error_message)
        self.assertEqual(
            self.repository.count(knowledge_base_id=base.knowledge_base_id), 0
        )
        self.assertFalse(any((self.root / "sources" / base.knowledge_base_id).glob("*")))

    def test_retriever_only_uses_ready_catalog_documents(self):
        base = self.service.create_knowledge_base(
            owner_user_id="ready-user",
            name="ready-only",
        )
        from ai_agent_learning.knowledge.models import KnowledgeChunk

        self.repository.replace_document(
            [
                KnowledgeChunk(
                    content="PROCESSING PRIVATE FACT",
                    knowledge_base_id=base.knowledge_base_id,
                    document_id="doc-processing",
                    source="processing.txt",
                    page=None,
                    chunk_id="chunk-processing",
                )
            ]
        )
        retriever = KnowledgeRetriever(
            self.repository,
            relevance_threshold=None,
            ready_document_ids=self.catalog.ready_document_ids,
        )
        response = retriever.search(
            query="PROCESSING PRIVATE FACT",
            knowledge_base_id=base.knowledge_base_id,
        )
        self.assertEqual(response.status, "no_evidence")

    def test_delete_document_and_base_remove_only_target_chunks(self):
        first = self.create_base("first")
        second = self.create_base("second")
        first_doc = self.upload(
            first["knowledge_base_id"], "first.txt", b"FIRST FACT", "text/plain"
        ).json()["items"][0]["document"]
        self.upload(
            second["knowledge_base_id"], "second.txt", b"SECOND FACT", "text/plain"
        )
        response = self.client.delete(
            f"/api/v1/knowledge-bases/{first['knowledge_base_id']}/documents/{first_doc['document_id']}",
            headers=self.headers(),
        )
        self.assertEqual(response.status_code, 204)
        self.assertEqual(
            self.repository.count(knowledge_base_id=first["knowledge_base_id"]), 0
        )
        retriever = KnowledgeRetriever(
            self.repository,
            relevance_threshold=None,
            ready_document_ids=self.catalog.ready_document_ids,
        )
        self.assertEqual(
            retriever.search(
                query="FIRST FACT",
                knowledge_base_id=first["knowledge_base_id"],
            ).status,
            "no_evidence",
        )
        self.assertGreater(
            self.repository.count(knowledge_base_id=second["knowledge_base_id"]), 0
        )
        self.client.delete(
            f"/api/v1/knowledge-bases/{first['knowledge_base_id']}",
            headers=self.headers(),
        )
        self.assertGreater(
            self.repository.count(knowledge_base_id=second["knowledge_base_id"]), 0
        )

    def test_chat_selection_is_revalidated_and_passed_as_trusted_context(self):
        base = self.create_base()
        request = {
            "message": "根据文档回答",
            "thread_id": "managed-rag-thread",
            "knowledge_base_id": base["knowledge_base_id"],
        }
        allowed = self.client.post(
            "/api/v1/agent/invoke",
            headers=self.headers(),
            json=request,
        )
        self.assertEqual(allowed.status_code, 200, allowed.text)
        self.assertEqual(
            self.graph.contexts[-1].knowledge_base_id,
            base["knowledge_base_id"],
        )
        forbidden = self.client.post(
            "/api/v1/agent/invoke",
            headers=self.headers("user_002"),
            json={**request, "thread_id": "other-thread"},
        )
        self.assertEqual(forbidden.status_code, 404)

    def test_catalog_sources_and_chroma_survive_restart(self):
        base_id = self.create_base()["knowledge_base_id"]
        self.upload(base_id, "persistent.txt", b"PERSISTENT FACT", "text/plain")
        self.repository.close()
        self.catalog.close()
        reopened_catalog = KnowledgeCatalog(self.root / "catalog.sqlite")
        with ChromaKnowledgeRepository(
            persist_directory=self.root / "chroma",
            embeddings=DeterministicTestEmbeddings(),
        ) as reopened_repository:
            retriever = KnowledgeRetriever(
                reopened_repository,
                relevance_threshold=None,
                ready_document_ids=reopened_catalog.ready_document_ids,
            )
            self.assertEqual(
                reopened_catalog.list_knowledge_bases("user_001")[0].knowledge_base_id,
                base_id,
            )
            self.assertEqual(
                retriever.search(query="PERSISTENT FACT", knowledge_base_id=base_id).status,
                "found",
            )
        reopened_catalog.close()
        self.repository = ChromaKnowledgeRepository(
            persist_directory=self.root / "chroma",
            embeddings=DeterministicTestEmbeddings(),
        )
        self.catalog = KnowledgeCatalog(self.root / "catalog.sqlite")

    def test_cli_entry_and_web_share_the_same_library_service(self):
        source = self.root / "cli.md"
        source.write_text("CLI SHARED INGESTION FACT", encoding="utf-8")
        result = self.service.ingest_cli_file(
            path=source,
            knowledge_base_id="cli_shared",
            owner_user_id="cli-user",
        )
        self.assertEqual(result.document.status, "ready")
        listed = self.catalog.list_documents("cli_shared")
        self.assertEqual(listed[0].document_id, result.document.document_id)
        self.assertTrue(
            (self.root / "sources" / listed[0].stored_filename).is_file()
        )
