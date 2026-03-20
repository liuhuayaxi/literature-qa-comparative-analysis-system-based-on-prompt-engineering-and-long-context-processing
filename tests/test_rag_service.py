from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from src.config import AppConfig
from src.memory_store import SessionMemoryStore, SQLiteMemoryStore
from src.retrieval_engine import RAGChatService, VectorStoreService, _resolve_prompt_override
from src.models import ChatTurn, SourceDocument


class RAGServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_resolve_prompt_override_uses_matching_language(self) -> None:
        self.assertEqual(
            _resolve_prompt_override({"zh": "中文提示词", "en": "English prompt"}, "zh"),
            "中文提示词",
        )
        self.assertEqual(
            _resolve_prompt_override({"zh": "中文提示词", "en": "English prompt"}, "en"),
            "English prompt",
        )
        self.assertIsNone(_resolve_prompt_override({"zh": "中文提示词", "en": ""}, "en"))

    async def test_answer_uses_retrieved_context_and_citations(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            config = AppConfig(
                openai_api_key="",
                openai_base_url="",
                chat_model="",
                embedding_model="",
                db_path=base / "app_state.db",
                vector_dir=base / "chroma",
            )
            vector_store = VectorStoreService(config)
            await vector_store.upsert_documents(
                [
                    SourceDocument(
                        page_content="RAG combines retrieval with generation and should cite evidence.",
                        metadata={
                            "course_id": "course-a",
                            "doc_id": "doc-1",
                            "file_name": "lecture1.txt",
                            "file_path": str(base / "lecture1.txt"),
                            "file_ext": "txt",
                            "source_type": "lecture",
                            "language": "en",
                            "chunk_id": "chunk-1",
                            "section_label": "intro",
                        },
                    ),
                    SourceDocument(
                        page_content="This chunk discusses unrelated chemistry content and should be ignored.",
                        metadata={
                            "course_id": "course-a",
                            "doc_id": "doc-2",
                            "file_name": "lecture2.txt",
                            "file_path": str(base / "lecture2.txt"),
                            "file_ext": "txt",
                            "source_type": "lecture",
                            "language": "en",
                            "chunk_id": "chunk-2",
                            "section_label": "chemistry",
                        },
                    ),
                ]
            )
            chat = RAGChatService(
                config=config,
                vector_store=vector_store,
                session_memory_store=SessionMemoryStore(),
                persistent_memory_store=SQLiteMemoryStore(base / "app_state.db"),
            )

            response = await chat.answer(
                session_id="session-1",
                course_id="course-a",
                question="What does RAG combine?",
                memory_mode="session",
                language="en",
                doc_ids=["doc-1"],
            )

            self.assertIn("retrieval", response.answer.lower())
            self.assertIn("Sources:", response.answer)
            self.assertEqual(response.session_id, "session-1")
            self.assertEqual(len(response.citations), 1)
            self.assertEqual(response.citations[0].doc_id, "doc-1")
            profile = await chat.session_memory_store.get_session_profile("session-1")
            self.assertEqual(profile["session_title"], "What does RAG combine?")
            self.assertGreater(int(profile.get("last_prompt_token_estimate", 0)), 0)
            self.assertEqual(int(profile.get("last_context_doc_count", 0)), 1)
            self.assertFalse(bool(profile.get("last_context_compressed", False)))
            self.assertGreaterEqual(int(profile.get("last_candidate_doc_count", 0)), 1)
            self.assertEqual(int(profile.get("last_rerank_kept_count", 0)), 1)
            self.assertEqual(profile.get("last_citation_doc_ids"), ["doc-1"])
            self.assertEqual(profile.get("last_selected_doc_ids"), ["doc-1"])

    async def test_answer_respects_citation_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            config = AppConfig(
                openai_api_key="",
                openai_base_url="",
                chat_model="",
                embedding_model="",
                db_path=base / "app_state.db",
                vector_dir=base / "chroma",
            )
            vector_store = VectorStoreService(config)
            await vector_store.upsert_documents(
                [
                    SourceDocument(
                        page_content="Transformers rely on attention mechanisms.",
                        metadata={
                            "course_id": "course-a",
                            "doc_id": "doc-1",
                            "file_name": "a.txt",
                            "file_path": str(base / "a.txt"),
                            "file_ext": "txt",
                            "source_type": "lecture",
                            "language": "en",
                            "chunk_id": "chunk-1",
                        },
                    ),
                    SourceDocument(
                        page_content="Attention helps models focus on relevant tokens.",
                        metadata={
                            "course_id": "course-a",
                            "doc_id": "doc-2",
                            "file_name": "b.txt",
                            "file_path": str(base / "b.txt"),
                            "file_ext": "txt",
                            "source_type": "lecture",
                            "language": "en",
                            "chunk_id": "chunk-2",
                        },
                    ),
                ]
            )
            chat = RAGChatService(
                config=config,
                vector_store=vector_store,
                session_memory_store=SessionMemoryStore(),
                persistent_memory_store=SQLiteMemoryStore(base / "app_state.db"),
            )

            response = await chat.answer(
                session_id="session-2",
                course_id="course-a",
                question="What is attention used for?",
                memory_mode="session",
                language="en",
                retrieval_top_k=2,
                retrieval_fetch_k=2,
                citation_limit=1,
            )

            self.assertEqual(len(response.citations), 1)

    async def test_persist_turns_updates_conversation_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            config = AppConfig(
                openai_api_key="",
                openai_base_url="",
                chat_model="",
                embedding_model="",
                recent_history_turns=2,
                history_summary_token_limit=120,
                db_path=base / "app_state.db",
                vector_dir=base / "chroma",
            )
            vector_store = VectorStoreService(config)
            await vector_store.upsert_documents(
                [
                    SourceDocument(
                        page_content="RAG combines retrieval with generation and cites evidence.",
                        metadata={
                            "course_id": "course-a",
                            "doc_id": "doc-1",
                            "file_name": "lecture1.txt",
                            "file_path": str(base / "lecture1.txt"),
                            "file_ext": "txt",
                            "source_type": "lecture",
                            "language": "en",
                            "chunk_id": "chunk-1",
                        },
                    )
                ]
            )
            memory_store = SessionMemoryStore()
            chat = RAGChatService(
                config=config,
                vector_store=vector_store,
                session_memory_store=memory_store,
                persistent_memory_store=SQLiteMemoryStore(base / "app_state.db"),
            )

            await chat.answer("session-3", "course-a", "What is RAG?", "session", language="en")
            await chat.answer("session-3", "course-a", "Why use citations?", "session", language="en")

            profile = await memory_store.get_session_profile("session-3")
            self.assertIn("conversation_summary", profile)
            self.assertTrue(profile["conversation_summary"])
            self.assertGreaterEqual(int(profile.get("turn_count", 0)), 4)

    async def test_stream_answer_rejects_invalid_or_sensitive_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            config = AppConfig(
                openai_api_key="",
                openai_base_url="",
                chat_model="",
                embedding_model="",
                db_path=base / "app_state.db",
                vector_dir=base / "chroma",
            )
            vector_store = VectorStoreService(config)
            chat = RAGChatService(
                config=config,
                vector_store=vector_store,
                session_memory_store=SessionMemoryStore(),
                persistent_memory_store=SQLiteMemoryStore(base / "app_state.db"),
            )

            empty_events = [
                event
                async for event in chat.stream_answer(
                    session_id="session-4",
                    course_id="course-a",
                    question="   ",
                    memory_mode="session",
                    language="zh",
                )
            ]
            sensitive_events = [
                event
                async for event in chat.stream_answer(
                    session_id="session-5",
                    course_id="course-a",
                    question="api_key=sk-sensitive-token-1234567890",
                    memory_mode="session",
                    language="zh",
                )
            ]

            self.assertEqual(empty_events[0]["type"], "error")
            self.assertIn("不能为空", str(empty_events[0]["content"]))
            self.assertEqual(sensitive_events[0]["type"], "error")
            self.assertIn("敏感信息", str(sensitive_events[0]["content"]))

    async def test_long_history_triggers_prompt_compression(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            config = AppConfig(
                openai_api_key="",
                openai_base_url="",
                chat_model="",
                embedding_model="",
                model_context_window=900,
                answer_token_reserve=300,
                recent_history_turns=8,
                prompt_compression_turn_token_limit=30,
                db_path=base / "app_state.db",
                vector_dir=base / "chroma",
            )
            vector_store = VectorStoreService(config)
            await vector_store.upsert_documents(
                [
                    SourceDocument(
                        page_content="RAG relies on retrieval and generation together.",
                        metadata={
                            "course_id": "course-a",
                            "doc_id": "doc-1",
                            "file_name": "lecture1.txt",
                            "file_path": str(base / "lecture1.txt"),
                            "file_ext": "txt",
                            "source_type": "lecture",
                            "language": "en",
                            "chunk_id": "chunk-1",
                        },
                    )
                ]
            )
            memory_store = SessionMemoryStore()
            for index in range(10):
                await memory_store.append_turn(
                    "session-6",
                    ChatTurn(
                        role="user" if index % 2 == 0 else "assistant",
                        content=("history turn " + str(index) + " ") * 60,
                        created_at=datetime.now(timezone.utc),
                    ),
                )
            chat = RAGChatService(
                config=config,
                vector_store=vector_store,
                session_memory_store=memory_store,
                persistent_memory_store=SQLiteMemoryStore(base / "app_state.db"),
            )

            response = await chat.answer(
                session_id="session-6",
                course_id="course-a",
                question="Explain RAG briefly",
                memory_mode="session",
                language="en",
            )

            profile = await memory_store.get_session_profile("session-6")
            self.assertTrue(response.answer)
            self.assertTrue(bool(profile.get("last_context_compressed", False)))
            self.assertIn("历史压缩", profile.get("last_context_strategies", []))


if __name__ == "__main__":
    unittest.main()
