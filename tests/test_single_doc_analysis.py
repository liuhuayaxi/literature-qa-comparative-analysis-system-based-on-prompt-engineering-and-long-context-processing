from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.analysis_engine import SingleDocumentAnalysisService
from src.config import AppConfig
from src.retrieval_engine import VectorStoreService
from src.models import SourceDocument


class SingleDocumentAnalysisTests(unittest.IsolatedAsyncioTestCase):
    async def test_heuristic_analysis_returns_expected_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            config = AppConfig(
                openai_api_key="",
                openai_base_url="",
                chat_model="",
                embedding_model="",
                enable_result_cache=False,
                db_path=base / "state.db",
                vector_dir=base / "chroma",
            )
            vector_store = VectorStoreService(config)
            await vector_store.upsert_documents(
                [
                    SourceDocument(
                        page_content="Transformer models improve sequence modeling. A limitation is their memory cost.",
                        metadata={
                            "course_id": "course-a",
                            "doc_id": "doc-1",
                            "file_name": "paper.txt",
                            "file_path": str(base / "paper.txt"),
                            "file_ext": "txt",
                            "source_type": "paper",
                            "language": "en",
                            "chunk_id": "chunk-1",
                        },
                    )
                ]
            )
            service = SingleDocumentAnalysisService(config, vector_store)

            result = await service.analyze_document("course-a", "doc-1", output_language="en")

            self.assertEqual(result.doc_id, "doc-1")
            self.assertTrue(result.summary)
            self.assertTrue(result.keywords)
            self.assertTrue(result.main_topics)
            self.assertTrue(result.risk_points)

    async def test_long_document_uses_window_and_recursive_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            config = AppConfig(
                openai_api_key="x",
                openai_base_url="https://example.com/v1",
                chat_model="test-model",
                embedding_model="",
                model_context_window=2048,
                answer_token_reserve=512,
                long_context_window_tokens=80,
                long_context_window_overlap_tokens=20,
                recursive_summary_target_tokens=120,
                recursive_summary_batch_size=2,
                enable_result_cache=False,
                db_path=base / "state.db",
                vector_dir=base / "chroma",
            )
            vector_store = VectorStoreService(config)
            long_text = " ".join(f"segment{i}" for i in range(800))
            await vector_store.upsert_documents(
                [
                    SourceDocument(
                        page_content=long_text,
                        metadata={
                            "course_id": "course-a",
                            "doc_id": "doc-2",
                            "file_name": "long-paper.txt",
                            "file_path": str(base / "long-paper.txt"),
                            "file_ext": "txt",
                            "source_type": "paper",
                            "language": "en",
                            "chunk_id": "chunk-1",
                        },
                    )
                ]
            )
            service = SingleDocumentAnalysisService(config, vector_store)

            call_log: list[str] = []

            async def fake_invoke(_config, prompt: str, _language: str) -> str:
                call_log.append(prompt)
                if "sliding-window segment" in prompt:
                    return "- window summary"
                if "intermediate summaries" in prompt:
                    return "- merged summary"
                return (
                    '{"summary":"final summary","sentiment":"neutral","keywords":["rag"],'
                    '"main_topics":["context"],"risk_points":["token budget"]}'
                )

            with patch("src.analysis_engine.invoke_chat_text", new=fake_invoke):
                result = await service.analyze_document("course-a", "doc-2", output_language="en")

            self.assertEqual(result.summary, "final summary")
            self.assertGreaterEqual(sum("sliding-window segment" in prompt for prompt in call_log), 2)
            self.assertGreaterEqual(sum("intermediate summaries" in prompt for prompt in call_log), 1)

    async def test_analysis_cache_reuses_same_content_with_new_doc_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            file_path = base / "paper.txt"
            file_path.write_text("Transformer models improve sequence modeling.", encoding="utf-8")
            config = AppConfig(
                openai_api_key="",
                openai_base_url="",
                chat_model="",
                embedding_model="",
                enable_result_cache=True,
                db_path=base / "state.db",
                vector_dir=base / "chroma",
                cache_dir=base / "cache",
            )
            vector_store = VectorStoreService(config)
            await vector_store.upsert_documents(
                [
                    SourceDocument(
                        page_content="Transformer models improve sequence modeling.",
                        metadata={
                            "course_id": "course-a",
                            "doc_id": "doc-1",
                            "file_name": "paper.txt",
                            "file_path": str(file_path),
                            "file_ext": "txt",
                            "source_type": "paper",
                            "language": "en",
                            "chunk_id": "chunk-1",
                        },
                    ),
                    SourceDocument(
                        page_content="Transformer models improve sequence modeling.",
                        metadata={
                            "course_id": "course-a",
                            "doc_id": "doc-2",
                            "file_name": "paper.txt",
                            "file_path": str(file_path),
                            "file_ext": "txt",
                            "source_type": "paper",
                            "language": "en",
                            "chunk_id": "chunk-2",
                        },
                    ),
                ]
            )
            service = SingleDocumentAnalysisService(config, vector_store)

            with patch.object(service, "_heuristic_analysis", wraps=service._heuristic_analysis) as mocked:
                await service.analyze_document("course-a", "doc-1", output_language="en")
                await service.analyze_document("course-a", "doc-2", output_language="en")

            self.assertEqual(mocked.call_count, 1)


if __name__ == "__main__":
    unittest.main()
