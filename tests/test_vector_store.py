from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.config import AppConfig
from src.retrieval_engine import VectorStoreService
from src.models import SourceDocument


class VectorStoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_similarity_search_accepts_custom_k_and_fetch_k(self) -> None:
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
            store = VectorStoreService(config)
            await store.upsert_documents(
                [
                    SourceDocument(
                        page_content="Transformer models use attention.",
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
                        page_content="Attention helps select relevant tokens.",
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

            results = await store.similarity_search(
                course_id="course-a",
                query="What is attention?",
                k=1,
                fetch_k=2,
            )

            self.assertEqual(len(results), 1)

    async def test_local_retriever_respects_custom_k(self) -> None:
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
            store = VectorStoreService(config)
            await store.upsert_documents(
                [
                    SourceDocument(
                        page_content="The syllabus discusses transformers.",
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
                        page_content="The assignment covers attention.",
                        metadata={
                            "course_id": "course-a",
                            "doc_id": "doc-2",
                            "file_name": "b.txt",
                            "file_path": str(base / "b.txt"),
                            "file_ext": "txt",
                            "source_type": "assignment",
                            "language": "en",
                            "chunk_id": "chunk-2",
                        },
                    ),
                ]
            )

            retriever = store.get_retriever(course_id="course-a", k=1, fetch_k=2)
            results = await retriever.aget_relevant_documents("attention")

            self.assertEqual(len(results), 1)

    async def test_rerank_documents_filters_low_scores_and_keeps_best_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            config = AppConfig(
                openai_api_key="",
                openai_base_url="",
                chat_model="",
                embedding_model="",
                retrieval_top_k=2,
                retrieval_fetch_k=5,
                rerank_min_score=0.35,
                rerank_min_keep=1,
                db_path=base / "app_state.db",
                vector_dir=base / "chroma",
            )
            store = VectorStoreService(config)
            await store.upsert_documents(
                [
                    SourceDocument(
                        page_content="RAG combines retrieval with generation and cites evidence from documents.",
                        metadata={
                            "course_id": "course-a",
                            "doc_id": "doc-1",
                            "file_name": "lecture-rag.txt",
                            "file_path": str(base / "lecture-rag.txt"),
                            "file_ext": "txt",
                            "source_type": "lecture",
                            "language": "en",
                            "chunk_id": "chunk-1",
                        },
                    ),
                    SourceDocument(
                        page_content="This section discusses unrelated chemistry experiments.",
                        metadata={
                            "course_id": "course-a",
                            "doc_id": "doc-2",
                            "file_name": "chemistry.txt",
                            "file_path": str(base / "chemistry.txt"),
                            "file_ext": "txt",
                            "source_type": "lecture",
                            "language": "en",
                            "chunk_id": "chunk-2",
                        },
                    ),
                ]
            )

            recalled = await store.recall_documents(
                course_id="course-a",
                query="What does RAG combine?",
                candidate_k=5,
                fetch_k=5,
            )
            rerank = store.rerank_documents(
                "What does RAG combine?",
                recalled,
                top_k=2,
                min_score=0.35,
                min_keep=1,
            )

            self.assertEqual(len(rerank.candidates), 2)
            self.assertEqual(len(rerank.kept_documents), 1)
            self.assertEqual(rerank.kept_documents[0].metadata.get("doc_id"), "doc-1")
            self.assertGreater(float(rerank.kept_documents[0].metadata.get("retrieval_score", 0.0)), 0.35)
            self.assertTrue(rerank.low_score_filtered)

    async def test_rerank_documents_can_prefer_follow_up_focus_document(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            config = AppConfig(
                openai_api_key="",
                openai_base_url="",
                chat_model="",
                embedding_model="",
                retrieval_top_k=2,
                retrieval_fetch_k=5,
                rerank_min_score=0.0,
                rerank_min_keep=1,
                db_path=base / "app_state.db",
                vector_dir=base / "chroma",
            )
            store = VectorStoreService(config)
            recalled = [
                SourceDocument(
                    page_content="This paper discusses photoelectrochemical upcycling of polyimide waste.",
                    metadata={
                        "course_id": "course-a",
                        "doc_id": "doc-zhao",
                        "file_name": "zhao.pdf",
                        "retrieval_vector_score": 0.55,
                    },
                ),
                SourceDocument(
                    page_content="This paper discusses photocatalytic H2O2 production from PET waste.",
                    metadata={
                        "course_id": "course-a",
                        "doc_id": "doc-alk-cn",
                        "file_name": "alk-cn.pdf",
                        "retrieval_vector_score": 0.58,
                    },
                ),
            ]

            rerank = store.rerank_documents(
                "Is it closer to photoelectrochemical upcycling or photoreforming?",
                recalled,
                top_k=1,
                min_score=0.0,
                min_keep=1,
                preferred_doc_ids=["doc-zhao"],
            )

            self.assertEqual(rerank.kept_documents[0].metadata.get("doc_id"), "doc-zhao")
            self.assertEqual(rerank.kept_documents[0].metadata.get("score_breakdown", {}).get("focus"), 1.0)


if __name__ == "__main__":
    unittest.main()
