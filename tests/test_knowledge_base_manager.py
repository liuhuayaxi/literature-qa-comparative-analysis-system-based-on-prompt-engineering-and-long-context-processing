from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from src.config import AppConfig
from src.knowledge_ingestion import DocumentIndexer
from src.knowledge_base_manager import KnowledgeBaseManager
from src.retrieval_engine import VectorStoreService


class KnowledgeBaseManagerTests(unittest.IsolatedAsyncioTestCase):
    async def test_rename_delete_and_chunk_details(self) -> None:
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            try:
                file_path = Path("data/raw/course-a/lectures/lesson.txt")
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text("RAG can ground answers.\n\nChunking keeps retrieval focused.", encoding="utf-8")

                config = AppConfig(
                    openai_api_key="",
                    openai_base_url="",
                    chat_model="",
                    embedding_model="",
                    db_path=Path("storage/app_state.db"),
                    vector_dir=Path("storage/chroma"),
                )
                vector_store = VectorStoreService(config)
                indexer = DocumentIndexer(config, vector_store)
                manager = KnowledgeBaseManager(config, vector_store, indexer)

                manageable_records = await manager.list_manageable_files("course-a")
                self.assertEqual(len(manageable_records), 1)
                self.assertFalse(manageable_records[0].is_vectorized)

                result = await manager.vectorize_files("course-a", [manageable_records[0].file_path], chunk_size=60, chunk_overlap=10)
                self.assertEqual(result["vectorized_files"], 1)

                records = await manager.list_documents("course-a")
                self.assertEqual(len(records), 1)
                self.assertGreaterEqual(records[0].chunk_count, 1)

                manageable_records = await manager.list_manageable_files("course-a")
                self.assertTrue(manageable_records[0].is_vectorized)

                chunk_details = await manager.get_chunk_details("course-a", manageable_records[0].file_path)
                self.assertTrue(chunk_details)
                self.assertIn("RAG can ground answers.", chunk_details[0]["content"])

                renamed = await manager.rename_file_by_path("course-a", manageable_records[0].file_path, "lesson-renamed")
                self.assertEqual(renamed.file_name, "lesson-renamed.txt")
                self.assertTrue(Path("data/raw/course-a/lectures/lesson-renamed.txt").exists())

                result = await manager.delete_file_paths("course-a", [renamed.file_path])
                self.assertEqual(result["deleted_files"], 1)
                self.assertEqual(await manager.list_documents("course-a"), [])
            finally:
                os.chdir(original_cwd)

    async def test_delete_knowledge_base_removes_raw_files_and_vectors(self) -> None:
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            try:
                file_path = Path("data/raw/course-b/papers/reference.txt")
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text("Energy materials can be analyzed through retrieval.", encoding="utf-8")

                config = AppConfig(
                    openai_api_key="",
                    openai_base_url="",
                    chat_model="",
                    embedding_model="",
                    db_path=Path("storage/app_state.db"),
                    vector_dir=Path("storage/chroma"),
                )
                vector_store = VectorStoreService(config)
                indexer = DocumentIndexer(config, vector_store)
                manager = KnowledgeBaseManager(config, vector_store, indexer)

                manageable_records = await manager.list_manageable_files("course-b")
                self.assertEqual(len(manageable_records), 1)

                result = await manager.vectorize_files("course-b", [manageable_records[0].file_path], chunk_size=80, chunk_overlap=10)
                self.assertEqual(result["vectorized_files"], 1)
                self.assertTrue(await manager.list_documents("course-b"))

                delete_result = await manager.delete_knowledge_base("course-b")
                self.assertEqual(delete_result["deleted_files"], 1)
                self.assertGreaterEqual(delete_result["deleted_vector_docs"], 1)
                self.assertFalse(Path("data/raw/course-b").exists())
                self.assertEqual(await manager.list_documents("course-b"), [])
                self.assertEqual(await manager.list_manageable_files("course-b"), [])
            finally:
                os.chdir(original_cwd)

    async def test_vectorize_files_skips_unchanged_vectorized_files(self) -> None:
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            try:
                file_path = Path("data/raw/course-skip/papers/reference.txt")
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text("Stable content for vectorization reuse.", encoding="utf-8")

                config = AppConfig(
                    openai_api_key="",
                    openai_base_url="",
                    chat_model="",
                    embedding_model="",
                    db_path=Path("storage/app_state.db"),
                    vector_dir=Path("storage/chroma"),
                )
                vector_store = VectorStoreService(config)
                indexer = DocumentIndexer(config, vector_store)
                manager = KnowledgeBaseManager(config, vector_store, indexer)

                manageable_records = await manager.list_manageable_files("course-skip")
                first_result = await manager.vectorize_files(
                    "course-skip",
                    [manageable_records[0].file_path],
                    chunk_size=80,
                    chunk_overlap=10,
                )
                self.assertEqual(first_result["vectorized_files"], 1)
                self.assertEqual(first_result["skipped_files"], 0)

                manageable_records = await manager.list_manageable_files("course-skip")
                second_result = await manager.vectorize_files(
                    "course-skip",
                    [manageable_records[0].file_path],
                    chunk_size=80,
                    chunk_overlap=10,
                )
                self.assertEqual(second_result["vectorized_files"], 0)
                self.assertEqual(second_result["skipped_files"], 1)
                self.assertEqual(len(await manager.list_documents("course-skip")), 1)
            finally:
                os.chdir(original_cwd)

    async def test_auto_repairs_legacy_vector_course_after_partial_rename(self) -> None:
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            try:
                file_path = Path("data/raw/lunwen/papers/reference.txt")
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text("Photocatalytic reforming can convert plastics into hydrogen.", encoding="utf-8")

                config = AppConfig(
                    openai_api_key="",
                    openai_base_url="",
                    chat_model="",
                    embedding_model="",
                    db_path=Path("storage/app_state.db"),
                    vector_dir=Path("storage/chroma"),
                )
                vector_store = VectorStoreService(config)
                indexer = DocumentIndexer(config, vector_store)
                manager = KnowledgeBaseManager(config, vector_store, indexer)

                manageable_records = await manager.list_manageable_files("lunwen")
                result = await manager.vectorize_files("lunwen", [manageable_records[0].file_path], chunk_size=80, chunk_overlap=10)
                self.assertEqual(result["vectorized_files"], 1)
                self.assertEqual(len(await manager.list_documents("lunwen")), 1)

                legacy_dir = Path("data/raw/lunwen")
                renamed_dir = Path("data/raw/论文")
                legacy_dir.rename(renamed_dir)

                repaired_records = await manager.list_documents("论文")
                self.assertEqual(len(repaired_records), 1)
                self.assertEqual(repaired_records[0].course_id, "论文")
                self.assertIn("/data/raw/论文/", repaired_records[0].file_path)

                manageable_after = await manager.list_manageable_files("论文")
                self.assertEqual(len(manageable_after), 1)
                self.assertTrue(manageable_after[0].is_vectorized)

                knowledge_bases = await manager.list_knowledge_bases()
                self.assertIn("论文", knowledge_bases)
                self.assertNotIn("lunwen", knowledge_bases)
            finally:
                os.chdir(original_cwd)


if __name__ == "__main__":
    unittest.main()
