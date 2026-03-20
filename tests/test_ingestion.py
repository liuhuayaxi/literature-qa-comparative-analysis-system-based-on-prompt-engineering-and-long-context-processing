from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.knowledge_ingestion import load_documents, split_documents
from src.models import SourceDocument


class IngestionTests(unittest.IsolatedAsyncioTestCase):
    async def test_load_and_split_mixed_text_documents(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            markdown_path = tmp_path / "lecture.md"
            text_path = tmp_path / "notes.txt"
            markdown_path.write_text("# Intro\nTransformer models are powerful.\n\n## Attention\nAttention helps context.", encoding="utf-8")
            text_path.write_text("This assignment explains retrieval augmented generation.\n\nUse citations carefully.", encoding="utf-8")

            loaded = await load_documents(
                course_id="course-a",
                file_paths=[markdown_path, text_path],
                source_type="lecture",
            )

            self.assertEqual(len(loaded), 2)
            chunks = split_documents(loaded, chunk_size=60, chunk_overlap=10)
            self.assertGreaterEqual(len(chunks), 3)
            self.assertTrue(all(chunk.metadata.get("chunk_id") for chunk in chunks))
            self.assertTrue(any(chunk.metadata.get("section_label") == "Attention" for chunk in chunks))

    async def test_split_documents_can_merge_small_chunks(self) -> None:
        source = SourceDocument(
            page_content="Alpha short paragraph.\n\nBeta short paragraph.\n\nGamma paragraph with a bit more detail.",
            metadata={
                "course_id": "course-a",
                "doc_id": "doc-1",
                "file_name": "notes.txt",
                "file_path": "/tmp/notes.txt",
                "file_ext": "txt",
                "source_type": "lecture",
                "language": "en",
            },
        )

        normal_chunks = split_documents([source], chunk_size=80, chunk_overlap=10)
        merged_chunks = split_documents(
            [source],
            chunk_size=80,
            chunk_overlap=10,
            merge_small_chunks=True,
            min_chunk_size=45,
        )

        self.assertEqual(len(normal_chunks), 3)
        self.assertLess(len(merged_chunks), len(normal_chunks))
        self.assertGreaterEqual(int(merged_chunks[0].metadata.get("merged_from_count", 1)), 2)
        self.assertIn("Alpha short paragraph.", merged_chunks[0].page_content)
        self.assertIn("Beta short paragraph.", merged_chunks[0].page_content)


if __name__ == "__main__":
    unittest.main()
