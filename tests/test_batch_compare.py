from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.analysis_engine import (
    BatchComparisonService,
    SingleDocumentAnalysisService,
    parse_extraction_field_specs,
)
from src.app_utils import TaskControl, task_control_context
from src.config import AppConfig
from src.errors import OperationPausedError
from src.retrieval_engine import VectorStoreService
from src.models import SourceDocument


class BatchComparisonTests(unittest.IsolatedAsyncioTestCase):
    async def test_compare_documents_generates_markdown_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            config = AppConfig(
                openai_api_key="",
                openai_base_url="",
                chat_model="",
                embedding_model="",
                db_path=base / "state.db",
                vector_dir=base / "chroma",
                enable_result_cache=False,
            )
            vector_store = VectorStoreService(config)
            await vector_store.upsert_documents(
                [
                    SourceDocument(
                        page_content="Paper one studies retrieval. Risk: evaluation is narrow.",
                        metadata={
                            "course_id": "course-a",
                            "doc_id": "doc-1",
                            "file_name": "paper1.txt",
                            "file_path": str(base / "paper1.txt"),
                            "file_ext": "txt",
                            "source_type": "paper",
                            "language": "en",
                            "chunk_id": "chunk-1",
                        },
                    ),
                    SourceDocument(
                        page_content="Paper two studies generation. Limitation: training cost is high.",
                        metadata={
                            "course_id": "course-a",
                            "doc_id": "doc-2",
                            "file_name": "paper2.txt",
                            "file_path": str(base / "paper2.txt"),
                            "file_ext": "txt",
                            "source_type": "paper",
                            "language": "en",
                            "chunk_id": "chunk-2",
                        },
                    ),
                ]
            )
            single_doc_service = SingleDocumentAnalysisService(config, vector_store)
            batch_service = BatchComparisonService(config, vector_store, single_doc_service)

            report = await batch_service.compare_documents("course-a", ["doc-1", "doc-2"], output_language="zh")

            self.assertIn("## 4. 关键差异", report.markdown)
            self.assertIn("## 8. 关键结论引用", report.markdown)
            self.assertTrue(Path(report.output_path).exists())

    async def test_compare_documents_rewrites_doc_ids_and_links_citations_in_chat_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            config = AppConfig(
                openai_api_key="x",
                openai_base_url="https://example.com/v1",
                chat_model="test-model",
                embedding_model="",
                db_path=base / "state.db",
                vector_dir=base / "chroma",
                enable_result_cache=False,
            )
            vector_store = VectorStoreService(config)
            await vector_store.upsert_documents(
                [
                    SourceDocument(
                        page_content="Paper one studies retrieval. Risk: evaluation is narrow.",
                        metadata={
                            "course_id": "course-chat",
                            "doc_id": "doc-1",
                            "file_name": "paper1.txt",
                            "file_path": str(base / "paper1.txt"),
                            "file_ext": "txt",
                            "source_type": "paper",
                            "language": "en",
                            "chunk_id": "chunk-1",
                            "page_label": "p.1",
                        },
                    ),
                    SourceDocument(
                        page_content="Paper two studies generation. Limitation: training cost is high.",
                        metadata={
                            "course_id": "course-chat",
                            "doc_id": "doc-2",
                            "file_name": "paper2.txt",
                            "file_path": str(base / "paper2.txt"),
                            "file_ext": "txt",
                            "source_type": "paper",
                            "language": "en",
                            "chunk_id": "chunk-2",
                            "page_label": "p.2",
                        },
                    ),
                ]
            )
            single_doc_service = SingleDocumentAnalysisService(config, vector_store)
            batch_service = BatchComparisonService(config, vector_store, single_doc_service)

            async def fake_invoke(*_args, **_kwargs) -> str:
                return "# 对比结论\n\n在 doc-1 与 doc-2 之间，结论主要来自 [1, 2]。"

            with patch("src.analysis_engine.invoke_chat_text", new=fake_invoke):
                report = await batch_service.compare_documents(
                    "course-chat",
                    ["doc-1", "doc-2"],
                    output_language="zh",
                )

            self.assertNotIn("doc-1", report.markdown)
            self.assertNotIn("doc-2", report.markdown)
            self.assertIn("《paper1.txt》", report.markdown)
            self.assertIn("《paper2.txt》", report.markdown)
            self.assertIn("[1](#report-citation-1)", report.markdown)
            self.assertIn("[2](#report-citation-2)", report.markdown)

    async def test_compare_documents_with_target_fields_outputs_table_and_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            config = AppConfig(
                openai_api_key="",
                openai_base_url="",
                chat_model="",
                embedding_model="",
                db_path=base / "state.db",
                vector_dir=base / "chroma",
                enable_result_cache=False,
            )
            vector_store = VectorStoreService(config)
            await vector_store.upsert_documents(
                [
                    SourceDocument(
                        page_content="Photocurrent density reached 12.5 mA/cm² at 1.23 V. Faradaic efficiency was 78%.",
                        metadata={
                            "course_id": "course-b",
                            "doc_id": "doc-1",
                            "file_name": "paper1.txt",
                            "file_path": str(base / "paper1.txt"),
                            "file_ext": "txt",
                            "source_type": "paper",
                            "language": "en",
                            "chunk_id": "chunk-1",
                            "page_label": "p.1",
                        },
                    ),
                    SourceDocument(
                        page_content="Photocurrent density reached 9.8 mA/cm². Faradaic efficiency was 83%.",
                        metadata={
                            "course_id": "course-b",
                            "doc_id": "doc-2",
                            "file_name": "paper2.txt",
                            "file_path": str(base / "paper2.txt"),
                            "file_ext": "txt",
                            "source_type": "paper",
                            "language": "en",
                            "chunk_id": "chunk-2",
                            "page_label": "p.2",
                        },
                    ),
                ]
            )
            field_specs = parse_extraction_field_specs(
                "Photocurrent density | extract the main reported value | mA/cm²\nFaradaic efficiency | extract the reported percentage | %"
            )
            single_doc_service = SingleDocumentAnalysisService(config, vector_store)
            batch_service = BatchComparisonService(config, vector_store, single_doc_service)

            report = await batch_service.compare_documents(
                "course-b",
                ["doc-1", "doc-2"],
                output_language="en",
                target_fields=field_specs,
                export_csv=True,
            )

            self.assertIn("## Targeted Data Comparison Table", report.markdown)
            self.assertIn("Photocurrent density", report.markdown)
            self.assertIn("12.5 mA/cm²", report.markdown)
            self.assertIn("83 %", report.markdown)
            self.assertNotIn("<details>", report.markdown)
            self.assertNotIn("</details>", report.markdown)
            self.assertTrue(report.csv_output_path)
            self.assertTrue(Path(report.csv_output_path).exists())

    async def test_compare_documents_emits_rich_progress_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            config = AppConfig(
                openai_api_key="",
                openai_base_url="",
                chat_model="",
                embedding_model="",
                db_path=base / "state.db",
                vector_dir=base / "chroma",
                enable_result_cache=False,
            )
            vector_store = VectorStoreService(config)
            await vector_store.upsert_documents(
                [
                    SourceDocument(
                        page_content="Paper one studies retrieval. Risk: evaluation is narrow.",
                        metadata={
                            "course_id": "course-progress",
                            "doc_id": "doc-1",
                            "file_name": "paper1.txt",
                            "file_path": str(base / "paper1.txt"),
                            "file_ext": "txt",
                            "source_type": "paper",
                            "language": "en",
                            "chunk_id": "chunk-1",
                        },
                    ),
                    SourceDocument(
                        page_content="Paper two studies generation. Limitation: training cost is high.",
                        metadata={
                            "course_id": "course-progress",
                            "doc_id": "doc-2",
                            "file_name": "paper2.txt",
                            "file_path": str(base / "paper2.txt"),
                            "file_ext": "txt",
                            "source_type": "paper",
                            "language": "en",
                            "chunk_id": "chunk-2",
                        },
                    ),
                ]
            )
            single_doc_service = SingleDocumentAnalysisService(config, vector_store)
            batch_service = BatchComparisonService(config, vector_store, single_doc_service)
            messages: list[str] = []

            async def progress_callback(message: str) -> None:
                messages.append(message)

            await batch_service.compare_documents(
                "course-progress",
                ["doc-1", "doc-2"],
                output_language="zh",
                progress_callback=progress_callback,
            )

            self.assertTrue(any("总进度" in message for message in messages))
            self.assertTrue(any("paper1.txt" in message for message in messages))
            self.assertTrue(any("单文档分析" in message for message in messages))
            self.assertTrue(any("结果保存" in message for message in messages))

    async def test_compare_documents_can_pause_and_resume_from_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            config = AppConfig(
                openai_api_key="",
                openai_base_url="",
                chat_model="",
                embedding_model="",
                db_path=base / "state.db",
                vector_dir=base / "chroma",
                analysis_checkpoint_dir=base / "checkpoints",
                enable_result_cache=False,
            )
            vector_store = VectorStoreService(config)
            await vector_store.upsert_documents(
                [
                    SourceDocument(
                        page_content="Paper one studies retrieval. Risk: evaluation is narrow.",
                        metadata={
                            "course_id": "course-resume",
                            "doc_id": "doc-1",
                            "file_name": "paper1.txt",
                            "file_path": str(base / "paper1.txt"),
                            "file_ext": "txt",
                            "source_type": "paper",
                            "language": "en",
                            "chunk_id": "chunk-1",
                        },
                    ),
                    SourceDocument(
                        page_content="Paper two studies generation. Limitation: training cost is high.",
                        metadata={
                            "course_id": "course-resume",
                            "doc_id": "doc-2",
                            "file_name": "paper2.txt",
                            "file_path": str(base / "paper2.txt"),
                            "file_ext": "txt",
                            "source_type": "paper",
                            "language": "en",
                            "chunk_id": "chunk-2",
                        },
                    ),
                ]
            )
            single_doc_service = SingleDocumentAnalysisService(config, vector_store)
            batch_service = BatchComparisonService(config, vector_store, single_doc_service)
            control = TaskControl(label="批量对比")
            messages: list[str] = []
            pause_requested = {"done": False}

            async def progress_callback(message: str) -> None:
                messages.append(message)
                if "本篇分析完成" in message and not pause_requested["done"]:
                    control.request_pause()
                    pause_requested["done"] = True

            with self.assertRaises(OperationPausedError):
                with task_control_context(control):
                    await batch_service.compare_documents(
                        "course-resume",
                        ["doc-1", "doc-2"],
                        output_language="zh",
                        progress_callback=progress_callback,
                    )

            checkpoint_info = await batch_service.inspect_compare_checkpoint(
                "course-resume",
                ["doc-1", "doc-2"],
                output_language="zh",
            )
            self.assertTrue(checkpoint_info["exists"])
            self.assertEqual(checkpoint_info["status"], "paused")
            self.assertGreaterEqual(int(checkpoint_info["analysis_done"]), 1)
            self.assertTrue(Path(checkpoint_info["progress_output_path"]).exists())

            report = await batch_service.compare_documents(
                "course-resume",
                ["doc-1", "doc-2"],
                output_language="zh",
                progress_callback=progress_callback,
                resume=True,
            )

            self.assertTrue(Path(report.output_path).exists())

    async def test_clear_compare_checkpoint_removes_saved_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            config = AppConfig(
                openai_api_key="",
                openai_base_url="",
                chat_model="",
                embedding_model="",
                db_path=base / "state.db",
                vector_dir=base / "chroma",
                analysis_checkpoint_dir=base / "checkpoints",
                enable_result_cache=False,
            )
            vector_store = VectorStoreService(config)
            await vector_store.upsert_documents(
                [
                    SourceDocument(
                        page_content="Paper one studies retrieval.",
                        metadata={
                            "course_id": "course-clear",
                            "doc_id": "doc-1",
                            "file_name": "paper1.txt",
                            "file_path": str(base / "paper1.txt"),
                            "file_ext": "txt",
                            "source_type": "paper",
                            "language": "en",
                            "chunk_id": "chunk-1",
                        },
                    )
                ]
            )
            single_doc_service = SingleDocumentAnalysisService(config, vector_store)
            batch_service = BatchComparisonService(config, vector_store, single_doc_service)
            control = TaskControl(label="批量对比")

            async def progress_callback(message: str) -> None:
                if "本篇分析完成" in message:
                    control.request_pause()

            with self.assertRaises(OperationPausedError):
                with task_control_context(control):
                    await batch_service.compare_documents(
                        "course-clear",
                        ["doc-1"],
                        output_language="zh",
                        progress_callback=progress_callback,
                    )

            info = await batch_service.inspect_compare_checkpoint(
                "course-clear",
                ["doc-1"],
                output_language="zh",
            )
            self.assertTrue(info["exists"])
            result = await batch_service.clear_compare_checkpoint(
                "course-clear",
                ["doc-1"],
                output_language="zh",
            )
            self.assertTrue(result["removed"])
            info_after = await batch_service.inspect_compare_checkpoint(
                "course-clear",
                ["doc-1"],
                output_language="zh",
            )
            self.assertFalse(info_after["exists"])

    async def test_compare_checkpoint_tracks_active_document_window_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            config = AppConfig(
                openai_api_key="",
                openai_base_url="",
                chat_model="",
                embedding_model="",
                db_path=base / "state.db",
                vector_dir=base / "chroma",
                analysis_checkpoint_dir=base / "checkpoints",
                enable_result_cache=False,
            )
            vector_store = VectorStoreService(config)
            await vector_store.upsert_documents(
                [
                    SourceDocument(
                        page_content="Paper one studies retrieval.",
                        metadata={
                            "course_id": "course-active",
                            "doc_id": "doc-1",
                            "file_name": "paper1.txt",
                            "file_path": str(base / "paper1.txt"),
                            "file_ext": "txt",
                            "source_type": "paper",
                            "language": "en",
                            "chunk_id": "chunk-1",
                        },
                    )
                ]
            )
            single_doc_service = SingleDocumentAnalysisService(config, vector_store)
            batch_service = BatchComparisonService(config, vector_store, single_doc_service)

            async def fake_analyze_document(*, progress_callback=None, state_callback=None, **_kwargs):
                if progress_callback is not None:
                    await progress_callback("正在处理滑窗 2/5: paper1.txt")
                if state_callback is not None:
                    await state_callback(
                        {
                            "title": "paper1.txt",
                            "current_window": 2,
                            "total_windows": 5,
                            "window_summaries": ["summary-1", "summary-2"],
                        }
                    )
                raise OperationPausedError()

            single_doc_service.analyze_document = fake_analyze_document  # type: ignore[method-assign]

            with self.assertRaises(OperationPausedError):
                await batch_service.compare_documents(
                    "course-active",
                    ["doc-1"],
                    output_language="zh",
                )

            info = await batch_service.inspect_compare_checkpoint(
                "course-active",
                ["doc-1"],
                output_language="zh",
            )
            self.assertTrue(info["exists"])
            self.assertEqual(info["status"], "paused")
            self.assertEqual(len(info["active_docs"]), 1)
            self.assertEqual(info["active_docs"][0]["title"], "paper1.txt")
            self.assertEqual(info["active_docs"][0]["current_window"], 2)
            self.assertEqual(info["active_docs"][0]["total_windows"], 5)

    async def test_field_extraction_converts_to_expected_unit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            config = AppConfig(
                openai_api_key="",
                openai_base_url="",
                chat_model="",
                embedding_model="",
                db_path=base / "state.db",
                vector_dir=base / "chroma",
                enable_result_cache=False,
            )
            vector_store = VectorStoreService(config)
            await vector_store.upsert_documents(
                [
                    SourceDocument(
                        page_content="The photocurrent density reached 12.5 mA/cm² under illumination.",
                        metadata={
                            "course_id": "course-c",
                            "doc_id": "doc-1",
                            "file_name": "paper3.txt",
                            "file_path": str(base / "paper3.txt"),
                            "file_ext": "txt",
                            "source_type": "paper",
                            "language": "en",
                            "chunk_id": "chunk-1",
                            "page_label": "p.3",
                        },
                    )
                ]
            )
            field_specs = parse_extraction_field_specs(
                "Photocurrent density | extract the reported current density | uA/cm²"
            )
            service = SingleDocumentAnalysisService(config, vector_store)

            result = await service.extract_document_fields(
                course_id="course-c",
                doc_id="doc-1",
                field_specs=field_specs,
                output_language="en",
            )

            self.assertEqual(result.fields[0].unit, "uA/cm²")
            self.assertEqual(result.fields[0].source_unit, "mA/cm²")
            self.assertTrue(result.fields[0].converted)
            self.assertIn("12500", result.fields[0].normalized_value)

    async def test_field_extraction_dedupes_textual_units_from_llm_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            config = AppConfig(
                openai_api_key="x",
                openai_base_url="https://example.com/v1",
                chat_model="test-model",
                embedding_model="",
                db_path=base / "state.db",
                vector_dir=base / "chroma",
                enable_result_cache=False,
            )
            vector_store = VectorStoreService(config)
            await vector_store.upsert_documents(
                [
                    SourceDocument(
                        page_content="Light intensity: AM 1.5G under visible light.",
                        metadata={
                            "course_id": "course-d",
                            "doc_id": "doc-1",
                            "file_name": "paper4.txt",
                            "file_path": str(base / "paper4.txt"),
                            "file_ext": "txt",
                            "source_type": "paper",
                            "language": "en",
                            "chunk_id": "chunk-1",
                            "page_label": "p.1",
                        },
                    )
                ]
            )
            field_specs = parse_extraction_field_specs("反应光强 | 如果没有提及写无 | AM")
            service = SingleDocumentAnalysisService(config, vector_store)

            async def fake_invoke(*_args, **_kwargs) -> str:
                return (
                    '{"fields":[{"field_name":"反应光强","value":"visible light AM","normalized_value":"visible light AM AM",'
                    '"unit":"AM","status":"found","evidence_chunk_id":"chunk-1","notes":""}]}'
                )

            with patch("src.analysis_engine.invoke_chat_text", new=fake_invoke):
                result = await service.extract_document_fields(
                    course_id="course-d",
                    doc_id="doc-1",
                    field_specs=field_specs,
                    output_language="zh",
                )

            self.assertEqual(result.fields[0].unit, "AM")
            self.assertIn("AM", result.fields[0].normalized_value)
            self.assertNotIn("AM AM", result.fields[0].normalized_value)

    async def test_field_extraction_corrects_ocr_umol_unit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            config = AppConfig(
                openai_api_key="",
                openai_base_url="",
                chat_model="",
                embedding_model="",
                db_path=base / "state.db",
                vector_dir=base / "chroma",
                enable_result_cache=False,
            )
            vector_store = VectorStoreService(config)
            await vector_store.upsert_documents(
                [
                    SourceDocument(
                        page_content="Hydrogen generation rate: 402.9 lmol/g/h.",
                        metadata={
                            "course_id": "course-e",
                            "doc_id": "doc-1",
                            "file_name": "paper5.txt",
                            "file_path": str(base / "paper5.txt"),
                            "file_ext": "txt",
                            "source_type": "paper",
                            "language": "en",
                            "chunk_id": "chunk-1",
                            "page_label": "p.2",
                        },
                    )
                ]
            )
            field_specs = parse_extraction_field_specs("氢气产生速率 | 提取主结果中的数值 | umol/g/h")
            service = SingleDocumentAnalysisService(config, vector_store)

            result = await service.extract_document_fields(
                course_id="course-e",
                doc_id="doc-1",
                field_specs=field_specs,
                output_language="en",
            )

            self.assertEqual(result.fields[0].unit, "umol/g/h")
            self.assertEqual(result.fields[0].source_unit, "umol/g/h")
            self.assertIn("402.9", result.fields[0].normalized_value)

    async def test_field_extraction_prefers_real_hydrogen_rate_over_material_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            config = AppConfig(
                openai_api_key="",
                openai_base_url="",
                chat_model="",
                embedding_model="",
                db_path=base / "state.db",
                vector_dir=base / "chroma",
                enable_result_cache=False,
            )
            vector_store = VectorStoreService(config)
            await vector_store.upsert_documents(
                [
                    SourceDocument(
                        page_content="With WO3 contents optimized, PCN/WO3-10 exhibits excellent photocatalytic activity for hydrogen generation via photoreforming of PLA, with hydrogen generation rate reaching 402.90 lmol/g/h.",
                        metadata={
                            "course_id": "course-f",
                            "doc_id": "doc-1",
                            "file_name": "paper6.txt",
                            "file_path": str(base / "paper6.txt"),
                            "file_ext": "txt",
                            "source_type": "paper",
                            "language": "en",
                            "chunk_id": "chunk-1",
                            "page_label": "p.7",
                        },
                    )
                ]
            )
            field_specs = parse_extraction_field_specs("氢气产生速率 | 提取主结果中的数值 | umol/g/h")
            service = SingleDocumentAnalysisService(config, vector_store)

            result = await service.extract_document_fields(
                course_id="course-f",
                doc_id="doc-1",
                field_specs=field_specs,
                output_language="zh",
            )

            self.assertEqual(result.fields[0].status, "found")
            self.assertEqual(result.fields[0].unit, "umol/g/h")
            self.assertIn("402.90", result.fields[0].normalized_value)
            self.assertNotIn("contents", result.fields[0].normalized_value)

    async def test_field_extraction_ignores_preparation_temperature_for_reaction_temperature(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            config = AppConfig(
                openai_api_key="",
                openai_base_url="",
                chat_model="",
                embedding_model="",
                db_path=base / "state.db",
                vector_dir=base / "chroma",
                enable_result_cache=False,
            )
            vector_store = VectorStoreService(config)
            await vector_store.upsert_documents(
                [
                    SourceDocument(
                        page_content="WO3 was obtained by annealing precursor at 500 °C for 4 h. The mixture was transferred to a 25 mL autoclave and maintained at 150 °C for 24 h.",
                        metadata={
                            "course_id": "course-g",
                            "doc_id": "doc-1",
                            "file_name": "paper7.txt",
                            "file_path": str(base / "paper7.txt"),
                            "file_ext": "txt",
                            "source_type": "paper",
                            "language": "en",
                            "chunk_id": "chunk-1",
                            "page_label": "p.3",
                        },
                    )
                ]
            )
            field_specs = parse_extraction_field_specs("反应温度 | 如果没有提及写无 | 摄氏度")
            service = SingleDocumentAnalysisService(config, vector_store)

            result = await service.extract_document_fields(
                course_id="course-g",
                doc_id="doc-1",
                field_specs=field_specs,
                output_language="zh",
            )

            self.assertEqual(result.fields[0].status, "not_found")

    async def test_field_extraction_prefers_specific_products_over_mechanism_sentence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            config = AppConfig(
                openai_api_key="x",
                openai_base_url="https://example.com/v1",
                chat_model="test-model",
                embedding_model="",
                db_path=base / "state.db",
                vector_dir=base / "chroma",
                enable_result_cache=False,
            )
            vector_store = VectorStoreService(config)
            await vector_store.upsert_documents(
                [
                    SourceDocument(
                        page_content="Value-added chemicals such as pyruvate and acetate were produced with high selectivities.",
                        metadata={
                            "course_id": "course-h",
                            "doc_id": "doc-1",
                            "file_name": "paper8.txt",
                            "file_path": str(base / "paper8.txt"),
                            "file_ext": "txt",
                            "source_type": "paper",
                            "language": "en",
                            "chunk_id": "chunk-good",
                            "page_label": "p.2",
                        },
                    ),
                    SourceDocument(
                        page_content="The mechanism is confirmed, wherein Ni3S4 extracts electrons from ZnCdS and promotes charge separation to facilitate H2 production.",
                        metadata={
                            "course_id": "course-h",
                            "doc_id": "doc-1",
                            "file_name": "paper8.txt",
                            "file_path": str(base / "paper8.txt"),
                            "file_ext": "txt",
                            "source_type": "paper",
                            "language": "en",
                            "chunk_id": "chunk-bad",
                            "page_label": "p.3",
                        },
                    ),
                ]
            )
            field_specs = parse_extraction_field_specs("产物 | 如果没有提及写无 |")
            service = SingleDocumentAnalysisService(config, vector_store)

            async def fake_invoke(*_args, **_kwargs) -> str:
                return (
                    '{"fields":[{"field_name":"产物","value":"wherein Ni3S4 extracts electrons from ZnCdS and promotes charge separation",'
                    '"normalized_value":"wherein Ni3S4 extracts electrons from ZnCdS and promotes charge separation",'
                    '"unit":"","status":"found","evidence_chunk_id":"chunk-bad","notes":""}]}'
                )

            with patch("src.analysis_engine.invoke_chat_text", new=fake_invoke):
                result = await service.extract_document_fields(
                    course_id="course-h",
                    doc_id="doc-1",
                    field_specs=field_specs,
                    output_language="zh",
                )

            self.assertEqual(result.fields[0].status, "found")
            self.assertIn("pyruvate", result.fields[0].normalized_value.lower())
            self.assertIn("acetate", result.fields[0].normalized_value.lower())

    async def test_field_extraction_prefers_specific_reactant_over_generic_plastic_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            config = AppConfig(
                openai_api_key="x",
                openai_base_url="https://example.com/v1",
                chat_model="test-model",
                embedding_model="",
                db_path=base / "state.db",
                vector_dir=base / "chroma",
                enable_result_cache=False,
            )
            vector_store = VectorStoreService(config)
            await vector_store.upsert_documents(
                [
                    SourceDocument(
                        page_content="A fully solar powered photoelectrochemical system was developed to selectively upgrade polyimide waste into valuable commodity chemicals.",
                        metadata={
                            "course_id": "course-i",
                            "doc_id": "doc-1",
                            "file_name": "paper9.txt",
                            "file_path": str(base / "paper9.txt"),
                            "file_ext": "txt",
                            "source_type": "paper",
                            "language": "en",
                            "chunk_id": "chunk-good",
                            "page_label": "p.1",
                        },
                    ),
                    SourceDocument(
                        page_content="Photoelectrochemical upcycling of plastic waste for green chemicals and hydrogen production has attracted attention.",
                        metadata={
                            "course_id": "course-i",
                            "doc_id": "doc-1",
                            "file_name": "paper9.txt",
                            "file_path": str(base / "paper9.txt"),
                            "file_ext": "txt",
                            "source_type": "paper",
                            "language": "en",
                            "chunk_id": "chunk-generic",
                            "page_label": "p.2",
                        },
                    ),
                ]
            )
            field_specs = parse_extraction_field_specs("反应物 | 如果没有提及写无 |")
            service = SingleDocumentAnalysisService(config, vector_store)

            async def fake_invoke(*_args, **_kwargs) -> str:
                return (
                    '{"fields":[{"field_name":"反应物","value":"plastic waste","normalized_value":"plastic waste",'
                    '"unit":"","status":"found","evidence_chunk_id":"chunk-generic","notes":""}]}'
                )

            with patch("src.analysis_engine.invoke_chat_text", new=fake_invoke):
                result = await service.extract_document_fields(
                    course_id="course-i",
                    doc_id="doc-1",
                    field_specs=field_specs,
                    output_language="zh",
                )

            self.assertEqual(result.fields[0].status, "found")
            self.assertIn("polyimide", result.fields[0].normalized_value.lower())
            self.assertNotEqual(result.fields[0].chunk_id, "chunk-generic")

    async def test_field_extraction_ignores_intermediate_product_sentence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            config = AppConfig(
                openai_api_key="",
                openai_base_url="",
                chat_model="",
                embedding_model="",
                db_path=base / "state.db",
                vector_dir=base / "chroma",
                enable_result_cache=False,
            )
            vector_store = VectorStoreService(config)
            await vector_store.upsert_documents(
                [
                    SourceDocument(
                        page_content="The system upgraded PI waste into commodity chemicals including formic acid, acetic acid, and succinic acid.",
                        metadata={
                            "course_id": "course-j",
                            "doc_id": "doc-1",
                            "file_name": "paper10.txt",
                            "file_path": str(base / "paper10.txt"),
                            "file_ext": "txt",
                            "source_type": "paper",
                            "language": "en",
                            "chunk_id": "chunk-good",
                            "page_label": "p.1",
                        },
                    ),
                    SourceDocument(
                        page_content="Detailed products and intermediates analysis identified aminophenol, nitrophenol, HQ, and BQ along the pathway.",
                        metadata={
                            "course_id": "course-j",
                            "doc_id": "doc-1",
                            "file_name": "paper10.txt",
                            "file_path": str(base / "paper10.txt"),
                            "file_ext": "txt",
                            "source_type": "paper",
                            "language": "en",
                            "chunk_id": "chunk-bad",
                            "page_label": "p.5",
                        },
                    ),
                ]
            )
            field_specs = parse_extraction_field_specs("产物 | 如果没有提及写无 |")
            service = SingleDocumentAnalysisService(config, vector_store)

            result = await service.extract_document_fields(
                course_id="course-j",
                doc_id="doc-1",
                field_specs=field_specs,
                output_language="zh",
            )

            self.assertEqual(result.fields[0].status, "found")
            self.assertIn("formic acid", result.fields[0].normalized_value.lower())
            self.assertNotIn("aminophenol", result.fields[0].normalized_value.lower())

    async def test_field_extraction_rejects_preparation_temperature_from_llm_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            config = AppConfig(
                openai_api_key="x",
                openai_base_url="https://example.com/v1",
                chat_model="test-model",
                embedding_model="",
                db_path=base / "state.db",
                vector_dir=base / "chroma",
                enable_result_cache=False,
            )
            vector_store = VectorStoreService(config)
            await vector_store.upsert_documents(
                [
                    SourceDocument(
                        page_content="PI Waste Dissolution in KOH Solution: the solution was heated in an oven at 95°C for 20 h before use.",
                        metadata={
                            "course_id": "course-k",
                            "doc_id": "doc-1",
                            "file_name": "paper11.txt",
                            "file_path": str(base / "paper11.txt"),
                            "file_ext": "txt",
                            "source_type": "paper",
                            "language": "en",
                            "chunk_id": "chunk-prep",
                            "page_label": "p.7",
                        },
                    )
                ]
            )
            field_specs = parse_extraction_field_specs("反应温度 | 如果没有提及写无 | 摄氏度")
            service = SingleDocumentAnalysisService(config, vector_store)

            async def fake_invoke(*_args, **_kwargs) -> str:
                return (
                    '{"fields":[{"field_name":"反应温度","value":"95°C","normalized_value":"95 摄氏度",'
                    '"unit":"摄氏度","status":"found","evidence_chunk_id":"chunk-prep","notes":""}]}'
                )

            with patch("src.analysis_engine.invoke_chat_text", new=fake_invoke):
                result = await service.extract_document_fields(
                    course_id="course-k",
                    doc_id="doc-1",
                    field_specs=field_specs,
                    output_language="zh",
                )

            self.assertEqual(result.fields[0].status, "not_found")

    async def test_field_extraction_prefers_converted_value_over_model_normalized_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            config = AppConfig(
                openai_api_key="x",
                openai_base_url="https://example.com/v1",
                chat_model="test-model",
                embedding_model="",
                db_path=base / "state.db",
                vector_dir=base / "chroma",
                enable_result_cache=False,
            )
            vector_store = VectorStoreService(config)
            await vector_store.upsert_documents(
                [
                    SourceDocument(
                        page_content="The catalyst achieved H2 production rates of 27.9 mmol g−1 h−1 for PLA and 17.4 mmol g−1 h−1 for PET.",
                        metadata={
                            "course_id": "course-m",
                            "doc_id": "doc-1",
                            "file_name": "paper13.txt",
                            "file_path": str(base / "paper13.txt"),
                            "file_ext": "txt",
                            "source_type": "paper",
                            "language": "en",
                            "chunk_id": "chunk-rate",
                            "page_label": "p.2",
                        },
                    )
                ]
            )
            field_specs = parse_extraction_field_specs("氢气产生速率 | 提取主结果中的数值 | umol/g/h")
            service = SingleDocumentAnalysisService(config, vector_store)

            async def fake_invoke(*_args, **_kwargs) -> str:
                return (
                    '{"fields":[{"field_name":"氢气产生速率","value":"27.9 mmol g−1 h−1 and 17.4 mmol g−1 h−1",'
                    '"normalized_value":"27.9 mmol/g/h and 17.4 mmol/g/h","unit":"mmol/g/h",'
                    '"status":"found","evidence_chunk_id":"chunk-rate","notes":""}]}'
                )

            with patch("src.analysis_engine.invoke_chat_text", new=fake_invoke):
                result = await service.extract_document_fields(
                    course_id="course-m",
                    doc_id="doc-1",
                    field_specs=field_specs,
                    output_language="zh",
                )

            self.assertIn("27900", result.fields[0].normalized_value)
            self.assertIn("17400", result.fields[0].normalized_value)
            self.assertNotIn("mmol/g/h", result.fields[0].normalized_value)

    async def test_field_extraction_converts_multiple_rates_without_unit_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            config = AppConfig(
                openai_api_key="",
                openai_base_url="",
                chat_model="",
                embedding_model="",
                db_path=base / "state.db",
                vector_dir=base / "chroma",
                enable_result_cache=False,
            )
            vector_store = VectorStoreService(config)
            await vector_store.upsert_documents(
                [
                    SourceDocument(
                        page_content="The compound exhibited H2 production rates as high as 27.9 mmol g−1 h−1 for PLA and 17.4 mmol g−1 h−1 for PET.",
                        metadata={
                            "course_id": "course-l",
                            "doc_id": "doc-1",
                            "file_name": "paper12.txt",
                            "file_path": str(base / "paper12.txt"),
                            "file_ext": "txt",
                            "source_type": "paper",
                            "language": "en",
                            "chunk_id": "chunk-1",
                            "page_label": "p.2",
                        },
                    )
                ]
            )
            field_specs = parse_extraction_field_specs("氢气产生速率 | 提取主结果中的数值 | umol/g/h")
            service = SingleDocumentAnalysisService(config, vector_store)

            result = await service.extract_document_fields(
                course_id="course-l",
                doc_id="doc-1",
                field_specs=field_specs,
                output_language="zh",
            )

            self.assertIn("27900", result.fields[0].normalized_value)
            self.assertIn("17400", result.fields[0].normalized_value)
            self.assertNotIn("1000 and 1000", result.fields[0].normalized_value)


if __name__ == "__main__":
    unittest.main()
