"""UI rendering regression tests for knowledge-base detail helpers."""

from __future__ import annotations

import unittest

from src.models import DocumentRecord, DocumentExtractionResult, ExtractedFieldValue
from src.ui_app import (
    _render_manage_detail_card,
    _render_manage_details_html,
    _render_markdown_html,
    _render_report_log_text,
    _single_doc_extraction_markdown,
)


def _record(index: int) -> DocumentRecord:
    return DocumentRecord(
        doc_id=f"doc-{index}",
        course_id="demo-course",
        source_type="lecture",
        file_name=f"file_{index}.pdf",
        file_path=f"/tmp/file_{index}.pdf",
        file_ext="pdf",
        language="en",
        chunk_count=3,
        is_vectorized=True,
    )


class UiAppRenderTests(unittest.TestCase):
    """Validate the pure HTML rendering helpers used by the notebook UI."""

    def test_manage_details_html_paginates_file_cards(self) -> None:
        records = [_record(index) for index in range(12)]
        detail_lists = [[] for _ in records]

        _, status_html, total_items, total_pages, current_page = _render_manage_details_html(
            course_id="demo-course",
            records=records,
            detail_lists=detail_lists,
            search_query="",
            selected_file_count=0,
            total_file_count=len(records),
            details_loaded=False,
            total_chunk_count=36,
            current_page=2,
            page_size=10,
        )

        self.assertEqual(total_items, 12)
        self.assertEqual(total_pages, 2)
        self.assertEqual(current_page, 2)
        self.assertIn("当前第 2 / 2 页", status_html)

    def test_manage_details_html_clamps_page_to_valid_range(self) -> None:
        records = [_record(index) for index in range(3)]
        detail_lists = [[] for _ in records]

        _, _, total_items, total_pages, current_page = _render_manage_details_html(
            course_id="demo-course",
            records=records,
            detail_lists=detail_lists,
            search_query="",
            selected_file_count=0,
            total_file_count=len(records),
            details_loaded=False,
            total_chunk_count=9,
            current_page=9,
            page_size=10,
        )

        self.assertEqual(total_items, 3)
        self.assertEqual(total_pages, 1)
        self.assertEqual(current_page, 1)

    def test_manage_detail_card_guides_user_to_select_files_before_loading(self) -> None:
        rendered = _render_manage_detail_card(
            _record(1),
            chunk_details=[],
            normalized_query="",
            details_loaded=False,
        )

        self.assertTrue(rendered["include"])
        self.assertIn("请先在上面的文件列表中选中需要查看的文件", rendered["html"])

    def test_single_doc_extraction_markdown_expands_evidence_sections(self) -> None:
        extraction = DocumentExtractionResult(
            doc_id="doc-1",
            title="paper.pdf",
            fields=[
                ExtractedFieldValue(
                    field_name="反应物",
                    normalized_value="PET",
                    status="found",
                    source_file="paper.pdf",
                    page_label="p.2",
                    chunk_id="chunk-1",
                    evidence_quote="PET was used as the feedstock.",
                )
            ],
        )

        markdown = _single_doc_extraction_markdown(extraction)

        self.assertIn("#### 反应物 | found | PET", markdown)
        self.assertNotIn("<details>", markdown)
        self.assertNotIn("</details>", markdown)
        self.assertNotIn("Chunk ID", markdown)

    def test_render_markdown_html_outputs_heading_html(self) -> None:
        rendered = _render_markdown_html("# 标题\n\n- 项目A")

        self.assertIn("<h1", rendered)
        self.assertIn("标题", rendered)
        self.assertIn("<li>", rendered)

    def test_render_report_log_text_keeps_recent_lines(self) -> None:
        rendered = _render_report_log_text(["[13:00:00] 第一条", "[13:00:01] 第二条"])

        self.assertIn("第一条", rendered)
        self.assertIn("第二条", rendered)
        self.assertIn("\n", rendered)


if __name__ == "__main__":
    unittest.main()
