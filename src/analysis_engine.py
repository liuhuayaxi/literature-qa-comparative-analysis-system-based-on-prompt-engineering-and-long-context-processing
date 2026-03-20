"""Document analysis, field extraction, and comparison reporting services."""

from __future__ import annotations

import asyncio
import csv
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Literal

from .config import AppConfig
from .errors import AppServiceError, OperationCancelledError, OperationPausedError, ProviderRequestError
from .llm_tools import (
    build_compare_prompt,
    build_data_extraction_prompt,
    build_recursive_summary_prompt,
    build_single_analysis_prompt,
    build_table_summary_prompt,
    build_window_summary_prompt,
    invoke_chat_text,
)


ProgressCallback = Callable[[str], Awaitable[None] | None]
AnalysisStateCallback = Callable[[dict[str, Any]], Awaitable[None] | None]
from .models import (
    ChunkCitation,
    ComparisonReport,
    DocumentExtractionResult,
    ExtractedFieldValue,
    ExtractionFieldSpec,
    SingleDocAnalysis,
    SourceDocument,
)
from .app_utils import (
    JsonResultCache,
    append_report_citation_sections,
    build_cache_key,
    build_course_signature,
    build_file_signature,
    build_comparison_markdown,
    detect_language,
    estimate_token_count,
    format_citation_line,
    new_report_id,
    postprocess_report_markdown,
    resolve_output_language,
    check_task_cancelled,
    check_task_paused,
    load_json_mapping,
    sanitize_storage_key,
    save_json_mapping,
    split_text_into_token_windows,
    trim_text_to_token_limit,
)


_ANALYSIS_CACHE_NAMESPACE = "analysis_v2"
_ANALYSIS_CACHE_SCHEMA = "content_v2"


def save_report(report: ComparisonReport, output_dir: str | Path | None = None) -> ComparisonReport:
    """Persist one Markdown report to disk and return the resolved output path."""

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "reports"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(report.output_path)
    if not output_path.is_absolute():
        output_path = output_dir / output_path
    output_path.write_text(report.markdown, encoding="utf-8")
    return report.model_copy(update={"output_path": str(output_path.resolve())})


def extract_json_object(raw_text: str) -> dict[str, Any]:
    """Parse the first JSON object found inside a model response."""

    raw_text = raw_text.strip()
    if not raw_text:
        return {}
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    candidate = raw_text[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return {}


def parse_extraction_field_specs(raw_text: str) -> list[ExtractionFieldSpec]:
    """Parse user field definitions from a multiline textbox.

    Supported line formats:
    - `字段名`
    - `字段名 | 补充说明`
    - `字段名 | 补充说明 | 期望单位`
    """

    specs: list[ExtractionFieldSpec] = []
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split("|")]
        name = parts[0]
        if not name:
            continue
        instruction = parts[1] if len(parts) > 1 else ""
        expected_unit = parts[2] if len(parts) > 2 else ""
        specs.append(ExtractionFieldSpec(name=name, instruction=instruction, expected_unit=expected_unit))
    return specs


def save_csv_rows(rows: list[dict[str, str]], output_path: str | Path) -> str:
    """Persist structured extraction rows to a CSV file and return the resolved path."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "field_name",
        "doc_title",
        "value",
        "normalized_value",
        "unit",
        "source_unit",
        "converted",
        "status",
        "notes",
        "source_file",
        "locator",
        "chunk_id",
        "evidence_quote",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return str(path.resolve())


class SingleDocumentAnalysisService:
    """Produce a structured summary for one document from its indexed chunks."""

    def __init__(self, config: AppConfig, vector_store) -> None:
        self.config = config
        self.vector_store = vector_store
        self.result_cache = JsonResultCache(config.cache_dir)

    async def analyze_document(
        self,
        course_id: str,
        doc_id: str,
        output_language: Literal["auto", "zh", "en"] = "auto",
        progress_callback: ProgressCallback | None = None,
        resume_state: dict[str, Any] | None = None,
        state_callback: AnalysisStateCallback | None = None,
    ) -> SingleDocAnalysis:
        """Run the long-document analysis pipeline for one indexed document."""

        check_task_cancelled()
        await _emit_progress(progress_callback, f"正在读取文档切片: {doc_id}")
        documents = await self.vector_store.get_document_chunks(course_id=course_id, doc_id=doc_id)
        if not documents:
            raise AppServiceError(
                f"未找到文档切片：{doc_id}" if output_language != "en" else f"No document chunks found for doc_id={doc_id}",
                code="document_not_found",
            )
        text = "\n\n".join(doc.page_content for doc in documents)
        language = resolve_output_language(output_language, text)
        title = str(documents[0].metadata.get("file_name", doc_id))
        prompt_template = self.config.single_analysis_prompt_zh if language == "zh" else self.config.single_analysis_prompt_en
        cache_namespace = _ANALYSIS_CACHE_NAMESPACE
        cache_key = build_cache_key(
            _stable_analysis_cache_state(documents),
            language,
            prompt_template,
        )
        legacy_cache_namespace = f"analysis_{course_id}"
        legacy_cache_key = build_cache_key(
            _legacy_document_cache_state(documents),
            language,
            prompt_template,
        )
        if self.config.enable_result_cache:
            cached = _read_cached_payload(
                self.result_cache,
                primary_namespace=cache_namespace,
                primary_key=cache_key,
                legacy_namespace=legacy_cache_namespace,
                legacy_key=legacy_cache_key,
            )
            if isinstance(cached, dict) and cached:
                await _emit_progress(progress_callback, f"命中分析缓存，直接返回结果: {title}")
                return SingleDocAnalysis.model_validate(cached)
        if self.config.has_chat_model_credentials:
            check_task_cancelled()
            await _emit_progress(progress_callback, f"正在准备长文档分析上下文: {title}")
            prepared_text = await self._prepare_analysis_text(
                title=title,
                text=text,
                language=language,
                progress_callback=progress_callback,
                resume_state=resume_state,
                state_callback=state_callback,
            )
            check_task_cancelled()
            await _emit_progress(progress_callback, f"正在调用模型分析文档: {title}")
            result = await self._llm_analysis(
                title=title,
                text=prepared_text,
                language=language,
                progress_callback=progress_callback,
            )
            if result:
                await _emit_progress(progress_callback, f"文档分析完成: {title}")
                analysis = SingleDocAnalysis(
                    doc_id=doc_id,
                    title=title,
                    language=detect_language(text),
                    summary=str(result.get("summary", "")),
                    sentiment=_normalize_sentiment(str(result.get("sentiment", "neutral"))),
                    keywords=_normalize_list(result.get("keywords")),
                    main_topics=_normalize_list(result.get("main_topics")),
                    risk_points=_normalize_list(result.get("risk_points")),
                )
                if self.config.enable_result_cache:
                    self.result_cache.set(cache_namespace, cache_key, analysis.model_dump())
                return analysis
        await _emit_progress(progress_callback, f"模型不可用，使用本地规则分析: {title}")
        analysis = self._heuristic_analysis(doc_id=doc_id, title=title, text=text)
        if self.config.enable_result_cache:
            self.result_cache.set(cache_namespace, cache_key, analysis.model_dump())
        return analysis

    async def inspect_analysis_cache(
        self,
        course_id: str,
        doc_id: str,
        output_language: Literal["auto", "zh", "en"] = "auto",
    ) -> dict[str, Any]:
        """Check whether one document already has a reusable analysis cache entry."""

        documents = await self.vector_store.get_document_chunks(course_id=course_id, doc_id=doc_id)
        if not documents:
            raise AppServiceError(
                f"未找到文档切片：{doc_id}" if output_language != "en" else f"No document chunks found for doc_id={doc_id}",
                code="document_not_found",
            )
        text = "\n\n".join(doc.page_content for doc in documents)
        language = resolve_output_language(output_language, text)
        title = str(documents[0].metadata.get("file_name", doc_id))
        prompt_template = self.config.single_analysis_prompt_zh if language == "zh" else self.config.single_analysis_prompt_en
        cache_namespace = _ANALYSIS_CACHE_NAMESPACE
        cache_key = build_cache_key(
            _stable_analysis_cache_state(documents),
            language,
            prompt_template,
        )
        legacy_cache_namespace = f"analysis_{course_id}"
        legacy_cache_key = build_cache_key(
            _legacy_document_cache_state(documents),
            language,
            prompt_template,
        )
        cached = bool(
            self.config.enable_result_cache
            and _read_cached_payload(
                self.result_cache,
                primary_namespace=cache_namespace,
                primary_key=cache_key,
                legacy_namespace=legacy_cache_namespace,
                legacy_key=legacy_cache_key,
            )
        )
        return {
            "doc_id": doc_id,
            "title": title,
            "language": language,
            "cached": cached,
        }

    async def clear_analysis_cache(
        self,
        course_id: str,
        doc_id: str,
        output_language: Literal["auto", "zh", "en"] = "auto",
    ) -> bool:
        """Remove the analysis cache entry for one document, if it exists."""

        documents = await self.vector_store.get_document_chunks(course_id=course_id, doc_id=doc_id)
        if not documents:
            return False
        text = "\n\n".join(doc.page_content for doc in documents)
        language = resolve_output_language(output_language, text)
        prompt_template = self.config.single_analysis_prompt_zh if language == "zh" else self.config.single_analysis_prompt_en
        cache_namespace = _ANALYSIS_CACHE_NAMESPACE
        cache_key = build_cache_key(
            _stable_analysis_cache_state(documents),
            language,
            prompt_template,
        )
        legacy_cache_namespace = f"analysis_{course_id}"
        legacy_cache_key = build_cache_key(
            _legacy_document_cache_state(documents),
            language,
            prompt_template,
        )
        path = self.result_cache._path(cache_namespace, cache_key)
        legacy_path = self.result_cache._path(legacy_cache_namespace, legacy_cache_key)
        removed = False
        if path.exists():
            path.unlink(missing_ok=True)
            removed = True
        if legacy_path.exists():
            legacy_path.unlink(missing_ok=True)
            removed = True
        return removed

    async def extract_document_fields(
        self,
        course_id: str,
        doc_id: str,
        field_specs: list[ExtractionFieldSpec],
        output_language: Literal["auto", "zh", "en"] = "auto",
        progress_callback: ProgressCallback | None = None,
    ) -> DocumentExtractionResult:
        """Extract user-defined comparison fields from one document with evidence."""

        check_task_cancelled()
        normalized_specs = [spec for spec in field_specs if spec.name.strip()]
        if not normalized_specs:
            raise AppServiceError("请至少填写一个目标字段。", code="empty_field_specs")
        documents = await self.vector_store.get_document_chunks(course_id=course_id, doc_id=doc_id)
        if not documents:
            raise AppServiceError(
                f"未找到文档切片：{doc_id}" if output_language != "en" else f"No document chunks found for doc_id={doc_id}",
                code="document_not_found",
            )
        text = "\n\n".join(doc.page_content for doc in documents)
        language = resolve_output_language(output_language, text)
        title = str(documents[0].metadata.get("file_name", doc_id))
        cache_namespace = f"extraction_{course_id}"
        cache_key = build_cache_key(
            _document_cache_state(documents),
            language,
            [(spec.name, spec.instruction, spec.expected_unit) for spec in normalized_specs],
            self.config.data_extraction_prompt_zh if language == "zh" else self.config.data_extraction_prompt_en,
        )
        if self.config.enable_result_cache:
            cached = self.result_cache.get(cache_namespace, cache_key)
            if isinstance(cached, dict) and cached:
                await _emit_progress(progress_callback, f"命中字段抽取缓存，直接返回结果: {title}")
                return _validate_document_extraction_result(cached)
        if self.config.has_chat_model_credentials:
            try:
                check_task_cancelled()
                await _emit_progress(progress_callback, f"正在抽取定向字段: {title}")
                llm_result = await self._llm_extract_fields(
                    title=title,
                    documents=documents,
                    field_specs=normalized_specs,
                    language=language,
                    progress_callback=progress_callback,
                )
                if llm_result.fields:
                    if self.config.enable_result_cache:
                        self.result_cache.set(cache_namespace, cache_key, llm_result.model_dump())
                    await _emit_progress(progress_callback, f"字段抽取完成: {title}")
                    return llm_result
            except ProviderRequestError:
                pass
        await _emit_progress(progress_callback, f"模型抽取不可用，使用本地规则抽取: {title}")
        extraction = self._heuristic_extract_fields(
            doc_id=doc_id,
            title=title,
            documents=documents,
            field_specs=normalized_specs,
            language=language,
        )
        if self.config.enable_result_cache:
            self.result_cache.set(cache_namespace, cache_key, extraction.model_dump())
        return extraction

    async def inspect_extraction_cache(
        self,
        course_id: str,
        doc_id: str,
        field_specs: list[ExtractionFieldSpec],
        output_language: Literal["auto", "zh", "en"] = "auto",
    ) -> dict[str, Any]:
        """Check whether one document already has a reusable extraction cache entry."""

        normalized_specs = [spec for spec in field_specs if spec.name.strip()]
        if not normalized_specs:
            raise AppServiceError("请至少填写一个目标字段。", code="empty_field_specs")
        documents = await self.vector_store.get_document_chunks(course_id=course_id, doc_id=doc_id)
        if not documents:
            raise AppServiceError(
                f"未找到文档切片：{doc_id}" if output_language != "en" else f"No document chunks found for doc_id={doc_id}",
                code="document_not_found",
            )
        text = "\n\n".join(doc.page_content for doc in documents)
        language = resolve_output_language(output_language, text)
        title = str(documents[0].metadata.get("file_name", doc_id))
        cache_namespace = f"extraction_{course_id}"
        cache_key = build_cache_key(
            _document_cache_state(documents),
            language,
            [(spec.name, spec.instruction, spec.expected_unit) for spec in normalized_specs],
            self.config.data_extraction_prompt_zh if language == "zh" else self.config.data_extraction_prompt_en,
        )
        cached = bool(self.config.enable_result_cache and self.result_cache.get(cache_namespace, cache_key))
        return {
            "doc_id": doc_id,
            "title": title,
            "language": language,
            "cached": cached,
        }

    async def _llm_analysis(
        self,
        title: str,
        text: str,
        language: Literal["zh", "en"],
        progress_callback: ProgressCallback | None = None,
    ) -> dict:
        prompt = build_single_analysis_prompt(
            text=text,
            title=title,
            language=language,
            instruction_override=(
                self.config.single_analysis_prompt_zh
                if language == "zh"
                else self.config.single_analysis_prompt_en
            ),
        )
        content = await _invoke_chat_with_progress(self.config, prompt, language, progress_callback)
        return extract_json_object(str(content))

    async def _llm_extract_fields(
        self,
        *,
        title: str,
        documents: list[SourceDocument],
        field_specs: list[ExtractionFieldSpec],
        language: Literal["zh", "en"],
        progress_callback: ProgressCallback | None = None,
    ) -> DocumentExtractionResult:
        selected_chunks = _select_candidate_chunks_for_fields(documents, field_specs, max_chunks=12, per_field=3)
        prompt = build_data_extraction_prompt(
            title=title,
            fields=field_specs,
            chunks=selected_chunks,
            language=language,
            instruction_override=(
                self.config.data_extraction_prompt_zh
                if language == "zh"
                else self.config.data_extraction_prompt_en
            ),
        )
        content = await _invoke_chat_with_progress(self.config, prompt, language, progress_callback)
        payload = extract_json_object(str(content))
        raw_items = payload.get("fields") if isinstance(payload, dict) else []
        by_chunk_id = {
            str(doc.metadata.get("chunk_id", "")): doc
            for doc in selected_chunks
        }
        extracted_by_name: dict[str, ExtractedFieldValue] = {}
        for raw_item in raw_items if isinstance(raw_items, list) else []:
            if not isinstance(raw_item, dict):
                continue
            field_name = _match_field_name(str(raw_item.get("field_name", "")), field_specs)
            if not field_name:
                continue
            matched_spec = next((spec for spec in field_specs if spec.name == field_name), None)
            if matched_spec is None:
                continue
            chunk_id = str(raw_item.get("evidence_chunk_id", "")).strip()
            source_doc = by_chunk_id.get(chunk_id)
            raw_value, source_unit = _coerce_value_and_unit(
                str(raw_item.get("value", "")).strip(),
                str(raw_item.get("unit", "")).strip(),
            )
            raw_value = _sanitize_field_value(raw_value, expected_unit=matched_spec.expected_unit)
            if _prefers_textual_extraction(matched_spec):
                raw_value = _clean_textual_field_candidate(raw_value, matched_spec)
            converted_value, final_unit, conversion_note, converted = _convert_value_to_expected_unit(
                raw_value,
                source_unit=source_unit,
                expected_unit=matched_spec.expected_unit,
                language=language,
            )
            existing_notes = str(raw_item.get("notes", "")).strip()
            note_parts = [part for part in [existing_notes, conversion_note] if part]
            raw_normalized_value = str(raw_item.get("normalized_value", "")).strip()
            if converted and converted_value:
                normalized_value = converted_value
            else:
                normalized_value = raw_normalized_value or converted_value or raw_value
            if _prefers_textual_extraction(matched_spec):
                normalized_value = _clean_textual_field_candidate(normalized_value, matched_spec)
            model_value = ExtractedFieldValue(
                field_name=field_name,
                value=raw_value,
                normalized_value=_normalize_extracted_value(
                    normalized_value,
                    unit=final_unit,
                ),
                unit=final_unit,
                source_unit=source_unit,
                converted=converted,
                status=_normalize_extraction_status(str(raw_item.get("status", "not_found"))),
                notes=" ".join(note_parts).strip(),
                source_file=str(source_doc.metadata.get("file_name", "")) if source_doc else "",
                page_label=str(source_doc.metadata.get("page_label")) if source_doc and source_doc.metadata.get("page_label") else None,
                section_label=str(source_doc.metadata.get("section_label")) if source_doc and source_doc.metadata.get("section_label") else None,
                chunk_id=chunk_id,
                evidence_quote=source_doc.page_content if source_doc else "",
            )
            extracted_by_name[field_name] = _validate_llm_field_candidate(
                field_spec=matched_spec,
                model_value=model_value,
                source_doc=source_doc,
                documents=documents,
                language=language,
            )
        return DocumentExtractionResult(
            doc_id=str(documents[0].metadata.get("doc_id", "")),
            title=title,
            fields=[
                extracted_by_name.get(
                    spec.name,
                    ExtractedFieldValue(
                        field_name=spec.name,
                        status="not_found",
                        notes="模型未返回该字段。" if language == "zh" else "The model did not return this field.",
                    ),
                )
                for spec in field_specs
            ],
        )

    async def _prepare_analysis_text(
        self,
        title: str,
        text: str,
        language: Literal["zh", "en"],
        progress_callback: ProgressCallback | None = None,
        resume_state: dict[str, Any] | None = None,
        state_callback: AnalysisStateCallback | None = None,
    ) -> str:
        target_budget = max(
            512,
            min(
                self.config.recursive_summary_target_tokens,
                self.config.model_context_window - self.config.answer_token_reserve - 512,
            ),
        )
        if estimate_token_count(text, self.config.chat_model) <= target_budget:
            return trim_text_to_token_limit(text, target_budget, self.config.chat_model)
        normalized_resume_state = resume_state if isinstance(resume_state, dict) else {}
        saved_window_summaries = normalized_resume_state.get("window_summaries", [])
        if isinstance(saved_window_summaries, list) and saved_window_summaries:
            await _emit_progress(
                progress_callback,
                f"检测到已保存的中间摘要，准备从已完成的 {len(saved_window_summaries)} 个窗口继续: {title}",
            )
        await _emit_progress(progress_callback, f"文档较长，正在滑窗摘要: {title}")
        window_summaries = await self._summarize_long_text(
            title=title,
            text=text,
            language=language,
            progress_callback=progress_callback,
            resume_state=normalized_resume_state,
            state_callback=state_callback,
        )
        return await self._collapse_summaries(
            title=title,
            summaries=window_summaries,
            language=language,
            token_budget=target_budget,
            progress_callback=progress_callback,
        )

    async def _summarize_long_text(
        self,
        title: str,
        text: str,
        language: Literal["zh", "en"],
        progress_callback: ProgressCallback | None = None,
        resume_state: dict[str, Any] | None = None,
        state_callback: AnalysisStateCallback | None = None,
    ) -> list[str]:
        windows = split_text_into_token_windows(
            text,
            window_tokens=max(256, self.config.long_context_window_tokens),
            overlap_tokens=max(0, self.config.long_context_window_overlap_tokens),
            model_name=self.config.chat_model,
        )
        summaries: list[str] = []
        total_windows = len(windows)
        normalized_resume_state = resume_state if isinstance(resume_state, dict) else {}
        saved_summaries = normalized_resume_state.get("window_summaries", [])
        if isinstance(saved_summaries, list):
            summaries.extend(str(item).strip() for item in saved_summaries[:total_windows] if str(item).strip())
        if summaries:
            await _emit_progress(
                progress_callback,
                f"已复用 {len(summaries)}/{total_windows} 个滑窗摘要，继续处理剩余窗口: {title}",
            )
        check_task_cancelled()
        for index, window_text in enumerate(windows[len(summaries) :], start=len(summaries) + 1):
            check_task_cancelled()
            await _emit_progress(progress_callback, f"正在处理滑窗 {index}/{total_windows}: {title}")
            prompt = build_window_summary_prompt(
                title=title,
                text=window_text,
                language=language,
                window_index=index,
                total_windows=total_windows,
            )
            summary = await _invoke_chat_with_progress(self.config, prompt, language, progress_callback)
            check_task_cancelled()
            if not summary.strip():
                summary = trim_text_to_token_limit(
                    window_text,
                    max(120, self.config.long_context_window_tokens // 2),
                    self.config.chat_model,
                )
            summaries.append(summary.strip())
            await _emit_analysis_state(
                state_callback,
                {
                    "phase": "window_summary",
                    "title": title,
                    "current_window": index,
                    "total_windows": total_windows,
                    "window_summaries": summaries[:],
                },
            )
        return summaries or [trim_text_to_token_limit(text, max(120, self.config.long_context_window_tokens), self.config.chat_model)]

    async def _collapse_summaries(
        self,
        title: str,
        summaries: list[str],
        language: Literal["zh", "en"],
        token_budget: int,
        progress_callback: ProgressCallback | None = None,
    ) -> str:
        current = [item.strip() for item in summaries if item.strip()]
        level = 1
        batch_size = max(2, int(self.config.recursive_summary_batch_size))
        while current:
            check_task_cancelled()
            joined = "\n\n".join(current)
            should_merge = estimate_token_count(joined, self.config.chat_model) > token_budget or len(current) > batch_size
            if not should_merge:
                return joined
            if len(current) == 1:
                return trim_text_to_token_limit(current[0], token_budget, self.config.chat_model)
            next_level: list[str] = []
            await _emit_progress(progress_callback, f"正在递归压缩中间摘要，第 {level} 轮: {title}")
            for start in range(0, len(current), batch_size):
                check_task_cancelled()
                batch = current[start : start + batch_size]
                prompt = build_recursive_summary_prompt(
                    title=title,
                    summaries=batch,
                    language=language,
                    level=level,
                )
                merged = await _invoke_chat_with_progress(self.config, prompt, language, progress_callback)
                check_task_cancelled()
                if not merged.strip():
                    merged = trim_text_to_token_limit(
                        "\n\n".join(batch),
                        max(120, token_budget // 2),
                        self.config.chat_model,
                    )
                next_level.append(merged.strip())
            current = next_level
            level += 1
        return ""

    def _heuristic_analysis(self, doc_id: str, title: str, text: str) -> SingleDocAnalysis:
        language = detect_language(text)
        keywords = _extract_keywords(text, language)
        sentences = _split_sentences(text)
        summary = " ".join(sentences[:2]).strip() or text[:240].strip()
        sentiment = _infer_sentiment(text)
        risk_points = _extract_risk_points(text, sentences)
        topics = keywords[:5]
        return SingleDocAnalysis(
            doc_id=doc_id,
            title=title,
            language=language,
            summary=summary,
            sentiment=sentiment,
            keywords=keywords[:8],
            main_topics=topics,
            risk_points=risk_points,
        )

    def _heuristic_extract_fields(
        self,
        *,
        doc_id: str,
        title: str,
        documents: list[SourceDocument],
        field_specs: list[ExtractionFieldSpec],
        language: Literal["zh", "en"],
    ) -> DocumentExtractionResult:
        """Fallback extraction based on lexical matching and simple numeric parsing."""

        fields: list[ExtractedFieldValue] = []
        for spec in field_specs:
            ranked = _rank_chunks_for_field(documents, spec)
            if not ranked:
                fields.append(
                    ExtractedFieldValue(
                        field_name=spec.name,
                        status="not_found",
                        notes="未找到相关切片。" if language == "zh" else "No relevant chunk was found.",
                    )
                )
                continue
            candidates = [doc for _, doc in ranked[:3]]
            evidence = _extract_value_from_chunks(candidates, spec, language)
            fields.append(evidence)
        return DocumentExtractionResult(doc_id=doc_id, title=title, fields=fields)


def _normalize_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [part.strip() for part in re.split(r"[,;\n]+", value) if part.strip()]
    return []


def _normalize_sentiment(value: str) -> Literal["positive", "neutral", "negative", "mixed"]:
    normalized = value.strip().lower()
    if normalized in {"positive", "neutral", "negative", "mixed"}:
        return normalized
    if normalized in {"积极", "正向"}:
        return "positive"
    if normalized in {"中性"}:
        return "neutral"
    if normalized in {"消极", "负向"}:
        return "negative"
    return "mixed"


def _split_sentences(text: str) -> list[str]:
    return [segment.strip() for segment in re.split(r"(?<=[。！？.!?])\s+", text) if segment.strip()]


def _extract_keywords(text: str, language: str) -> list[str]:
    if language == "zh":
        tokens = re.findall(r"[\u4e00-\u9fff]{2,}", text)
        stopwords = {"我们", "你们", "这个", "一种", "进行", "研究", "课程", "以及", "通过", "可以"}
    else:
        tokens = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text.lower())
        stopwords = {
            "the",
            "and",
            "with",
            "that",
            "this",
            "from",
            "into",
            "using",
            "were",
            "been",
            "their",
            "course",
        }
    counter = Counter(token for token in tokens if token not in stopwords)
    return [token for token, _ in counter.most_common(10)]


def _infer_sentiment(text: str) -> Literal["positive", "neutral", "negative", "mixed"]:
    positives = {"effective", "improve", "success", "robust", "efficient", "优秀", "提升", "有效", "显著"}
    negatives = {"risk", "issue", "fail", "limitation", "problem", "挑战", "风险", "问题", "局限"}
    lowered = text.lower()
    positive_hits = sum(1 for token in positives if token in lowered or token in text)
    negative_hits = sum(1 for token in negatives if token in lowered or token in text)
    if positive_hits and negative_hits:
        return "mixed"
    if positive_hits:
        return "positive"
    if negative_hits:
        return "negative"
    return "neutral"


def _extract_risk_points(text: str, sentences: Iterable[str]) -> list[str]:
    keywords = ("risk", "limitation", "issue", "problem", "挑战", "风险", "局限", "问题")
    matched = [sentence for sentence in sentences if any(keyword in sentence.lower() or keyword in sentence for keyword in keywords)]
    if matched:
        return matched[:5]
    fallback = text.strip()[:160]
    if fallback:
        return [fallback]
    return ["No explicit risk point identified."]


class BatchComparisonService:
    """Analyze multiple documents and synthesize a comparison report."""

    def __init__(
        self,
        config: AppConfig,
        vector_store,
        single_doc_service: SingleDocumentAnalysisService,
    ) -> None:
        self.config = config
        self.vector_store = vector_store
        self.single_doc_service = single_doc_service
        self.result_cache = JsonResultCache(config.cache_dir)
        self.checkpoint_dir = config.analysis_checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    async def compare_documents(
        self,
        course_id: str,
        doc_ids: list[str],
        output_language: Literal["auto", "zh", "en"] = "auto",
        target_fields: list[ExtractionFieldSpec] | None = None,
        export_csv: bool = True,
        progress_callback: ProgressCallback | None = None,
        resume: bool = False,
    ):
        """Run per-document analysis, then synthesize one combined report."""
        if not doc_ids:
            raise AppServiceError("请至少选择一篇文档。", code="empty_doc_selection")
        doc_ids = list(dict.fromkeys(doc_ids))
        normalized_target_fields = [spec for spec in (target_fields or []) if spec.name.strip()]
        checkpoint_key = self._build_compare_checkpoint_key(
            course_id=course_id,
            doc_ids=doc_ids,
            output_language=output_language,
            target_fields=normalized_target_fields,
            export_csv=export_csv,
        )
        checkpoint = self._load_compare_checkpoint(checkpoint_key)
        if checkpoint and not resume and str(checkpoint.get("status", "")) in {"running", "paused", "failed"}:
            await _emit_progress(
                progress_callback,
                "检测到当前选择有未完成的分析进度。点击“继续上次批量对比”可从断点继续；如果直接重新开始，本次会覆盖旧进度。",
            )
            checkpoint = {}
        document_records = await self.vector_store.list_documents(course_id)
        doc_title_map = {record.doc_id: record.file_name for record in document_records}
        cache_namespace = f"report_{course_id}"
        cache_key = build_cache_key(
            output_language,
            doc_ids,
            build_course_signature(course_id, root_dir=self.config.data_root),
            [(spec.name, spec.instruction, spec.expected_unit) for spec in normalized_target_fields],
            export_csv,
            self.config.compare_report_prompt_zh if output_language != "en" else self.config.compare_report_prompt_en,
            self.config.table_summary_prompt_zh if output_language != "en" else self.config.table_summary_prompt_en,
        )
        if self.config.enable_result_cache:
            cached = self.result_cache.get(cache_namespace, cache_key)
            if isinstance(cached, dict) and cached:
                cached_report = ComparisonReport.model_validate(cached)
                report_exists = bool(cached_report.output_path) and Path(cached_report.output_path).exists()
                csv_exists = not cached_report.csv_output_path or Path(cached_report.csv_output_path).exists()
                if report_exists and csv_exists:
                    await _emit_progress(
                        progress_callback,
                        f"命中整份报告缓存，本次直接复用全部 {len(doc_ids)} 篇文档的对比结果，无需重新分析。",
                    )
                    return cached_report
                await _emit_progress(progress_callback, "检测到旧缓存中的报告文件已失效，本次将重新生成。")

        checkpoint = self._prepare_compare_checkpoint(
            checkpoint_key=checkpoint_key,
            existing=checkpoint,
            course_id=course_id,
            doc_ids=doc_ids,
            output_language=output_language,
            target_fields=normalized_target_fields,
            export_csv=export_csv,
        )
        if resume and checkpoint.get("resumed_from_checkpoint"):
            await _emit_progress(progress_callback, "已读取上次中断的检查点，准备从断点继续。")
        self._save_compare_checkpoint(checkpoint)
        checkpoint_lock = asyncio.Lock()

        resumed_analysis_doc_ids = set(checkpoint.get("analyses", {}).keys())
        resumed_extraction_doc_ids = set(checkpoint.get("extractions", {}).keys())
        for doc_id, payload in checkpoint.get("analyses", {}).items():
            doc_title_map[doc_id] = str(payload.get("title", doc_id))
        for doc_id, payload in checkpoint.get("extractions", {}).items():
            doc_title_map[doc_id] = str(payload.get("title", doc_id))

        semaphore = asyncio.Semaphore(self.config.batch_concurrency)
        total_docs = len(doc_ids)
        analysis_cached_doc_ids: set[str] = set()
        extraction_cached_doc_ids: set[str] = set()

        async def save_doc_progress(
            *,
            doc_id: str,
            title: str,
            stage: str,
            status: str,
            detail: str,
            index: int,
            stage_payload: dict[str, Any] | None = None,
        ) -> None:
            async with checkpoint_lock:
                entry = self._ensure_doc_progress_entry(checkpoint, doc_id=doc_id, title=title)
                entry["stage"] = stage
                entry["status"] = status
                entry["detail"] = detail
                entry["index"] = index
                entry["total_docs"] = total_docs
                entry["updated_at"] = _now_iso()
                if stage_payload is not None:
                    if stage_payload:
                        entry[f"{stage}_state"] = stage_payload
                    else:
                        entry.pop(f"{stage}_state", None)
                if status in {"completed", "cached", "resumed"}:
                    entry.pop("current_window", None)
                    entry.pop("total_windows", None)
                else:
                    self._update_doc_progress_window(entry, detail)
                checkpoint["last_stage"] = stage
                checkpoint["last_doc_id"] = doc_id
                checkpoint["last_message"] = f"{title}: {detail}"
                self._save_compare_checkpoint(checkpoint)

        try:
            if self.config.enable_result_cache:
                await _emit_progress(progress_callback, f"正在检查单文档分析缓存，共 {total_docs} 篇文档。")

                async def inspect_analysis_cache(index: int, doc_id: str):
                    check_task_cancelled()
                    title = doc_title_map.get(doc_id, doc_id)
                    await _emit_progress(
                        progress_callback,
                        f"[缓存检查 {index}/{total_docs}] 分析缓存：{_truncate_progress_title(title)}",
                    )
                    if doc_id in resumed_analysis_doc_ids:
                        return index, doc_id, {"doc_id": doc_id, "title": title, "cached": False}
                    info = await self.single_doc_service.inspect_analysis_cache(
                        course_id=course_id,
                        doc_id=doc_id,
                        output_language=output_language,
                    )
                    return index, doc_id, info

                analysis_cache_results = await asyncio.gather(
                    *(inspect_analysis_cache(index, doc_id) for index, doc_id in enumerate(doc_ids, start=1))
                )
                for _, doc_id, info in analysis_cache_results:
                    doc_title_map[doc_id] = str(info.get("title", doc_title_map.get(doc_id, doc_id)))
                    if info.get("cached"):
                        analysis_cached_doc_ids.add(doc_id)
                await _emit_progress(
                    progress_callback,
                    f"单文档分析缓存检查完成：可直接复用 {len(analysis_cached_doc_ids)}/{total_docs} 篇，"
                    f"断点已完成 {len(resumed_analysis_doc_ids)}/{total_docs} 篇，"
                    f"仍需实际分析 {total_docs - len(analysis_cached_doc_ids) - len(resumed_analysis_doc_ids)}/{total_docs} 篇。",
                )
                if normalized_target_fields:
                    await _emit_progress(progress_callback, f"正在检查字段抽取缓存，共 {total_docs} 篇文档。")

                    async def inspect_extraction_cache(index: int, doc_id: str):
                        check_task_cancelled()
                        title = doc_title_map.get(doc_id, doc_id)
                        await _emit_progress(
                            progress_callback,
                            f"[缓存检查 {index}/{total_docs}] 字段缓存：{_truncate_progress_title(title)}",
                        )
                        if doc_id in resumed_extraction_doc_ids:
                            return index, doc_id, {"doc_id": doc_id, "title": title, "cached": False}
                        info = await self.single_doc_service.inspect_extraction_cache(
                            course_id=course_id,
                            doc_id=doc_id,
                            field_specs=normalized_target_fields,
                            output_language=output_language,
                        )
                        return index, doc_id, info

                    extraction_cache_results = await asyncio.gather(
                        *(inspect_extraction_cache(index, doc_id) for index, doc_id in enumerate(doc_ids, start=1))
                    )
                    for _, doc_id, info in extraction_cache_results:
                        doc_title_map[doc_id] = str(info.get("title", doc_title_map.get(doc_id, doc_id)))
                        if info.get("cached"):
                            extraction_cached_doc_ids.add(doc_id)
                    await _emit_progress(
                        progress_callback,
                        f"字段抽取缓存检查完成：可直接复用 {len(extraction_cached_doc_ids)}/{total_docs} 篇，"
                        f"断点已完成 {len(resumed_extraction_doc_ids)}/{total_docs} 篇，"
                        f"仍需实际抽取 {total_docs - len(extraction_cached_doc_ids) - len(resumed_extraction_doc_ids)}/{total_docs} 篇。",
                    )

            analysis_pending_total = max(0, total_docs - len(analysis_cached_doc_ids) - len(resumed_analysis_doc_ids))
            extraction_pending_total = (
                max(0, total_docs - len(extraction_cached_doc_ids) - len(resumed_extraction_doc_ids))
                if normalized_target_fields
                else 0
            )
            final_stage_count = 3 + (1 if normalized_target_fields else 0)
            total_units = analysis_pending_total + extraction_pending_total + final_stage_count
            progress_lock = asyncio.Lock()
            progress_state = {
                "analysis_pending_completed": 0,
                "analysis_cached_completed": 0,
                "analysis_resumed_completed": len(resumed_analysis_doc_ids),
                "extraction_pending_completed": 0,
                "extraction_cached_completed": 0,
                "extraction_resumed_completed": len(resumed_extraction_doc_ids),
                "final_completed": 0,
            }

            await _emit_progress(
                progress_callback,
                f"已开始批量对比，共 {total_docs} 篇文档。接下来会依次进行单文档分析"
                + ("、字段抽取" if normalized_target_fields else "")
                + "、生成对比报告。",
            )

            async def emit_doc_stage(
                stage_label: str,
                index: int,
                doc_id: str,
                detail: str,
                *,
                stage_status: str,
            ) -> None:
                title = doc_title_map.get(doc_id, doc_id)
                cleaned_detail = _clean_batch_progress_detail(detail, title=title, doc_id=doc_id)
                async with progress_lock:
                    completed_units = (
                        int(progress_state["analysis_pending_completed"])
                        + int(progress_state["extraction_pending_completed"])
                        + int(progress_state["final_completed"])
                    )
                await _emit_progress(
                    progress_callback,
                    _format_batch_doc_progress(
                        stage_label=stage_label,
                        index=index,
                        total_docs=total_docs,
                        title=title,
                        stage_status=stage_status,
                        detail=cleaned_detail,
                        completed_units=completed_units,
                        total_units=total_units,
                    ),
                )
                checkpoint_stage = "analysis" if stage_label == "单文档分析" else "extraction"
                if "无需重新" in cleaned_detail:
                    checkpoint_status = "resumed"
                elif "命中已有" in cleaned_detail or "直接复用" in cleaned_detail:
                    checkpoint_status = "cached"
                elif "完成" in cleaned_detail:
                    checkpoint_status = "completed"
                else:
                    checkpoint_status = "running"
                await save_doc_progress(
                    doc_id=doc_id,
                    title=title,
                    stage=checkpoint_stage,
                    status=checkpoint_status,
                    detail=cleaned_detail,
                    index=index,
                )

            async def emit_analysis_stage(index: int, doc_id: str, detail: str) -> None:
                async with progress_lock:
                    stage_status = _format_stage_status(
                        pending_label="待分析",
                        pending_done=int(progress_state["analysis_pending_completed"]),
                        pending_total=analysis_pending_total,
                        cached_label="已复用缓存",
                        cached_done=int(progress_state["analysis_cached_completed"]),
                        cached_total=len(analysis_cached_doc_ids),
                        resumed_label="断点已完成",
                        resumed_done=int(progress_state["analysis_resumed_completed"]),
                        resumed_total=len(resumed_analysis_doc_ids),
                    )
                await emit_doc_stage("单文档分析", index, doc_id, detail, stage_status=stage_status)

            async def emit_extraction_stage(index: int, doc_id: str, detail: str) -> None:
                async with progress_lock:
                    stage_status = _format_stage_status(
                        pending_label="待抽取",
                        pending_done=int(progress_state["extraction_pending_completed"]),
                        pending_total=extraction_pending_total,
                        cached_label="已复用缓存",
                        cached_done=int(progress_state["extraction_cached_completed"]),
                        cached_total=len(extraction_cached_doc_ids),
                        resumed_label="断点已完成",
                        resumed_done=int(progress_state["extraction_resumed_completed"]),
                        resumed_total=len(resumed_extraction_doc_ids),
                    )
                await emit_doc_stage("字段抽取", index, doc_id, detail, stage_status=stage_status)

            async def mark_global_stage(stage_label: str, detail: str) -> None:
                async with progress_lock:
                    progress_state["final_completed"] = int(progress_state["final_completed"]) + 1
                    completed_units = (
                        int(progress_state["analysis_pending_completed"])
                        + int(progress_state["extraction_pending_completed"])
                        + int(progress_state["final_completed"])
                    )
                await _emit_progress(
                    progress_callback,
                    _format_batch_global_progress(
                        stage_label=stage_label,
                        detail=detail,
                        completed_units=completed_units,
                        total_units=total_units,
                    ),
                )

            async def analyze(index: int, doc_id: str):
                async with semaphore:
                    check_task_cancelled()
                    if doc_id in resumed_analysis_doc_ids:
                        result = SingleDocAnalysis.model_validate(checkpoint["analyses"][doc_id])
                        await emit_analysis_stage(index, doc_id, "已从上次的检查点恢复，本篇无需重新分析。")
                        return result
                    if doc_id in analysis_cached_doc_ids:
                        check_task_cancelled()
                        result = await self.single_doc_service.analyze_document(
                            course_id=course_id,
                            doc_id=doc_id,
                            output_language=output_language,
                            progress_callback=None,
                        )
                        async with progress_lock:
                            progress_state["analysis_cached_completed"] = int(progress_state["analysis_cached_completed"]) + 1
                            reused = int(progress_state["analysis_cached_completed"])
                        await emit_analysis_stage(
                            index,
                            doc_id,
                            f"命中已有分析缓存，本篇直接复用。当前已复用 {reused}/{len(analysis_cached_doc_ids)} 篇缓存。",
                        )
                        return result
                    check_task_cancelled()
                    await emit_analysis_stage(index, doc_id, "已进入分析队列，准备读取文档切片。")
                    title = doc_title_map.get(doc_id, doc_id)

                    async def on_analysis_progress(message: str) -> None:
                        await emit_analysis_stage(index, doc_id, message)

                    async def on_analysis_state(state: dict[str, Any]) -> None:
                        current_title = str(state.get("title") or title)
                        current_window = int(state.get("current_window", 0) or 0)
                        total_windows = int(state.get("total_windows", 0) or 0)
                        if current_window and total_windows:
                            detail = f"中间摘要已保存，当前滑窗 {current_window}/{total_windows}。"
                        else:
                            detail = "中间摘要已保存，等待继续处理。"
                        await save_doc_progress(
                            doc_id=doc_id,
                            title=current_title,
                            stage="analysis",
                            status="running",
                            detail=detail,
                            index=index,
                            stage_payload=state,
                        )

                    check_task_cancelled()
                    result = await self.single_doc_service.analyze_document(
                        course_id=course_id,
                        doc_id=doc_id,
                        output_language=output_language,
                        progress_callback=on_analysis_progress,
                        resume_state=self._get_doc_stage_state(checkpoint, doc_id=doc_id, stage="analysis"),
                        state_callback=on_analysis_state,
                    )
                    async with checkpoint_lock:
                        checkpoint["analyses"][doc_id] = result.model_dump()
                        entry = self._ensure_doc_progress_entry(checkpoint, doc_id=doc_id, title=title)
                        entry.pop("analysis_state", None)
                        self._save_compare_checkpoint(checkpoint)
                    async with progress_lock:
                        progress_state["analysis_pending_completed"] = int(progress_state["analysis_pending_completed"]) + 1
                        done = int(progress_state["analysis_pending_completed"])
                    await emit_analysis_stage(
                        index,
                        doc_id,
                        f"本篇分析完成，已完成 {done}/{analysis_pending_total} 篇待分析文档。",
                    )
                    self._raise_if_pause_requested(checkpoint)
                    return result

            analyses = await asyncio.gather(*(analyze(index, doc_id) for index, doc_id in enumerate(doc_ids, start=1)))
            language = output_language if output_language in {"zh", "en"} else "zh"
            extraction_results_by_doc = {
                doc_id: _validate_document_extraction_result(payload)
                for doc_id, payload in checkpoint.get("extractions", {}).items()
            }

            if normalized_target_fields:
                await _emit_progress(
                    progress_callback,
                    _format_batch_global_progress(
                        stage_label="字段抽取",
                        detail=f"单文档分析已完成，接下来开始抽取 {len(normalized_target_fields)} 个目标字段。",
                        completed_units=int(progress_state["analysis_pending_completed"]),
                        total_units=total_units,
                    ),
                )

                async def extract(index: int, doc_id: str):
                    async with semaphore:
                        check_task_cancelled()
                        if doc_id in resumed_extraction_doc_ids:
                            result = _validate_document_extraction_result(checkpoint["extractions"][doc_id])
                            await emit_extraction_stage(index, doc_id, "已从上次的检查点恢复，本篇无需重新抽取。")
                            return result
                        if doc_id in extraction_cached_doc_ids:
                            check_task_cancelled()
                            result = await self.single_doc_service.extract_document_fields(
                                course_id=course_id,
                                doc_id=doc_id,
                                field_specs=normalized_target_fields,
                                output_language=output_language,
                                progress_callback=None,
                            )
                            async with progress_lock:
                                progress_state["extraction_cached_completed"] = int(progress_state["extraction_cached_completed"]) + 1
                                reused = int(progress_state["extraction_cached_completed"])
                            await emit_extraction_stage(
                                index,
                                doc_id,
                                f"命中已有字段缓存，本篇直接复用。当前已复用 {reused}/{len(extraction_cached_doc_ids)} 篇缓存。",
                            )
                            return result
                        check_task_cancelled()
                        await emit_extraction_stage(index, doc_id, "已进入字段抽取队列，准备读取相关切片。")
                        result = await self.single_doc_service.extract_document_fields(
                            course_id=course_id,
                            doc_id=doc_id,
                            field_specs=normalized_target_fields,
                            output_language=output_language,
                            progress_callback=lambda message: emit_extraction_stage(index, doc_id, message),
                        )
                        async with checkpoint_lock:
                            checkpoint["extractions"][doc_id] = result.model_dump()
                            entry = self._ensure_doc_progress_entry(
                                checkpoint,
                                doc_id=doc_id,
                                title=doc_title_map.get(doc_id, doc_id),
                            )
                            entry.pop("extraction_state", None)
                            self._save_compare_checkpoint(checkpoint)
                        async with progress_lock:
                            progress_state["extraction_pending_completed"] = int(progress_state["extraction_pending_completed"]) + 1
                            done = int(progress_state["extraction_pending_completed"])
                        await emit_extraction_stage(
                            index,
                            doc_id,
                            f"本篇字段抽取完成，已完成 {done}/{extraction_pending_total} 篇待抽取文档。",
                        )
                        self._raise_if_pause_requested(checkpoint)
                        return result

                extraction_results = await asyncio.gather(
                    *(extract(index, doc_id) for index, doc_id in enumerate(doc_ids, start=1))
                )
                extraction_results_by_doc = {
                    item.doc_id: _validate_document_extraction_result(item.model_dump())
                    for item in extraction_results
                }

            await _emit_progress(
                progress_callback,
                _format_batch_global_progress(
                    stage_label="报告准备",
                    detail="正在汇总关键差异、学习启发和可引用证据。",
                    completed_units=(
                        int(progress_state["analysis_pending_completed"])
                        + int(progress_state["extraction_pending_completed"])
                    ),
                    total_units=total_units,
                ),
            )
            key_differences = _build_key_differences(analyses, language=language)
            inspirations = _build_inspirations(analyses, language=language)
            citations = await self._collect_citations(course_id=course_id, doc_ids=doc_ids)
            checkpoint["last_stage"] = "prepare"
            checkpoint["last_message"] = "关键差异、学习启发和引用证据已整理完成。"
            self._save_compare_checkpoint(checkpoint)
            await mark_global_stage("报告准备", "关键差异、学习启发和引用证据已整理完成。")
            self._raise_if_pause_requested(checkpoint)

            report_id = str(checkpoint.get("report_id") or new_report_id())
            checkpoint["report_id"] = report_id
            timestamp = str(checkpoint.get("timestamp") or datetime.now().strftime("%Y%m%d_%H%M%S"))
            checkpoint["timestamp"] = timestamp
            file_name = str(checkpoint.get("output_file_name") or f"{course_id}_{timestamp}.md")
            checkpoint["output_file_name"] = file_name
            await _emit_progress(
                progress_callback,
                _format_batch_global_progress(
                    stage_label="报告生成",
                    detail=f"正在生成 Markdown 对比报告，共 {total_docs} 篇文档。",
                    completed_units=(
                        int(progress_state["analysis_pending_completed"])
                        + int(progress_state["extraction_pending_completed"])
                        + int(progress_state["final_completed"])
                    ),
                    total_units=total_units,
                ),
            )
            report = await self._build_report(
                course_id=course_id,
                analyses=analyses,
                key_differences=key_differences,
                inspirations=inspirations,
                citations=citations,
                output_path=file_name,
                report_id=report_id,
                language=language,
                progress_callback=progress_callback,
            )
            checkpoint["report"] = report.model_dump()
            checkpoint["last_stage"] = "report"
            checkpoint["last_message"] = "Markdown 报告正文已生成。"
            self._save_compare_checkpoint(checkpoint)
            await mark_global_stage("报告生成", "Markdown 报告正文已生成，正在整理附加结果。")
            self._raise_if_pause_requested(checkpoint)

            extraction_results = [extraction_results_by_doc[doc_id] for doc_id in doc_ids if doc_id in extraction_results_by_doc]
            csv_output_path: str | None = None
            if extraction_results and normalized_target_fields:
                await _emit_progress(
                    progress_callback,
                    _format_batch_global_progress(
                        stage_label="字段整理",
                        detail="正在追加字段对比表、复查证据和 CSV 导出信息。",
                        completed_units=(
                            int(progress_state["analysis_pending_completed"])
                            + int(progress_state["extraction_pending_completed"])
                            + int(progress_state["final_completed"])
                        ),
                        total_units=total_units,
                    ),
                )
                report = await self._append_extraction_sections(
                    report=report,
                    course_id=course_id,
                    analyses=analyses,
                    extraction_results=extraction_results,
                    target_fields=normalized_target_fields,
                    language=language,
                    export_csv=export_csv,
                    timestamp=timestamp,
                    progress_callback=progress_callback,
                )
                csv_output_path = report.csv_output_path
                checkpoint["report"] = report.model_dump()
                checkpoint["csv_output_path"] = csv_output_path or ""
                checkpoint["last_stage"] = "finalize"
                checkpoint["last_message"] = "字段对比表、证据区和 CSV 信息已整理完成。"
                self._save_compare_checkpoint(checkpoint)
                await mark_global_stage("字段整理", "字段对比表、证据区和 CSV 信息已整理完成。")
                self._raise_if_pause_requested(checkpoint)

            saved_report = save_report(report, output_dir=self.config.reports_dir)
            if self.config.enable_result_cache:
                self.result_cache.set(cache_namespace, cache_key, saved_report.model_dump())
            checkpoint["status"] = "completed"
            checkpoint["report"] = saved_report.model_dump()
            checkpoint["csv_output_path"] = saved_report.csv_output_path or ""
            checkpoint["last_stage"] = "completed"
            checkpoint["last_message"] = f"报告生成完成: {saved_report.output_path}"
            self._save_compare_checkpoint(checkpoint)
            await mark_global_stage("结果保存", f"报告生成完成: {saved_report.output_path}")
            if csv_output_path:
                return saved_report.model_copy(update={"csv_output_path": csv_output_path})
            return saved_report
        except Exception as exc:
            if isinstance(exc, OperationPausedError):
                checkpoint["status"] = "paused"
                checkpoint["last_message"] = str(exc.user_message)
                self._save_compare_checkpoint(checkpoint)
            elif isinstance(exc, OperationCancelledError):
                checkpoint["status"] = "cancelled"
                checkpoint["last_message"] = str(exc.user_message)
                self._save_compare_checkpoint(checkpoint)
            else:
                self._mark_checkpoint_failed(checkpoint, exc)
            raise

    async def inspect_compare_checkpoint(
        self,
        course_id: str,
        doc_ids: list[str],
        output_language: Literal["auto", "zh", "en"] = "auto",
        target_fields: list[ExtractionFieldSpec] | None = None,
        export_csv: bool = True,
    ) -> dict[str, Any]:
        """Return checkpoint availability and progress for the current batch selection."""

        doc_ids = list(dict.fromkeys(doc_ids))
        checkpoint_key = self._build_compare_checkpoint_key(
            course_id=course_id,
            doc_ids=doc_ids,
            output_language=output_language,
            target_fields=[spec for spec in (target_fields or []) if spec.name.strip()],
            export_csv=export_csv,
        )
        checkpoint = self._load_compare_checkpoint(checkpoint_key)
        if not checkpoint:
            return {"exists": False}
        total_docs = len(doc_ids)
        analysis_done = len(checkpoint.get("analyses", {}))
        extraction_done = len(checkpoint.get("extractions", {}))
        doc_progress = checkpoint.get("doc_progress", {})
        active_docs: list[dict[str, Any]] = []
        if isinstance(doc_progress, dict):
            for doc_id in doc_ids:
                entry = doc_progress.get(doc_id)
                if not isinstance(entry, dict):
                    continue
                if str(entry.get("status", "")) in {"completed", "cached", "resumed"}:
                    continue
                active_docs.append(
                    {
                        "doc_id": doc_id,
                        "title": str(entry.get("title", doc_id)),
                        "stage": str(entry.get("stage", "")),
                        "status": str(entry.get("status", "")),
                        "detail": str(entry.get("detail", "")),
                        "current_window": int(entry.get("current_window", 0) or 0),
                        "total_windows": int(entry.get("total_windows", 0) or 0),
                    }
                )
        return {
            "exists": True,
            "status": str(checkpoint.get("status", "")),
            "updated_at": str(checkpoint.get("updated_at", "")),
            "analysis_done": analysis_done,
            "extraction_done": extraction_done,
            "analysis_remaining": max(0, total_docs - analysis_done),
            "extraction_remaining": max(0, total_docs - extraction_done),
            "total_docs": total_docs,
            "last_stage": str(checkpoint.get("last_stage", "")),
            "last_message": str(checkpoint.get("last_message", "")),
            "progress_output_path": str(checkpoint.get("progress_output_path", "")),
            "path": str(self._checkpoint_path(checkpoint_key)),
            "active_docs": active_docs,
        }

    async def clear_compare_checkpoint(
        self,
        course_id: str,
        doc_ids: list[str],
        output_language: Literal["auto", "zh", "en"] = "auto",
        target_fields: list[ExtractionFieldSpec] | None = None,
        export_csv: bool = True,
    ) -> dict[str, Any]:
        """Remove the saved resume checkpoint for the current batch selection."""

        doc_ids = list(dict.fromkeys(doc_ids))
        checkpoint_key = self._build_compare_checkpoint_key(
            course_id=course_id,
            doc_ids=doc_ids,
            output_language=output_language,
            target_fields=[spec for spec in (target_fields or []) if spec.name.strip()],
            export_csv=export_csv,
        )
        checkpoint = self._load_compare_checkpoint(checkpoint_key)
        if not checkpoint:
            return {"removed": False, "reason": "missing"}
        progress_output_path = str(checkpoint.get("progress_output_path", "")).strip()
        checkpoint_path = self._checkpoint_path(checkpoint_key)
        checkpoint_path.unlink(missing_ok=True)
        if progress_output_path:
            Path(progress_output_path).unlink(missing_ok=True)
        return {
            "removed": True,
            "checkpoint_path": str(checkpoint_path),
            "progress_output_path": progress_output_path,
        }

    def _build_compare_checkpoint_key(
        self,
        *,
        course_id: str,
        doc_ids: list[str],
        output_language: Literal["auto", "zh", "en"],
        target_fields: list[ExtractionFieldSpec],
        export_csv: bool,
    ) -> str:
        return build_cache_key(
            "compare_checkpoint",
            course_id,
            doc_ids,
            output_language,
            [(spec.name, spec.instruction, spec.expected_unit) for spec in target_fields],
            export_csv,
        )

    def _checkpoint_path(self, checkpoint_key: str) -> Path:
        return self.checkpoint_dir / f"{sanitize_storage_key(checkpoint_key)}.json"

    def _load_compare_checkpoint(self, checkpoint_key: str) -> dict[str, Any]:
        return load_json_mapping(self._checkpoint_path(checkpoint_key))

    def _ensure_doc_progress_entry(
        self,
        checkpoint: dict[str, Any],
        *,
        doc_id: str,
        title: str,
    ) -> dict[str, Any]:
        doc_progress = checkpoint.setdefault("doc_progress", {})
        entry = doc_progress.setdefault(doc_id, {})
        entry["doc_id"] = doc_id
        entry["title"] = title
        return entry

    def _get_doc_stage_state(
        self,
        checkpoint: dict[str, Any],
        *,
        doc_id: str,
        stage: str,
    ) -> dict[str, Any] | None:
        doc_progress = checkpoint.get("doc_progress", {})
        if not isinstance(doc_progress, dict):
            return None
        entry = doc_progress.get(doc_id)
        if not isinstance(entry, dict):
            return None
        state = entry.get(f"{stage}_state")
        return state if isinstance(state, dict) else None

    def _update_doc_progress_window(self, entry: dict[str, Any], detail: str) -> None:
        match = re.search(r"滑窗\s*(\d+)\s*/\s*(\d+)", detail)
        if match:
            entry["current_window"] = int(match.group(1))
            entry["total_windows"] = int(match.group(2))
            return
        entry.pop("current_window", None)
        entry.pop("total_windows", None)

    def _prepare_compare_checkpoint(
        self,
        *,
        checkpoint_key: str,
        existing: dict[str, Any],
        course_id: str,
        doc_ids: list[str],
        output_language: Literal["auto", "zh", "en"],
        target_fields: list[ExtractionFieldSpec],
        export_csv: bool,
    ) -> dict[str, Any]:
        if existing:
            existing["status"] = "running"
            existing["updated_at"] = _now_iso()
            existing["resumed_from_checkpoint"] = True
            existing.setdefault("doc_progress", {})
            return existing
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        progress_path = self.config.reports_dir / f"{course_id}_{timestamp}_progress.md"
        return {
            "checkpoint_key": checkpoint_key,
            "course_id": course_id,
            "doc_ids": doc_ids,
            "output_language": output_language,
            "target_fields": [spec.model_dump() for spec in target_fields],
            "export_csv": export_csv,
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "status": "running",
            "timestamp": timestamp,
            "report_id": "",
            "output_file_name": f"{course_id}_{timestamp}.md",
            "progress_output_path": str(progress_path),
            "analyses": {},
            "extractions": {},
            "doc_progress": {},
            "report": {},
            "csv_output_path": "",
            "last_stage": "start",
            "last_message": "已创建新的分析检查点。",
            "last_doc_id": "",
            "resumed_from_checkpoint": False,
        }

    def _save_compare_checkpoint(self, checkpoint: dict[str, Any]) -> Path:
        checkpoint["updated_at"] = _now_iso()
        path = save_json_mapping(self._checkpoint_path(str(checkpoint.get("checkpoint_key", ""))), checkpoint)
        self._write_compare_progress_snapshot(checkpoint)
        return path

    def _write_compare_progress_snapshot(self, checkpoint: dict[str, Any]) -> None:
        progress_target = str(checkpoint.get("progress_output_path", "")).strip()
        if not progress_target:
            return
        progress_path = Path(progress_target)
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        analyses = checkpoint.get("analyses", {})
        extractions = checkpoint.get("extractions", {})
        doc_ids = checkpoint.get("doc_ids", [])
        doc_progress = checkpoint.get("doc_progress", {})
        lines = [
            "# 分析进度快照",
            "",
            f"- 状态: {checkpoint.get('status', '-')}",
            f"- 知识库: {checkpoint.get('course_id', '-')}",
            f"- 文档总数: {len(doc_ids)}",
            f"- 已完成单文档分析: {len(analyses)}/{len(doc_ids)}",
            f"- 已完成字段抽取: {len(extractions)}/{len(doc_ids)}",
            f"- 最近阶段: {checkpoint.get('last_stage', '-')}",
            f"- 最近信息: {checkpoint.get('last_message', '-')}",
            f"- 更新时间: {checkpoint.get('updated_at', '-')}",
            "",
            "## 当前进行中的文档",
            "",
        ]
        active_entries: list[dict[str, Any]] = []
        if isinstance(doc_progress, dict):
            for doc_id in doc_ids:
                entry = doc_progress.get(doc_id)
                if not isinstance(entry, dict):
                    continue
                if str(entry.get("status", "")) in {"completed", "cached", "resumed"}:
                    continue
                active_entries.append(entry)
        if active_entries:
            for entry in active_entries:
                locator = ""
                if entry.get("current_window") and entry.get("total_windows"):
                    locator = f" | 滑窗 {entry.get('current_window')}/{entry.get('total_windows')}"
                lines.append(
                    f"- {entry.get('title', entry.get('doc_id', '-'))} | 阶段: {entry.get('stage', '-')} | "
                    f"状态: {entry.get('status', '-')} | 信息: {entry.get('detail', '-')}{locator}"
                )
        else:
            lines.append("- 暂无")
        lines.extend(
            [
                "",
                "## 已完成单文档分析",
                "",
            ]
        )
        if analyses:
            for doc_id in doc_ids:
                if doc_id in analyses:
                    lines.append(f"- {analyses[doc_id].get('title', doc_id)}")
        else:
            lines.append("- 暂无")
        lines.extend(["", "## 已完成字段抽取", ""])
        if extractions:
            for doc_id in doc_ids:
                if doc_id in extractions:
                    lines.append(f"- {extractions[doc_id].get('title', doc_id)}")
        else:
            lines.append("- 暂无")
        progress_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    def _raise_if_pause_requested(self, checkpoint: dict[str, Any]) -> None:
        check_task_cancelled()
        try:
            check_task_paused()
        except OperationPausedError:
            checkpoint["status"] = "paused"
            checkpoint["last_message"] = "当前分析已暂停，进度已保存，下次可以继续。"
            self._save_compare_checkpoint(checkpoint)
            raise

    def _mark_checkpoint_failed(self, checkpoint: dict[str, Any], exc: Exception) -> None:
        checkpoint["status"] = "failed"
        checkpoint["last_message"] = str(getattr(exc, "user_message", exc))
        self._save_compare_checkpoint(checkpoint)

    async def _build_report(
        self,
        *,
        course_id: str,
        analyses,
        key_differences: list[str],
        inspirations: list[str],
        citations: list[ChunkCitation],
        output_path: str,
        report_id: str,
        language: Literal["zh", "en"],
        progress_callback: ProgressCallback | None = None,
    ):
        if self.config.has_chat_model_credentials:
            prompt = build_compare_prompt(
                analyses=analyses,
                language=language,
                citations=citations,
                instruction_override=(
                    self.config.compare_report_prompt_zh
                    if language == "zh"
                    else self.config.compare_report_prompt_en
                ),
            )
            content = await _invoke_chat_with_progress(self.config, prompt, language, progress_callback)
            markdown = str(content).strip()
            if markdown:
                markdown = postprocess_report_markdown(markdown, analyses, citations, language)
                markdown = append_report_citation_sections(markdown, citations, language)
                return ComparisonReport(
                    report_id=report_id,
                    course_id=course_id,
                    doc_ids=[item.doc_id for item in analyses],
                    markdown=markdown if markdown.endswith("\n") else markdown + "\n",
                    output_path=output_path,
                )
        return build_comparison_markdown(
            course_id=course_id,
            analyses=analyses,
            key_differences=key_differences,
            inspirations=inspirations,
            citations=citations,
            output_path=output_path,
            report_id=report_id,
            language=language,
        )

    async def _append_extraction_sections(
        self,
        *,
        report: ComparisonReport,
        course_id: str,
        analyses: list[SingleDocAnalysis],
        extraction_results: list[DocumentExtractionResult],
        target_fields: list[ExtractionFieldSpec],
        language: Literal["zh", "en"],
        export_csv: bool,
        timestamp: str,
        progress_callback: ProgressCallback | None = None,
    ) -> ComparisonReport:
        """Append extraction tables, evidence, and optional CSV artifacts to one report."""

        table_markdown = _build_extraction_table_markdown(extraction_results, target_fields, language)
        warnings = _build_extraction_warning_lines(extraction_results, language)
        summary_markdown = await self._build_extraction_summary(
            analyses=analyses,
            extraction_results=extraction_results,
            language=language,
            progress_callback=progress_callback,
        )
        evidence_markdown = _build_extraction_evidence_markdown(extraction_results, language)
        csv_output_path = None
        if export_csv:
            rows = _build_extraction_csv_rows(extraction_results)
            csv_file = self.config.reports_dir / f"{course_id}_{timestamp}_extraction.csv"
            csv_output_path = save_csv_rows(rows, csv_file)
        sections = [report.markdown.rstrip()]
        if summary_markdown.strip():
            sections.extend(["", summary_markdown.strip()])
        sections.extend(["", table_markdown.strip()])
        if warnings:
            warning_title = "## 定向抽取提醒" if language == "zh" else "## Targeted Extraction Alerts"
            sections.extend(["", warning_title, ""])
            sections.extend(f"- {item}" for item in warnings)
        sections.extend(["", evidence_markdown.strip()])
        if csv_output_path:
            csv_title = "## CSV 导出" if language == "zh" else "## CSV Export"
            sections.extend(["", csv_title, "", f"- {csv_output_path}"])
        return report.model_copy(
            update={
                "markdown": "\n".join(sections).strip() + "\n",
                "csv_output_path": csv_output_path,
                "table_headers": _build_extraction_table_headers(target_fields, language),
                "table_rows": _build_extraction_table_rows(extraction_results, target_fields, language),
                "extraction_warnings": warnings,
            }
        )

    async def _build_extraction_summary(
        self,
        *,
        analyses: list[SingleDocAnalysis],
        extraction_results: list[DocumentExtractionResult],
        language: Literal["zh", "en"],
        progress_callback: ProgressCallback | None = None,
    ) -> str:
        if not extraction_results:
            return ""
        if self.config.has_chat_model_credentials:
            try:
                payload = json.dumps(
                    {
                        "analyses": [item.model_dump() for item in analyses],
                        "extractions": [item.model_dump() for item in extraction_results],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                prompt = build_table_summary_prompt(
                    title="定向字段对比" if language == "zh" else "Targeted field comparison",
                    extraction_json=payload,
                    language=language,
                    instruction_override=(
                        self.config.table_summary_prompt_zh
                        if language == "zh"
                        else self.config.table_summary_prompt_en
                    ),
                )
                content = await _invoke_chat_with_progress(self.config, prompt, language, progress_callback)
                markdown = str(content).strip()
                if markdown:
                    return markdown
            except ProviderRequestError:
                pass
        return _build_fallback_extraction_summary(extraction_results, language)

    async def _collect_citations(self, course_id: str, doc_ids: list[str]) -> list[ChunkCitation]:
        citations: list[ChunkCitation] = []
        for index, doc_id in enumerate(doc_ids, start=1):
            chunks = await self.vector_store.get_document_chunks(course_id=course_id, doc_id=doc_id)
            if not chunks:
                continue
            first_chunk = chunks[0]
            metadata = first_chunk.metadata
            citations.append(
                ChunkCitation(
                    citation_id=index,
                    doc_id=doc_id,
                    file_name=str(metadata.get("file_name", "")),
                    page_label=str(metadata.get("page_label")) if metadata.get("page_label") else None,
                    section_label=str(metadata.get("section_label")) if metadata.get("section_label") else None,
                    chunk_id=str(metadata.get("chunk_id", "")),
                    quote=first_chunk.page_content,
                )
            )
        return citations


def _build_extraction_table_markdown(
    extraction_results: list[DocumentExtractionResult],
    field_specs: list[ExtractionFieldSpec],
    language: Literal["zh", "en"],
) -> str:
    title = "## 定向数据对比表" if language == "zh" else "## Targeted Data Comparison Table"
    field_names = [spec.name for spec in field_specs]
    lines = [
        title,
        "",
        '<a id="targeted-extraction-table"></a>',
        "",
        "| 文件名 | " + " | ".join(field_names) + " |",
        "| --- | " + " | ".join(["---"] * len(field_names)) + " |",
    ]
    for result in extraction_results:
        row = [result.title]
        for spec in field_specs:
            field = _find_field_result(result, spec.name)
            row.append(_format_field_markdown_cell(field, result, language))
        lines.append("| " + " | ".join(_escape_markdown_cell(item) for item in row) + " |")
    return "\n".join(lines)


def _build_extraction_table_headers(field_specs: list[ExtractionFieldSpec], language: Literal["zh", "en"]) -> list[str]:
    first_header = "文件名" if language == "zh" else "Document"
    return [first_header] + [spec.name for spec in field_specs]


def _build_extraction_table_rows(
    extraction_results: list[DocumentExtractionResult],
    field_specs: list[ExtractionFieldSpec],
    language: Literal["zh", "en"],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    first_header = "文件名" if language == "zh" else "Document"
    for result in extraction_results:
        row: dict[str, str] = {first_header: result.title}
        for spec in field_specs:
            field = _find_field_result(result, spec.name)
            row[spec.name] = _format_field_cell(field, language)
        rows.append(row)
    return rows


def _build_extraction_evidence_markdown(
    extraction_results: list[DocumentExtractionResult],
    language: Literal["zh", "en"],
) -> str:
    heading = "## 定向数据复查证据" if language == "zh" else "## Extraction Evidence For Review"
    lines = [heading, ""]
    for result in extraction_results:
        lines.append(f"### {result.title}")
        lines.append("")
        for field in result.fields:
            locator = field.page_label or field.section_label or ("未定位" if language == "zh" else "unlocated")
            summary = f"{field.field_name} | {field.status} | {field.normalized_value or field.value or ('未提及' if language == 'zh' else 'Not found')}"
            anchor_id = _field_evidence_anchor(result, field)
            back_label = "返回对比表" if language == "zh" else "Back To Table"
            lines.append(f'<a id="{anchor_id}"></a>')
            lines.append(f"#### {summary}")
            lines.append("")
            lines.append(f"- {'来源文件' if language == 'zh' else 'Source file'}: {field.source_file or result.title}")
            lines.append(f"- {'定位' if language == 'zh' else 'Locator'}: {locator}")
            lines.append(f"- {'说明' if language == 'zh' else 'Notes'}: {field.notes or '-'}")
            if field.evidence_quote:
                lines.append("")
                lines.append("```text")
                lines.append(field.evidence_quote.strip())
                lines.append("```")
            lines.append("")
            lines.append(f"[{back_label}](#targeted-extraction-table)")
            lines.append("")
    return "\n".join(lines)


def _build_extraction_warning_lines(
    extraction_results: list[DocumentExtractionResult],
    language: Literal["zh", "en"],
) -> list[str]:
    warnings: list[str] = []
    for result in extraction_results:
        for field in result.fields:
            if field.status == "not_found":
                warnings.append(
                    f"{result.title} / {field.field_name}: {'未找到对应数据。' if language == 'zh' else 'No matching data was found.'}"
                )
            elif field.status == "conflict":
                warnings.append(
                    f"{result.title} / {field.field_name}: {field.notes or ('发现多个可能冲突的值。' if language == 'zh' else 'Multiple conflicting values were detected.')}"
                )
            elif field.status == "uncertain":
                warnings.append(
                    f"{result.title} / {field.field_name}: {field.notes or ('证据不完整，建议人工复查。' if language == 'zh' else 'The evidence is incomplete and should be reviewed manually.')}"
                )
    return warnings


def _build_extraction_csv_rows(extraction_results: list[DocumentExtractionResult]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for result in extraction_results:
        for field in result.fields:
            rows.append(
                {
                    "field_name": field.field_name,
                    "doc_title": result.title,
                    "value": field.value,
                    "normalized_value": field.normalized_value,
                    "unit": field.unit,
                    "source_unit": field.source_unit,
                    "converted": str(field.converted),
                    "status": field.status,
                    "notes": field.notes,
                    "source_file": field.source_file,
                    "locator": field.page_label or field.section_label or "",
                    "chunk_id": field.chunk_id,
                    "evidence_quote": field.evidence_quote,
                }
            )
    return rows


def _build_fallback_extraction_summary(
    extraction_results: list[DocumentExtractionResult],
    language: Literal["zh", "en"],
) -> str:
    title = "## 定向数据结论" if language == "zh" else "## Targeted Data Findings"
    lines = [title, ""]
    for result in extraction_results:
        found = sum(1 for field in result.fields if field.status == "found")
        uncertain = sum(1 for field in result.fields if field.status == "uncertain")
        conflict = sum(1 for field in result.fields if field.status == "conflict")
        missing = sum(1 for field in result.fields if field.status == "not_found")
        if language == "zh":
            lines.append(
                f"- {result.title}: 已提取 {found} 项，待复查 {uncertain} 项，冲突 {conflict} 项，缺失 {missing} 项。"
            )
        else:
            lines.append(
                f"- {result.title}: extracted={found}, review={uncertain}, conflict={conflict}, missing={missing}."
            )
    return "\n".join(lines)


def _build_key_differences(analyses, language: str) -> list[str]:
    keyword_sets = [set(item.keywords) for item in analyses]
    if len(keyword_sets) < 2:
        return ["仅选择了一篇文档，无法形成差异对比。"] if language == "zh" else ["Only one document was selected, so no differences were computed."]
    shared = set.intersection(*keyword_sets) if keyword_sets else set()
    unique_terms = []
    for item in analyses:
        diff = sorted(set(item.keywords) - shared)
        if diff:
            unique_terms.append(f"{item.title}: {', '.join(diff[:5])}")
    return unique_terms or (["文档间关键词高度重叠。"] if language == "zh" else ["The documents share highly overlapping keywords."])


def _build_inspirations(analyses, language: str) -> list[str]:
    topic_counter = Counter(topic for item in analyses for topic in item.main_topics)
    top_topics = [topic for topic, _ in topic_counter.most_common(5)]
    if language == "zh":
        return [f"课程复习时优先关注这些主题：{', '.join(top_topics)}"] if top_topics else ["可以围绕文档中的核心主题组织课程复习。"]
    return [f"Prioritize these topics during revision: {', '.join(top_topics)}"] if top_topics else ["Use the shared themes to structure course revision."]


def _find_field_result(result: DocumentExtractionResult, field_name: str) -> ExtractedFieldValue:
    for item in result.fields:
        if item.field_name == field_name:
            return item
    return ExtractedFieldValue(field_name=field_name, status="not_found")


def _format_field_cell(field: ExtractedFieldValue, language: str) -> str:
    if field.status == "not_found":
        return "未提及" if language == "zh" else "Not found"
    label = field.normalized_value or field.value or "-"
    if field.status == "conflict":
        return f"{label} (冲突)" if language == "zh" else f"{label} (conflict)"
    if field.status == "uncertain":
        return f"{label} (待复查)" if language == "zh" else f"{label} (review)"
    return label


def _format_field_markdown_cell(
    field: ExtractedFieldValue,
    result: DocumentExtractionResult,
    language: str,
) -> str:
    label = _format_field_cell(field, language)
    anchor_id = _field_evidence_anchor(result, field)
    return f"[{label}](#{anchor_id})"


def _field_evidence_anchor(result: DocumentExtractionResult, field: ExtractedFieldValue) -> str:
    return (
        "field-evidence-"
        f"{sanitize_storage_key(result.title or result.doc_id)}-"
        f"{sanitize_storage_key(field.field_name)}"
    )


def _escape_markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


def _stable_analysis_cache_state(documents: list[SourceDocument]) -> dict[str, Any]:
    ordered_documents = sorted(documents, key=_document_cache_sort_key)
    return {
        "schema": _ANALYSIS_CACHE_SCHEMA,
        "content_signature": build_cache_key(
            [
                {
                    "locator": str(
                        document.metadata.get("page_label")
                        or document.metadata.get("section_label")
                        or document.metadata.get("section")
                        or ""
                    ),
                    "chunk_index": int(document.metadata.get("chunk_index", 0) or 0),
                    "page_content": document.page_content,
                }
                for document in ordered_documents
            ]
        ),
        "total_chunks": len(ordered_documents),
        "total_chars": sum(len(document.page_content) for document in ordered_documents),
    }


def _legacy_document_cache_state(documents: list[SourceDocument]) -> list[dict[str, str]]:
    states: list[dict[str, str]] = []
    for document in documents:
        metadata = document.metadata
        file_path = str(metadata.get("file_path", ""))
        states.append(
            {
                "doc_id": str(metadata.get("doc_id", "")),
                "chunk_id": str(metadata.get("chunk_id", "")),
                "file_signature": build_file_signature(file_path) if file_path else "",
                "length": str(len(document.page_content)),
            }
        )
    return states


def _document_cache_state(documents: list[SourceDocument]) -> list[dict[str, str]]:
    """Compatibility helper kept for extraction and old call sites."""

    return _legacy_document_cache_state(documents)


def _document_cache_sort_key(document: SourceDocument) -> tuple[int, str, int, str, str]:
    metadata = document.metadata
    return (
        int(metadata.get("page", 0) or 0),
        str(metadata.get("section_label") or metadata.get("section") or ""),
        int(metadata.get("chunk_index", 0) or 0),
        str(metadata.get("page_label") or ""),
        str(metadata.get("chunk_id") or ""),
    )


def _read_cached_payload(
    cache: JsonResultCache,
    *,
    primary_namespace: str,
    primary_key: str,
    legacy_namespace: str | None = None,
    legacy_key: str | None = None,
) -> dict[str, Any] | None:
    cached = cache.get(primary_namespace, primary_key)
    if isinstance(cached, dict) and cached:
        return cached
    if not legacy_namespace or not legacy_key:
        return None
    if legacy_namespace == primary_namespace and legacy_key == primary_key:
        return None
    legacy = cache.get(legacy_namespace, legacy_key)
    if isinstance(legacy, dict) and legacy:
        cache.set(primary_namespace, primary_key, legacy)
        return legacy
    return None


def _validate_document_extraction_result(payload: dict[str, Any]) -> DocumentExtractionResult:
    fields = [
        ExtractedFieldValue.model_validate(item)
        for item in payload.get("fields", [])
        if isinstance(item, (dict, ExtractedFieldValue))
    ]
    return DocumentExtractionResult(
        doc_id=str(payload.get("doc_id", "")),
        title=str(payload.get("title", "")),
        fields=fields,
    )


def _match_field_name(candidate: str, field_specs: list[ExtractionFieldSpec]) -> str:
    normalized = candidate.strip().lower()
    for spec in field_specs:
        if normalized == spec.name.strip().lower():
            return spec.name
    for spec in field_specs:
        if normalized and normalized in spec.name.strip().lower():
            return spec.name
    for spec in field_specs:
        aliases = _field_terms(spec)
        if normalized and normalized in aliases:
            return spec.name
    return ""


def _rank_chunks_for_field(
    documents: list[SourceDocument],
    field_spec: ExtractionFieldSpec,
) -> list[tuple[float, SourceDocument]]:
    query_terms = _field_terms(field_spec)
    textual_field = _prefers_textual_extraction(field_spec)
    ranked: list[tuple[float, SourceDocument]] = []
    for document in documents:
        text = _normalize_text(document.page_content)
        metadata_text = _normalize_text(
            " ".join(
                [
                    str(document.metadata.get("file_name", "")),
                    str(document.metadata.get("section_label", "")),
                    str(document.metadata.get("section", "")),
                ]
            )
        )
        score = 0.0
        for term in query_terms:
            if term in text:
                score += 2.0
            if term in metadata_text:
                score += 1.0
        page_label = str(document.metadata.get("page_label", "")).lower()
        if textual_field and page_label == "p.1":
            score += 1.2
        elif textual_field and page_label == "p.2":
            score += 0.4
        if textual_field and _is_preparation_context(document.page_content):
            score -= 3.0
        if textual_field and any(marker in text for marker in ("experimental section", "supporting information", "materials and reagents")):
            score -= 2.0
        if _is_numeric_field(field_spec) and re.search(r"\d", document.page_content):
            score += 0.4
        if _looks_like_measurement_context(document.page_content, field_spec):
            score += 0.6
        if re.search(r"\d", document.page_content):
            score += 0.2
        if score > 0:
            ranked.append((score, document))
    ranked.sort(key=lambda item: item[0], reverse=True)
    if ranked:
        return ranked
    return [(0.0, document) for document in documents[:3]]


def _select_candidate_chunks_for_fields(
    documents: list[SourceDocument],
    field_specs: list[ExtractionFieldSpec],
    *,
    max_chunks: int,
    per_field: int,
) -> list[SourceDocument]:
    ordered: list[SourceDocument] = []
    seen: set[str] = set()
    for spec in field_specs:
        ranked = _rank_chunks_for_field(documents, spec)
        for _, document in ranked[:per_field]:
            chunk_id = str(document.metadata.get("chunk_id", ""))
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            ordered.append(document)
            if len(ordered) >= max_chunks:
                return ordered
    return ordered or documents[:max_chunks]


def _extract_value_from_chunks(
    chunks: list[SourceDocument],
    field_spec: ExtractionFieldSpec,
    language: Literal["zh", "en"],
) -> ExtractedFieldValue:
    found_candidates: list[tuple[float, str, str, SourceDocument, str]] = []
    terms = _field_terms(field_spec)
    for chunk_index, chunk in enumerate(chunks):
        sentences = _split_sentences(chunk.page_content) or [chunk.page_content]
        for sentence in sentences:
            normalized_sentence = _normalize_text(sentence)
            if terms and not any(term in normalized_sentence for term in terms):
                continue
            textual_value = _extract_textual_field_value(sentence, field_spec)
            if textual_value and _prefers_textual_extraction(field_spec):
                if _looks_like_invalid_textual_candidate(textual_value, field_spec, sentence):
                    continue
                found_candidates.append(
                    (
                        _score_local_field_candidate(
                            field_spec=field_spec,
                            value=textual_value,
                            unit=field_spec.expected_unit.strip(),
                            sentence=sentence,
                            chunk_index=chunk_index,
                        ),
                        textual_value,
                        field_spec.expected_unit.strip(),
                        chunk,
                        sentence.strip(),
                    )
                )
                continue
            parsed_matches = _extract_numeric_candidates(sentence, field_spec)
            if parsed_matches:
                if len(parsed_matches) > 1:
                    combined_value = _extract_combined_numeric_candidate(sentence)
                    combined_unit = parsed_matches[0][1]
                    if combined_value:
                        found_candidates.append(
                            (
                                _score_local_field_candidate(
                                    field_spec=field_spec,
                                    value=combined_value,
                                    unit=combined_unit,
                                    sentence=sentence,
                                    chunk_index=chunk_index,
                                )
                                + 0.8,
                                combined_value,
                                combined_unit,
                                chunk,
                                sentence.strip(),
                            )
                        )
                for value, unit in parsed_matches:
                    found_candidates.append(
                        (
                            _score_local_field_candidate(
                                field_spec=field_spec,
                                value=value,
                                unit=unit,
                                sentence=sentence,
                                chunk_index=chunk_index,
                            ),
                            value,
                            unit,
                            chunk,
                            sentence.strip(),
                        )
                    )
            elif not _is_numeric_field(field_spec):
                fallback_value = textual_value or sentence.strip()
                if _looks_like_invalid_textual_candidate(fallback_value, field_spec, sentence):
                    continue
                found_candidates.append(
                    (
                        _score_local_field_candidate(
                            field_spec=field_spec,
                            value=fallback_value,
                            unit="",
                            sentence=sentence,
                            chunk_index=chunk_index,
                        ),
                        fallback_value,
                        "",
                        chunk,
                        sentence.strip(),
                    )
                )
    if not found_candidates:
        return ExtractedFieldValue(
            field_name=field_spec.name,
            status="not_found",
            notes="未找到相关字段。" if language == "zh" else "The requested field was not found.",
        )
    found_candidates.sort(key=lambda item: item[0], reverse=True)
    top_score = found_candidates[0][0]
    unique_values = {
        _normalize_extracted_value(value, unit=_normalize_unit(unit))
        for score, value, unit, _, _ in found_candidates
        if value.strip() and score >= top_score - 0.75
    }
    _, best_value, best_unit, best_chunk, best_sentence = found_candidates[0]
    status: Literal["found", "not_found", "conflict", "uncertain"] = "found"
    notes = ""
    if len({item for item in unique_values if item}) > 1:
        status = "conflict"
        notes = "同一字段出现多个不同取值，请人工复查。" if language == "zh" else "Multiple different values were detected for this field; please review manually."
    elif _is_numeric_field(field_spec) and not re.search(r"\d", best_value) and best_sentence:
        status = "uncertain"
        notes = "提取到的是描述性句子，不是明确数值。" if language == "zh" else "Only a descriptive sentence was found instead of a precise value."
    metadata = best_chunk.metadata
    converted_value, final_unit, conversion_note, converted = _convert_value_to_expected_unit(
        best_value,
        source_unit=best_unit,
        expected_unit=field_spec.expected_unit,
        language=language,
    )
    if conversion_note:
        notes = f"{notes} {conversion_note}".strip()
    return ExtractedFieldValue(
        field_name=field_spec.name,
        value=best_value,
        normalized_value=_normalize_extracted_value(converted_value or best_value, unit=final_unit),
        unit=final_unit,
        source_unit=_normalize_unit(best_unit),
        converted=converted,
        status=status,
        notes=notes,
        source_file=str(metadata.get("file_name", "")),
        page_label=str(metadata.get("page_label")) if metadata.get("page_label") else None,
        section_label=str(metadata.get("section_label")) if metadata.get("section_label") else None,
        chunk_id=str(metadata.get("chunk_id", "")),
        evidence_quote=best_chunk.page_content,
    )


def _extract_combined_numeric_candidate(sentence: str) -> str:
    cleaned = _normalize_measurement_sentence(sentence)
    patterns = [
        re.compile(
            r"([><~]?\s*[-+]?\d+(?:\.\d+)?)\s*((?:u|m)?mol)\s*(?:/?\s*g(?:\s*-?1)?)\s*(?:/?\s*h(?:\s*-?1)?)",
            re.IGNORECASE,
        ),
        re.compile(r"([><~]?\s*[-+]?\d+(?:\.\d+)?)\s*(°\s*C|°C|K)\b", re.IGNORECASE),
    ]
    for pattern in patterns:
        first_match = pattern.search(cleaned)
        if first_match:
            return cleaned[first_match.start() :].strip(" .,:;")
    first_numeric = re.search(r"[><~]?\s*[-+]?\d+(?:\.\d+)?", cleaned)
    if not first_numeric:
        return cleaned.strip()
    return cleaned[first_numeric.start() :].strip(" .,:;")


def _field_terms(field_spec: ExtractionFieldSpec) -> list[str]:
    merged = " ".join(part for part in [field_spec.name, field_spec.instruction, field_spec.expected_unit] if part.strip())
    tokens = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z][A-Za-z0-9_\-]{1,}", merged.lower())
    aliases = _field_alias_terms(field_spec)
    merged_tokens = tokens + aliases
    return list(
        dict.fromkeys(token for token in merged_tokens if token not in {"value", "data", "field", "unit", "extract"})
    )


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _field_alias_terms(field_spec: ExtractionFieldSpec) -> list[str]:
    """Return bilingual aliases so Chinese field names can match English evidence."""

    text = " ".join(
        part.strip().lower()
        for part in [field_spec.name, field_spec.instruction, field_spec.expected_unit]
        if part and part.strip()
    )
    aliases: list[str] = []
    if any(token in text for token in {"反应物", "reactant", "substrate", "feedstock"}):
        aliases.extend(
            [
                "reactant",
                "reactants",
                "substrate",
                "feedstock",
                "plastic",
                "polymer",
                "waste plastic",
                "polyimide waste",
                "polyimide plastic waste",
                "pi waste",
                "polylactic acid",
                "polyethylene terephthalate",
                "polyolefin plastics",
                "pla",
                "pet",
            ]
        )
    if any(token in text for token in {"产物", "product", "products"}):
        aliases.extend(
            [
                "product",
                "products",
                "generated",
                "yield",
                "value-added chemicals",
                "commodity chemicals",
                "organic acids",
                "liquid product",
                "such as",
                "including",
                "produced",
                "generated products",
                "pyruvate",
                "acetate",
                "formic acid",
                "acetic acid",
                "succinic acid",
            ]
        )
    if any(token in text for token in {"氢气产生速率", "产氢速率", "hydrogen", "h2", "evolution", "production rate"}):
        aliases.extend(
            [
                "hydrogen production rate",
                "hydrogen generation rate",
                "hydrogen evolution rate",
                "h2 production rate",
                "h2 evolution rate",
                "h2 generation rate",
                "benchmark efficiency",
            ]
        )
    if any(token in text for token in {"反应温度", "temperature", "温度"}):
        aliases.extend(["temperature", "reaction temperature", "operating temperature", "ambient conditions", "mild conditions"])
    if any(token in text for token in {"反应光强", "光强", "irradiance", "light", "illumination", "am"}):
        aliases.extend(["light intensity", "irradiance", "illumination", "AM 1.5", "1 sun", "solar-driven", "visible light"])
    if any(token in text for token in {"起始电位", "onset", "potential", "voltage", "bias"}):
        aliases.extend(["onset potential", "onset voltage", "potential", "bias", "applied bias"])
    return list(dict.fromkeys(alias.lower() for alias in aliases if alias))


def _is_numeric_field(field_spec: ExtractionFieldSpec) -> bool:
    name = f"{field_spec.name} {field_spec.instruction} {field_spec.expected_unit}".lower()
    if _normalize_unit(field_spec.expected_unit) == "AM":
        return False
    numeric_markers = (
        "速率",
        "效率",
        "温度",
        "电位",
        "电流",
        "光强",
        "rate",
        "temperature",
        "potential",
        "voltage",
        "current",
        "efficiency",
        "%",
    )
    return bool(field_spec.expected_unit.strip()) or any(marker in name for marker in numeric_markers)


def _looks_like_measurement_context(text: str, field_spec: ExtractionFieldSpec) -> bool:
    normalized = _normalize_text(text)
    hints = _field_alias_terms(field_spec)
    if any(hint in normalized for hint in hints[:10]):
        return True
    if _is_numeric_field(field_spec):
        numeric_hints = ("rate", "reached", "production", "evolution", "yield", "temperature", "light", "irradiat")
        return any(hint in normalized for hint in numeric_hints)
    return False


def _looks_like_reaction_context(text: str, field_spec: ExtractionFieldSpec) -> bool:
    normalized = _normalize_text(text)
    shared = (
        "reaction",
        "photoreforming",
        "photo reforming",
        "photoelectrochemical",
        "upcycling",
        "upgrading",
        "reforming",
        "valorize",
        "valorise",
        "irradiated",
        "simulated solar light",
        "visible light",
        "under anoxic conditions",
        "in a typical experiment",
        "reactor",
    )
    if any(token in normalized for token in shared):
        return True
    if "反应温度" in field_spec.name or "temperature" in field_spec.name.lower():
        return any(token in normalized for token in ("room temperature", "at 25", "at 30", "reaction temperature"))
    if _normalize_unit(field_spec.expected_unit) == "AM":
        return any(token in normalized for token in ("am 1.5", "1 sun", "light source", "irradiance"))
    return False


def _is_preparation_context(text: str) -> bool:
    normalized = _normalize_text(text)
    preparation_markers = (
        "synthesized",
        "synthesis",
        "prepared",
        "preparation",
        "annealed",
        "autoclave",
        "washed",
        "dried",
        "calcined",
        "heated",
        "heating",
        "characterization",
        "xrd",
        "xps",
        "tem",
        "sem",
        "electrode",
        "precursor",
        "mixture",
        "furnace",
        "centrifugation",
        "dissolution",
        "dissolved",
        "material characterization",
    )
    return any(marker in normalized for marker in preparation_markers)


def _extract_textual_field_value(sentence: str, field_spec: ExtractionFieldSpec) -> str:
    """Extract a short text span for non-numeric fields such as reactant or product."""

    stripped = " ".join(sentence.split())
    if not stripped:
        return ""
    patterns: list[re.Pattern[str]] = []
    name = field_spec.name.lower()
    if any(token in name for token in ("反应物", "reactant")):
        patterns.extend(
            [
                re.compile(r"reactant[s]?\s*[:：]\s*([^.;,\n]+)", re.IGNORECASE),
                re.compile(r"(?:using|utilizing)\s+([^.;,\n]+?)\s+as\s+substrates?", re.IGNORECASE),
                re.compile(r"(?:substrates?)\s*(?:were|was|:)?\s*([^.;,\n]+)", re.IGNORECASE),
                re.compile(r"(?:upgrade|upgrades|upgrading|upcycle|upcycles|upcycling|valorize|valorizes|valorizing)\s+([^.;,\n]+?)\s+into", re.IGNORECASE),
                re.compile(r"(?:reforming|conversion|photoreforming|upcycling)\s+of\s+([^.;,\n]+?)(?:\s+into|\s+to|\s+for|\s+under|[.;,\n])", re.IGNORECASE),
            ]
        )
    if any(token in name for token in ("产物", "product")):
        patterns.extend(
            [
                re.compile(r"\bproduct[s]?\b\s*[:：]\s*([^.;,\n]+)", re.IGNORECASE),
                re.compile(r"\b(?:value-added chemicals?|products?)\b\s*\((?:e\.g\.|i\.e\.)\s*,?\s*([^)]+)\)", re.IGNORECASE),
                re.compile(r"\b(?:value-added chemicals?|products?)\b\s*(?:such as|including)\s*([^.;\n]+)", re.IGNORECASE),
                re.compile(r"\b(?:commodity|valuable|organic)\s+chemicals?\b\s*(?:\((?:i\.e\.|e\.g\.)\s*,?\s*([^)]+)\)|(?:such as|including)\s*([^.;\n]+))", re.IGNORECASE),
                re.compile(r"\b(?:organic acids?|liquid products?)\b\s*(?:such as|including)\s*([^.;\n]+)", re.IGNORECASE),
                re.compile(r"\b(?:major|main)\s+products?\b.*?\b(?:are|include(?:d)?)\s*([^.;\n]+)", re.IGNORECASE),
                re.compile(r"(?:such as|including)\s+([^.;\n]+)", re.IGNORECASE),
                re.compile(r"yield(?:ed|s)?\s+(?:valuable\s+|value-added\s+)?([^.;,\n]+)", re.IGNORECASE),
            ]
        )
    if any(token in name for token in ("光强", "light", "irradiance", "illumination")):
        patterns.extend(
            [
                re.compile(r"(am\s*1(?:\.\d+)?g?)", re.IGNORECASE),
                re.compile(r"(?:light intensity|irradiance|illumination)\s*[:：]\s*([^.;,\n]+)", re.IGNORECASE),
                re.compile(r"(visible light|solar-driven|1\s*sun)", re.IGNORECASE),
            ]
        )
    for pattern in patterns:
        match = pattern.search(stripped)
        if match:
            candidate_raw = next((group for group in match.groups() if group), match.group(1))
            candidate = _clean_textual_field_candidate(candidate_raw.strip(" .,:;"), field_spec)
            if candidate:
                return candidate
    return stripped[:160]


def _looks_like_invalid_textual_candidate(value: str, field_spec: ExtractionFieldSpec, sentence: str) -> bool:
    cleaned = " ".join(value.strip().split())
    if not cleaned:
        return True
    normalized = _normalize_text(cleaned)
    if len(cleaned) > 140:
        return True
    if normalized == _normalize_text(sentence) and len(cleaned) > 80:
        return True
    if len(cleaned) <= 3:
        return True
    lowered_name = field_spec.name.lower()
    if any(token in lowered_name for token in ("产物", "product", "reactant", "反应物")):
        bad_markers = (
            "mechanism",
            "facilitates",
            "promote",
            "demonstrate",
            "extracts electrons",
            "not only",
            "wherein",
            "charge",
            "carrier",
            "intermediate",
            "intermediates",
            "pathway",
            "pathways",
            "analysis",
            "spectra",
            "figure",
            "supporting information",
            "slow down",
            "mass transport",
            "adjacent to the electrode",
            "water oxidation",
            "photo-generated electrons",
        )
        if any(marker in normalized for marker in bad_markers):
            return True
    if any(token in lowered_name for token in ("反应物", "reactant")):
        if _looks_like_generic_plastic_reference(cleaned):
            return True
        if any(marker in normalized for marker in ("hydrogen", "evolution", "generation", "production", "voltage", "current")):
            return True
    if any(token in lowered_name for token in ("产物", "product")):
        if normalized in {"analysis", "products", "value-added chemicals", "organic acids"}:
            return True
        if "intermediate" in _normalize_text(sentence):
            return True
    return False


def _clean_textual_field_candidate(value: str, field_spec: ExtractionFieldSpec) -> str:
    cleaned = " ".join(value.strip().split())
    if not cleaned:
        return ""
    cleaned = re.sub(r"\([^)]*(?:supporting information|figure|table)[^)]*\)", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\([^)]{0,60}\)", "", cleaned).strip(" ,;:")
    lowered_name = field_spec.name.lower()
    if any(token in lowered_name for token in ("产物", "product")):
        cleaned = re.sub(
            r"^(?:value-added|valuable|commodity|organic)\s+chemicals?\s*(?:such as|including|i\.e\.)?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"^(?:organic acids?|liquid products?)\s*(?:such as|including|i\.e\.)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.split(r"\b(?:were|was|are|is|with|under|via|for|from|to)\b", cleaned, maxsplit=1, flags=re.IGNORECASE)[0].strip(" ,;:")
    if any(token in lowered_name for token in ("反应物", "reactant")):
        cleaned = re.sub(r"^(?:waste\s+)?plastic\s+waste\s+and\s+.*$", "plastic waste", cleaned, flags=re.IGNORECASE)
        cleaned = re.split(r"\b(?:could|would|while|because|which|that)\b", cleaned, maxsplit=1, flags=re.IGNORECASE)[0].strip(" ,;:")
    return cleaned


def _looks_like_generic_plastic_reference(value: str) -> bool:
    normalized = _normalize_text(value)
    if not normalized:
        return False
    generic = {
        "plastic",
        "plastics",
        "plastic waste",
        "plastics waste",
        "waste plastic",
        "waste plastics",
        "wasteplastic",
    }
    if normalized in generic:
        return True
    if "plastic waste" in normalized and not _contains_specific_material_token(normalized):
        return True
    return False


def _contains_specific_material_token(text: str) -> bool:
    normalized = _normalize_text(text)
    specific_tokens = (
        "polyimide",
        "polylactic",
        "polyethylene",
        "terephthalate",
        "polyolefin",
        "pla",
        "pet",
        "pi waste",
        "pi ",
    )
    return any(token in normalized for token in specific_tokens)


def _normalize_measurement_sentence(sentence: str) -> str:
    cleaned = sentence.replace("/C1", "/").replace("C176C", "°C").replace("℃", "°C")
    cleaned = cleaned.replace("μ", "u").replace("µ", "u")
    cleaned = cleaned.replace("−", "-").replace("–", "-").replace("⁻", "-").replace("⁺", "+")
    digit_map = str.maketrans({"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9"})
    cleaned = cleaned.translate(digit_map)
    cleaned = cleaned.replace("\x00", " ")
    return " ".join(cleaned.split())


def _extract_numeric_candidates(sentence: str, field_spec: ExtractionFieldSpec) -> list[tuple[str, str]]:
    cleaned = _normalize_measurement_sentence(sentence)
    target_unit = _normalize_unit(field_spec.expected_unit)
    candidates: list[tuple[str, str]] = []

    if target_unit in {"mol/g/h", "mmol/g/h", "umol/g/h"}:
        for match in re.finditer(
            r"([><~]?\s*[-+]?\d+(?:\.\d+)?)\s*((?:u|m)?mol)\s*(?:/?\s*g(?:\s*-?1)?)\s*(?:/?\s*h(?:\s*-?1)?)",
            cleaned,
            re.IGNORECASE,
        ):
            candidates.append((match.group(1).strip(), _normalize_unit(f"{match.group(2)}/g/h")))

    if _unit_signature(target_unit) == "temperature_k":
        for match in re.finditer(r"([><~]?\s*[-+]?\d+(?:\.\d+)?)\s*(°\s*C|°C|K)\b", cleaned, re.IGNORECASE):
            if _is_preparation_context(sentence) and not _looks_like_reaction_context(sentence, field_spec):
                continue
            candidates.append((match.group(1).strip(), _normalize_unit(match.group(2))))

    if candidates:
        return candidates

    generic_matches = re.findall(r"([><~]?\s*[-+]?\d+(?:\.\d+)?)\s*(%|[A-Za-zuΩ°/²^0-9\-]{1,24})", cleaned)
    for value, unit in generic_matches:
        normalized_unit = _normalize_unit(unit)
        if target_unit and not _can_convert_units(normalized_unit, target_unit):
            continue
        candidates.append((value.strip(), normalized_unit))
    return candidates


def _score_local_field_candidate(
    *,
    field_spec: ExtractionFieldSpec,
    value: str,
    unit: str,
    sentence: str,
    chunk_index: int,
) -> float:
    score = max(0.0, 2.0 - (chunk_index * 0.15))
    normalized_sentence = _normalize_text(sentence)
    lowered_name = field_spec.name.lower()
    if _looks_like_measurement_context(sentence, field_spec):
        score += 1.2
    if _looks_like_reaction_context(sentence, field_spec):
        score += 1.2
    if _is_numeric_field(field_spec) and re.search(r"\d", value):
        score += 1.0
    if any(token in lowered_name for token in ("产物", "product")) and any(
        marker in normalized_sentence for marker in ("value-added chemicals", "commodity chemicals", "organic acids", "products", "such as", "including")
    ):
        score += 1.0
    if any(token in lowered_name for token in ("反应物", "reactant")) and _contains_specific_material_token(value):
        score += 1.6
    if any(token in lowered_name for token in ("产物", "product")):
        if any(marker in normalized_sentence for marker in ("product selectivity", "liquid products", "commodity chemicals", "organic acids")):
            score += 1.0
        if any(marker in normalized_sentence for marker in ("intermediate", "pathway", "mechanism", "spectra", "figure")):
            score -= 1.4
    if any(token in lowered_name for token in ("反应物", "reactant")) and _looks_like_generic_plastic_reference(value):
        score -= 1.0
    target_unit = _normalize_unit(field_spec.expected_unit)
    normalized_unit = _normalize_unit(unit)
    if target_unit and normalized_unit and _can_convert_units(normalized_unit, target_unit):
        score += 1.4
    if _is_preparation_context(sentence) and not _looks_like_reaction_context(sentence, field_spec):
        score -= 1.8
    if len(value.strip()) > 120:
        score -= 0.8
    return score


def _prefers_textual_extraction(field_spec: ExtractionFieldSpec) -> bool:
    normalized_unit = _normalize_unit(field_spec.expected_unit)
    if normalized_unit == "AM":
        return True
    name = field_spec.name.lower()
    return any(token in name for token in ("反应物", "产物", "reactant", "product"))


def _normalize_extraction_status(value: str) -> Literal["found", "not_found", "conflict", "uncertain"]:
    normalized = value.strip().lower()
    if normalized in {"found", "not_found", "conflict", "uncertain"}:
        return normalized
    if normalized in {"missing", "none"}:
        return "not_found"
    if normalized in {"review", "ambiguous"}:
        return "uncertain"
    return "found"


async def _emit_progress(callback: ProgressCallback | None, message: str) -> None:
    """Forward analysis progress messages to an optional notebook callback."""

    if callback is None:
        return
    result = callback(message)
    if asyncio.iscoroutine(result):
        await result


async def _emit_analysis_state(callback: AnalysisStateCallback | None, state: dict[str, Any]) -> None:
    """Forward resumable per-document state updates to an optional checkpoint callback."""

    if callback is None:
        return
    result = callback(state)
    if asyncio.iscoroutine(result):
        await result


def _clean_batch_progress_detail(message: str, *, title: str, doc_id: str) -> str:
    """Trim repeated trailing document titles from nested progress messages."""

    normalized = message.strip()
    suffixes = [
        f": {title}",
        f"：{title}",
        f": {doc_id}",
        f"：{doc_id}",
    ]
    for suffix in suffixes:
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)].rstrip(" ：:")
    return normalized or "正在处理。"


def _truncate_progress_title(title: str, max_length: int = 72) -> str:
    """Shorten long file names in progress logs without losing the tail."""

    normalized = " ".join(str(title).split())
    if len(normalized) <= max_length:
        return normalized
    head = max(18, max_length // 2 - 6)
    tail = max(16, max_length - head - 3)
    return f"{normalized[:head]}...{normalized[-tail:]}"


def _format_batch_doc_progress(
    *,
    stage_label: str,
    index: int,
    total_docs: int,
    title: str,
    stage_status: str,
    detail: str,
    completed_units: int,
    total_units: int,
) -> str:
    """Render one user-facing progress line for a specific document."""

    percent = _format_batch_progress_percent(completed_units, total_units)
    short_title = _truncate_progress_title(title)
    return (
        f"[{stage_label} {index}/{total_docs} | {stage_status} | 总进度 {completed_units}/{total_units}，约 {percent}] "
        f"{short_title}：{detail}"
    )


def _format_batch_global_progress(
    *,
    stage_label: str,
    detail: str,
    completed_units: int,
    total_units: int,
) -> str:
    """Render one user-facing progress line for a batch-wide stage."""

    percent = _format_batch_progress_percent(completed_units, total_units)
    return f"[{stage_label} | 总进度 {completed_units}/{total_units}，约 {percent}] {detail}"


def _format_batch_progress_percent(completed_units: int, total_units: int) -> str:
    """Convert completed units into a readable percentage string."""

    if total_units <= 0:
        return "0%"
    return f"{round((max(0, completed_units) / total_units) * 100)}%"


def _format_stage_status(
    *,
    pending_label: str,
    pending_done: int,
    pending_total: int,
    cached_label: str,
    cached_done: int,
    cached_total: int,
    resumed_label: str,
    resumed_done: int,
    resumed_total: int,
) -> str:
    """Render one compact stage summary for pending, cached, and resumed work."""

    return (
        f"{pending_label} {max(0, pending_done)}/{max(0, pending_total)}，"
        f"{cached_label} {max(0, cached_done)}/{max(0, cached_total)}，"
        f"{resumed_label} {max(0, resumed_done)}/{max(0, resumed_total)}"
    )


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


async def _invoke_chat_with_progress(
    config: AppConfig,
    prompt: str,
    language: Literal["zh", "en"],
    progress_callback: ProgressCallback | None = None,
) -> str:
    """Call ``invoke_chat_text`` while staying compatible with mocked test doubles."""

    try:
        return await invoke_chat_text(config, prompt, language, status_callback=progress_callback)
    except TypeError as exc:
        if "status_callback" not in str(exc):
            raise
        return await invoke_chat_text(config, prompt, language)


def _normalize_unit(unit: str) -> str:
    normalized = unit.strip().replace("μ", "u").replace("µ", "u")
    normalized = normalized.replace("·", "/").replace("−", "-").replace("–", "-").replace(" ", "")
    normalized = normalized.replace("lmol", "umol").replace("μmol", "umol").replace("µmol", "umol")
    normalized = re.sub(r"(?i)([a-z]+)g-?1h-?1$", r"\1/g/h", normalized)
    normalized = re.sub(r"(?i)([a-z]+)g-?1$", r"\1/g", normalized)
    normalized = re.sub(r"(?i)([a-z]+)h-?1$", r"\1/h", normalized)
    normalized = re.sub(r"(?i)^([a-z%Ω]+)cm\^?-?2$", r"\1/cm²", normalized)
    normalized = re.sub(r"(?i)^([a-z%Ω]+)g-?1$", r"\1/g", normalized)
    normalized = re.sub(r"(?i)^([a-z%Ω]+)l-?1$", r"\1/L", normalized)
    normalized = normalized.replace("cm-2", "cm^-2").replace("cm−2", "cm^-2").replace("cm2", "cm^2")
    normalized = normalized.replace("/cm^-2", "/cm²").replace("/cm^2", "/cm²")
    replacements = {
        "a/cm²": "A/cm²",
        "ma/cm^-2": "mA/cm²",
        "ma cm^-2": "mA/cm²",
        "ma/cm²": "mA/cm²",
        "ua/cm²": "uA/cm²",
        "ua/cm^-2": "uA/cm²",
        "na/cm²": "nA/cm²",
        "a/g": "A/g",
        "ma/g": "mA/g",
        "ua/g": "uA/g",
        "mol/g/h": "mol/g/h",
        "mmol/g/h": "mmol/g/h",
        "umol/g/h": "umol/g/h",
        "umolgh": "umol/g/h",
        "mmolgh": "mmol/g/h",
        "molgh": "mol/g/h",
        "g/l": "g/L",
        "mg/l": "mg/L",
        "ug/l": "ug/L",
        "ms/cm": "mS/cm",
        "us/cm": "uS/cm",
        "mv": "mV",
        "v": "V",
        "kv": "kV",
        "uv": "uV",
        "a": "A",
        "ma": "mA",
        "ua": "uA",
        "na": "nA",
        "f/g": "F/g",
        "mf/g": "mF/g",
        "uf/g": "uF/g",
        "s": "s",
        "ms": "ms",
        "us": "us",
        "ns": "ns",
        "min": "min",
        "h": "h",
        "g": "g",
        "mg": "mg",
        "ug": "ug",
        "kg": "kg",
        "c": "°C",
        "°c": "°C",
        "k": "K",
        "am": "AM",
        "ohm": "Ω",
        "kohm": "kΩ",
        "mohm": "MΩ",
        "%": "%",
    }
    key = normalized.lower()
    return replacements.get(key, normalized)


def _can_convert_units(source_unit: str, target_unit: str) -> bool:
    if not source_unit or not target_unit:
        return False
    if source_unit == target_unit:
        return True
    source_signature = _unit_signature(source_unit)
    target_signature = _unit_signature(target_unit)
    return bool(source_signature and source_signature == target_signature)


def _convert_value_to_expected_unit(
    value: str,
    *,
    source_unit: str,
    expected_unit: str,
    language: Literal["zh", "en"] = "zh",
) -> tuple[str, str, str, bool]:
    normalized_source = _normalize_unit(source_unit)
    normalized_target = _normalize_unit(expected_unit)
    if not normalized_target:
        return value, normalized_source, "", False
    if not normalized_source:
        return value, normalized_target if normalized_target else normalized_source, "", False
    if normalized_source == normalized_target:
        return value, normalized_target, "", False
    multi_value = _convert_multiple_numeric_values(value, normalized_source, normalized_target)
    if multi_value is not None:
        note = (
            f"已根据原文单位将多个数值从 {normalized_source} 换算为 {normalized_target}。"
            if language == "zh"
            else f"Converted multiple values from {normalized_source} to {normalized_target} based on the source unit."
        )
        return multi_value, normalized_target, note, True
    numeric_value = _parse_numeric_value(value)
    if numeric_value is None:
        note = (
            f"原文单位为 {normalized_source}，未能自动换算到 {normalized_target}。"
            if language == "zh"
            else f"The source unit is {normalized_source}, but it could not be converted automatically to {normalized_target}."
        )
        return value, normalized_source, note, False
    converted = _convert_numeric_between_units(numeric_value, normalized_source, normalized_target)
    if converted is None:
        note = (
            f"原文单位为 {normalized_source}，当前未支持自动换算到 {normalized_target}。"
            if language == "zh"
            else f"The source unit is {normalized_source}, and conversion to {normalized_target} is not supported yet."
        )
        return value, normalized_source, note, False
    rendered = _format_numeric_value(converted)
    note = (
        f"已根据原文单位将 {value} {normalized_source} 换算为 {rendered} {normalized_target}。"
        if language == "zh"
        else f"Converted from {value} {normalized_source} to {rendered} {normalized_target} based on the source unit."
    )
    return rendered, normalized_target, note, True


def _parse_numeric_value(value: str) -> float | None:
    match = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", value.strip())
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _format_numeric_value(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    if abs(value) >= 1e6 or (0 < abs(value) < 1e-4):
        return f"{value:.4g}"
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _convert_multiple_numeric_values(value: str, source_unit: str, target_unit: str) -> str | None:
    normalized_source = _normalize_unit(source_unit)
    normalized_target = _normalize_unit(target_unit)
    signature = _unit_signature(normalized_source)
    if signature != _unit_signature(normalized_target):
        return None

    cleaned = _normalize_measurement_sentence(value)
    if signature == "amount_per_mass_time":
        pattern = re.compile(
            r"(\s*)([><~]?\s*[-+]?\d+(?:\.\d+)?)\s*((?:u|m)?mol)\s*(?:/?\s*g(?:\s*-?1)?)\s*(?:/?\s*h(?:\s*-?1)?)",
            re.IGNORECASE,
        )
    elif signature == "temperature_k":
        pattern = re.compile(r"(\s*)([><~]?\s*[-+]?\d+(?:\.\d+)?)\s*(°\s*C|°C|K)\b", re.IGNORECASE)
    else:
        return None

    replacement_count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal replacement_count
        converted = _convert_numeric_between_units(float(match.group(2).replace(">", "").replace("<", "").replace("~", "").strip()), normalized_source, normalized_target)
        if converted is None:
            return match.group(0)
        replacement_count += 1
        prefix = re.match(r"[><~]?", match.group(2).strip())
        operator = prefix.group(0) if prefix else ""
        return f"{match.group(1)}{operator}{_format_numeric_value(converted)}"

    rendered = pattern.sub(replace, cleaned)
    return rendered if replacement_count > 1 else None


def _convert_numeric_between_units(value: float, source_unit: str, target_unit: str) -> float | None:
    source_info = _unit_conversion_info(source_unit)
    target_info = _unit_conversion_info(target_unit)
    if not source_info or not target_info:
        return None
    source_signature, source_factor, source_offset = source_info
    target_signature, target_factor, target_offset = target_info
    if source_signature != target_signature:
        return None
    base_value = (value + source_offset) * source_factor if source_signature == "temperature_k" else value * source_factor
    if target_signature == "temperature_k":
        converted = (base_value / target_factor) - target_offset
        return converted
    return base_value / target_factor


def _unit_signature(unit: str) -> str | None:
    info = _unit_conversion_info(unit)
    return info[0] if info else None


def _unit_conversion_info(unit: str) -> tuple[str, float, float] | None:
    normalized = _normalize_unit(unit)
    lookup: dict[str, tuple[str, float, float]] = {
        "%": ("ratio", 0.01, 0.0),
        "ratio": ("ratio", 1.0, 0.0),
        "A": ("current", 1.0, 0.0),
        "mA": ("current", 1e-3, 0.0),
        "uA": ("current", 1e-6, 0.0),
        "nA": ("current", 1e-9, 0.0),
        "A/cm²": ("current_density_cm2", 1.0, 0.0),
        "mA/cm²": ("current_density_cm2", 1e-3, 0.0),
        "uA/cm²": ("current_density_cm2", 1e-6, 0.0),
        "nA/cm²": ("current_density_cm2", 1e-9, 0.0),
        "A/g": ("current_per_mass", 1.0, 0.0),
        "mA/g": ("current_per_mass", 1e-3, 0.0),
        "uA/g": ("current_per_mass", 1e-6, 0.0),
        "mol/g/h": ("amount_per_mass_time", 1.0, 0.0),
        "mmol/g/h": ("amount_per_mass_time", 1e-3, 0.0),
        "umol/g/h": ("amount_per_mass_time", 1e-6, 0.0),
        "V": ("voltage", 1.0, 0.0),
        "mV": ("voltage", 1e-3, 0.0),
        "uV": ("voltage", 1e-6, 0.0),
        "kV": ("voltage", 1e3, 0.0),
        "F/g": ("capacitance_per_mass", 1.0, 0.0),
        "mF/g": ("capacitance_per_mass", 1e-3, 0.0),
        "uF/g": ("capacitance_per_mass", 1e-6, 0.0),
        "S/cm": ("conductivity", 1.0, 0.0),
        "mS/cm": ("conductivity", 1e-3, 0.0),
        "uS/cm": ("conductivity", 1e-6, 0.0),
        "Ω": ("resistance", 1.0, 0.0),
        "kΩ": ("resistance", 1e3, 0.0),
        "MΩ": ("resistance", 1e6, 0.0),
        "s": ("time", 1.0, 0.0),
        "ms": ("time", 1e-3, 0.0),
        "us": ("time", 1e-6, 0.0),
        "ns": ("time", 1e-9, 0.0),
        "min": ("time", 60.0, 0.0),
        "h": ("time", 3600.0, 0.0),
        "g": ("mass", 1.0, 0.0),
        "mg": ("mass", 1e-3, 0.0),
        "ug": ("mass", 1e-6, 0.0),
        "kg": ("mass", 1e3, 0.0),
        "g/L": ("mass_per_volume", 1.0, 0.0),
        "mg/L": ("mass_per_volume", 1e-3, 0.0),
        "ug/L": ("mass_per_volume", 1e-6, 0.0),
        "K": ("temperature_k", 1.0, 0.0),
        "°C": ("temperature_k", 1.0, 273.15),
    }
    return lookup.get(normalized)


def _normalize_extracted_value(value: str, *, unit: str = "") -> str:
    cleaned = " ".join(value.strip().split())
    if not cleaned:
        return ""
    if unit:
        cleaned = _strip_equivalent_trailing_unit(cleaned, unit)
        cleaned = _dedupe_trailing_unit(cleaned, unit)
        if cleaned.endswith(unit):
            return cleaned
        if unit.lower() == "am" and "am" in cleaned.lower():
            return cleaned
        return f"{cleaned} {unit}".strip()
    return cleaned


def _coerce_value_and_unit(value: str, unit: str) -> tuple[str, str]:
    source_unit = _normalize_unit(unit)
    raw_value = _sanitize_field_value(value.strip(), expected_unit=source_unit or unit)
    if source_unit:
        return raw_value, source_unit
    split_value, split_unit = _split_value_and_unit(raw_value)
    normalized_split_unit = _normalize_unit(split_unit)
    if normalized_split_unit:
        return split_value, normalized_split_unit
    return raw_value, source_unit


def _split_value_and_unit(raw_value: str) -> tuple[str, str]:
    match = re.match(r"\s*([-+]?\d+(?:\.\d+)?)\s*(.*)\s*$", raw_value)
    if not match:
        return raw_value.strip(), ""
    return match.group(1).strip(), match.group(2).strip()


def _validate_llm_field_candidate(
    *,
    field_spec: ExtractionFieldSpec,
    model_value: ExtractedFieldValue,
    source_doc: SourceDocument | None,
    documents: list[SourceDocument],
    language: Literal["zh", "en"],
) -> ExtractedFieldValue:
    """Prefer the most evidence-grounded value between LLM output and local heuristics."""

    heuristic_chunks: list[SourceDocument] = []
    if source_doc is not None:
        heuristic_chunks.append(source_doc)
    for _, ranked_doc in _rank_chunks_for_field(documents, field_spec)[:4]:
        if ranked_doc not in heuristic_chunks:
            heuristic_chunks.append(ranked_doc)
    heuristic_value = _extract_value_from_chunks(heuristic_chunks or documents[:3], field_spec, language)
    scored_model = _score_field_candidate(field_spec, model_value, source_doc)
    scored_heuristic = _score_field_candidate(field_spec, heuristic_value, source_doc or _first_source_from_candidate(heuristic_value, documents))
    textual_field = any(token in field_spec.name.lower() for token in ("反应物", "产物", "reactant", "product"))
    if textual_field and heuristic_value.status == "found":
        if (
            model_value.status != "found"
            or _looks_like_invalid_textual_candidate(model_value.value, field_spec, model_value.evidence_quote or model_value.value)
            or (heuristic_value.value.strip() and len(heuristic_value.value.strip()) * 1.4 < max(1, len(model_value.value.strip())))
        ):
            return heuristic_value
    if _candidate_is_invalid(model_value, field_spec) or scored_heuristic > scored_model + 0.5:
        return heuristic_value
    model_value.value = _sanitize_field_value(model_value.value, expected_unit=model_value.unit or field_spec.expected_unit)
    model_value.normalized_value = _normalize_extracted_value(model_value.normalized_value or model_value.value, unit=model_value.unit)
    return model_value


def _first_source_from_candidate(candidate: ExtractedFieldValue, documents: list[SourceDocument]) -> SourceDocument | None:
    chunk_id = str(candidate.chunk_id).strip()
    if not chunk_id:
        return None
    for document in documents:
        if str(document.metadata.get("chunk_id", "")).strip() == chunk_id:
            return document
    return None


def _score_field_candidate(
    field_spec: ExtractionFieldSpec,
    candidate: ExtractedFieldValue,
    source_doc: SourceDocument | None,
) -> float:
    if candidate.status == "not_found":
        return 0.0
    score = 1.0
    if candidate.status == "found":
        score += 1.5
    if candidate.status == "conflict":
        score += 1.0
    evidence_text = source_doc.page_content if source_doc is not None else candidate.evidence_quote
    normalized_evidence = _normalize_text(evidence_text)
    normalized_value = _normalize_text(candidate.value)
    if normalized_value and normalized_value in normalized_evidence:
        score += 2.0
    elif _numeric_value_in_text(candidate.value, evidence_text):
        score += 1.5
    if candidate.unit and _normalize_text(candidate.unit) in normalized_evidence:
        score += 0.8
    if candidate.source_unit and _normalize_text(candidate.source_unit) in normalized_evidence:
        score += 0.4
    if candidate.chunk_id:
        score += 0.4
    if _is_numeric_field(field_spec) and not re.search(r"\d", candidate.value):
        score -= 1.5
    if len(candidate.value.strip()) > 120:
        score -= 0.8
    if candidate.unit and candidate.value.lower().endswith(f"{candidate.unit.lower()} {candidate.unit.lower()}"):
        score -= 1.2
    return score


def _candidate_is_invalid(candidate: ExtractedFieldValue, field_spec: ExtractionFieldSpec) -> bool:
    if candidate.status in {"not_found", "uncertain"} and not candidate.value.strip():
        return True
    if _is_numeric_field(field_spec) and candidate.status == "found" and not re.search(r"\d", candidate.value):
        return True
    if candidate.unit and candidate.value.lower().endswith(f"{candidate.unit.lower()} {candidate.unit.lower()}"):
        return True
    if (
        ("反应温度" in field_spec.name or "temperature" in field_spec.name.lower())
        and candidate.evidence_quote
        and _is_preparation_context(candidate.evidence_quote)
        and not _looks_like_reaction_context(candidate.evidence_quote, field_spec)
    ):
        return True
    if _looks_like_invalid_textual_candidate(candidate.value, field_spec, candidate.evidence_quote or candidate.value):
        return True
    return False


def _numeric_value_in_text(value: str, text: str) -> bool:
    numeric = _parse_numeric_value(value)
    if numeric is None:
        return False
    candidates = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    for candidate in candidates:
        try:
            if abs(float(candidate) - numeric) < 1e-9:
                return True
        except ValueError:
            continue
    return False


def _dedupe_trailing_unit(value: str, unit: str) -> str:
    normalized_value = " ".join(value.strip().split())
    if not normalized_value or not unit:
        return normalized_value
    pattern = re.compile(rf"(?:\s*{re.escape(unit)})+$", re.IGNORECASE)
    if pattern.search(normalized_value):
        normalized_value = pattern.sub("", normalized_value).rstrip()
        return f"{normalized_value} {unit}".strip()
    return normalized_value


def _strip_equivalent_trailing_unit(value: str, unit: str) -> str:
    normalized_value = " ".join(value.strip().split())
    normalized_unit = _normalize_unit(unit)
    if not normalized_value or not normalized_unit or normalized_unit == "AM":
        return normalized_value
    split_value, split_unit = _split_value_and_unit(normalized_value)
    if split_unit and _normalize_unit(split_unit) == normalized_unit:
        return split_value.strip()
    equivalent_suffixes = {
        normalized_unit,
        normalized_unit.replace("u", "μ"),
        normalized_unit.replace("u", "µ"),
        normalized_unit.replace("/g/h", " g-1 h-1"),
        normalized_unit.replace("/g/h", " g−1 h−1"),
    }
    for suffix in sorted((item for item in equivalent_suffixes if item), key=len, reverse=True):
        normalized_value = re.sub(rf"\s*{re.escape(suffix)}\s*$", "", normalized_value, flags=re.IGNORECASE).rstrip()
    return normalized_value


def _sanitize_field_value(value: str, *, expected_unit: str = "") -> str:
    cleaned = " ".join(value.strip().split())
    cleaned = cleaned.replace("μ", "u").replace("µ", "u")
    cleaned = cleaned.replace("lmol", "umol")
    if expected_unit:
        cleaned = _dedupe_trailing_unit(cleaned, _normalize_unit(expected_unit))
    return cleaned
