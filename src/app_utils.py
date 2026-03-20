"""Shared helper functions for config files, ids, text guards, task control, caching, and token math."""

from __future__ import annotations

import math
import json
import re
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from typing import Any, Literal

from .errors import InputValidationError, OperationCancelledError, OperationPausedError
from .models import ChunkCitation, ComparisonReport, SingleDocAnalysis

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_SENSITIVE_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}\b"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{20,}\b", re.IGNORECASE),
    re.compile(r"\b(?:api[_-]?key|access[_-]?token|secret|client_secret|password|passwd)\b\s*[:=]\s*['\"]?[^\s'\"]{6,}", re.IGNORECASE),
    re.compile(r"\b\d{17}[0-9Xx]\b"),
    re.compile(r"\b\d{16,19}\b"),
]
_CURRENT_TASK_CONTROL: ContextVar["TaskControl | None"] = ContextVar("current_task_control", default=None)


def update_json_config_file(config_path: str | Path, updates: dict[str, Any]) -> Path:
    """Merge partial updates into a JSON config file and keep unrelated keys untouched."""

    path = Path(config_path)
    payload = load_json_mapping(path)
    payload.update(updates)
    return save_json_mapping(path, payload)


def new_doc_id(prefix: str = "doc") -> str:
    """Return a short random identifier for documents or chunks."""

    from uuid import uuid4

    return f"{prefix}_{uuid4().hex[:12]}"


def new_session_id() -> str:
    """Return a short random identifier for one chat session."""

    from uuid import uuid4

    return f"session_{uuid4().hex[:10]}"


def new_report_id() -> str:
    """Return a timestamped identifier for one analysis report."""

    from uuid import uuid4

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"report_{timestamp}_{uuid4().hex[:6]}"


class TaskControl:
    """Share cancellation state across async tasks and worker threads.

    The notebook UI creates one ``TaskControl`` per long-running operation.
    Lower layers check it before retries, while waiting in queues, and inside
    long document-processing loops so the user can stop work immediately.
    """

    def __init__(self, label: str = "") -> None:
        self.label = label
        self._cancel_event = threading.Event()
        self._pause_event = threading.Event()
        self.created_at = time.time()

    @property
    def cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def cancel(self) -> None:
        self._cancel_event.set()

    @property
    def pause_requested(self) -> bool:
        return self._pause_event.is_set()

    def request_pause(self) -> None:
        self._pause_event.set()

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise OperationCancelledError(
                "当前任务已停止。"
            )

    def raise_if_paused(self) -> None:
        if self.pause_requested:
            raise OperationPausedError(
                "当前分析已暂停，已保存到本地进度文件。下次可点击继续上次分析。"
            )

    def raise_if_interrupted(self) -> None:
        self.raise_if_cancelled()
        self.raise_if_paused()

    async def sleep(self, seconds: float, *, interval: float = 0.25) -> None:
        remaining = max(0.0, float(seconds))
        while remaining > 0:
            self.raise_if_interrupted()
            step = min(interval, remaining)
            await asyncio_sleep(step)
            remaining -= step

    def sleep_blocking(self, seconds: float, *, interval: float = 0.25) -> None:
        remaining = max(0.0, float(seconds))
        while remaining > 0:
            self.raise_if_interrupted()
            step = min(interval, remaining)
            time.sleep(step)
            remaining -= step


def get_current_task_control() -> TaskControl | None:
    return _CURRENT_TASK_CONTROL.get()


@contextmanager
def task_control_context(control: TaskControl | None):
    token = _CURRENT_TASK_CONTROL.set(control)
    try:
        yield control
    finally:
        _CURRENT_TASK_CONTROL.reset(token)


def check_task_cancelled() -> None:
    control = get_current_task_control()
    if control is not None:
        control.raise_if_cancelled()


def check_task_paused() -> None:
    control = get_current_task_control()
    if control is not None:
        control.raise_if_paused()


async def sleep_with_task_control(seconds: float) -> None:
    control = get_current_task_control()
    if control is None:
        await asyncio_sleep(max(0.0, float(seconds)))
        return
    await control.sleep(seconds)


def sleep_with_task_control_blocking(seconds: float) -> None:
    control = get_current_task_control()
    if control is None:
        time.sleep(max(0.0, float(seconds)))
        return
    control.sleep_blocking(seconds)


class JsonResultCache:
    """Very small file-based cache for repeatable expensive results."""

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def get(self, namespace: str, key: str) -> dict[str, Any] | None:
        path = self._path(namespace, key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def set(self, namespace: str, key: str, value: dict[str, Any]) -> Path:
        path = self._path(namespace, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_json_safe(value), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def delete_namespace(self, namespace: str) -> None:
        namespace_dir = self.root_dir / namespace
        if not namespace_dir.exists():
            return
        for path in namespace_dir.rglob("*"):
            if path.is_file():
                path.unlink()
        for path in sorted(namespace_dir.rglob("*"), reverse=True):
            if path.is_dir():
                path.rmdir()
        if namespace_dir.exists():
            namespace_dir.rmdir()

    def delete_key_prefix(self, namespace: str, prefix: str) -> int:
        namespace_dir = self.root_dir / namespace
        if not namespace_dir.exists():
            return 0
        removed = 0
        for path in namespace_dir.glob(f"{prefix}*.json"):
            path.unlink(missing_ok=True)
            removed += 1
        return removed

    def _path(self, namespace: str, key: str) -> Path:
        safe_namespace = sanitize_storage_key(namespace)
        safe_key = sanitize_storage_key(key)
        return self.root_dir / safe_namespace / f"{safe_key}.json"


def build_cache_key(*parts: Any) -> str:
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    return sha1(payload.encode("utf-8")).hexdigest()


def sanitize_storage_key(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return normalized.strip("._") or "default"


def load_json_mapping(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        return {}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def save_json_mapping(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def build_file_signature(path: str | Path) -> str:
    target = Path(path)
    if not target.exists():
        return "missing"
    stat = target.stat()
    return sha1(
        f"{target.resolve(strict=False)}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8")
    ).hexdigest()


def build_course_signature(course_id: str, root_dir: str | Path | None = None) -> str:
    if root_dir is None:
        root_dir = Path(__file__).resolve().parent.parent / "data/raw"
    root = Path(root_dir) / course_id
    if not root.exists():
        return "missing"
    parts: list[str] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        stat = path.stat()
        parts.append(f"{path.relative_to(root)}|{stat.st_size}|{stat.st_mtime_ns}")
    return sha1("\n".join(parts).encode("utf-8")).hexdigest()


async def asyncio_sleep(seconds: float) -> None:
    import asyncio

    await asyncio.sleep(seconds)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump())
    return str(value)


def validate_user_text(
    value: str,
    *,
    language: Literal["zh", "en"] = "zh",
    max_chars: int = 4000,
    enable_sensitive_check: bool = True,
) -> str:
    """Validate user input before it enters retrieval or generation."""

    normalized = value.strip()
    if not normalized:
        raise InputValidationError(_guard_message("empty", language), code="empty_input")
    if len(normalized) > max_chars:
        raise InputValidationError(_guard_message("too_long", language, max_chars=max_chars), code="input_too_long")
    if _contains_illegal_control_chars(normalized):
        raise InputValidationError(_guard_message("illegal_chars", language), code="illegal_characters")
    if enable_sensitive_check and _contains_sensitive_content(normalized):
        raise InputValidationError(_guard_message("sensitive", language), code="sensitive_content")
    return normalized


def sanitize_user_question_for_prompt(
    value: str,
    *,
    language: Literal["zh", "en"] = "zh",
    model_name: str | None = None,
    token_limit: int | None = None,
) -> tuple[str, list[str]]:
    """Reduce prompt noise in user questions without changing normal inputs.

    The function focuses on pathological long-noise patterns observed in stress
    tests (for example, huge repeated character runs or repeated identical
    lines). It returns the transformed text plus applied strategy labels.
    """

    normalized = value.strip()
    if not normalized:
        return "", []
    strategies: list[str] = []

    compact_text, collapsed_repeated_chars = _collapse_repeated_character_runs(normalized, language)
    if collapsed_repeated_chars:
        strategies.append("问题去噪")

    compact_text, collapsed_repeated_lines = _collapse_repeated_lines(compact_text, language)
    if collapsed_repeated_lines and "问题去噪" not in strategies:
        strategies.append("问题去噪")

    if token_limit and token_limit > 0 and estimate_token_count(compact_text, model_name) > token_limit:
        compact_text = compress_text_for_prompt(compact_text, token_limit, model_name)
        strategies.append("问题压缩")

    compact_text = compact_text.strip()
    if not compact_text:
        return normalized, []
    return compact_text, strategies


def detect_language(text: str) -> Literal["zh", "en", "mixed"]:
    """Roughly detect whether text is Chinese, English, or mixed."""

    if not text.strip():
        return "en"
    has_cjk = bool(_CJK_RE.search(text))
    has_ascii_words = bool(re.search(r"[A-Za-z]{2,}", text))
    if has_cjk and has_ascii_words:
        return "mixed"
    if has_cjk:
        return "zh"
    return "en"


def resolve_output_language(
    requested: Literal["auto", "zh", "en"],
    *samples: str,
) -> Literal["zh", "en"]:
    """Resolve the final output language from explicit choice or sample text."""

    if requested in {"zh", "en"}:
        return requested
    joined = "\n".join(sample for sample in samples if sample)
    detected = detect_language(joined)
    return "zh" if detected in {"zh", "mixed"} else "en"


def truncate_quote(text: str, limit: int = 180) -> str:
    """Shrink a long quote so citations stay readable in the final answer."""

    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def format_citation_line(citation: ChunkCitation) -> str:
    """Render one citation line for the answer footer."""

    locator = ""
    if citation.page_label:
        locator = f" {citation.page_label}"
    elif citation.section_label:
        locator = f" {citation.section_label}"
    quote = truncate_quote(citation.quote)
    return f"[{citation.citation_id}] {citation.file_name}{locator} \"{quote}\""


def postprocess_report_markdown(
    markdown: str,
    analyses: list[SingleDocAnalysis],
    citations: list[ChunkCitation],
    language: str,
) -> str:
    """Replace internal ids in report prose and convert bare citation markers into jump links."""

    text = str(markdown or "").strip()
    if not text:
        return text
    parts = re.split(r"(```.*?```)", text, flags=re.DOTALL)
    valid_citation_ids = {int(item.citation_id) for item in citations}
    doc_title_map = {
        str(item.doc_id).strip(): str(item.title).strip()
        for item in analyses
        if str(item.doc_id).strip() and str(item.title).strip()
    }
    processed: list[str] = []
    for part in parts:
        if part.startswith("```") and part.endswith("```"):
            processed.append(part)
            continue
        updated = _replace_doc_ids_in_report_text(part, doc_title_map, language)
        updated = _linkify_report_citation_refs(updated, valid_citation_ids, language)
        processed.append(updated)
    return "".join(processed).strip()


def build_report_citation_sections(citations: list[ChunkCitation], language: str) -> str:
    """Render one clickable citation directory plus detailed evidence anchors."""

    heading = "## 8. 关键结论引用" if language == "zh" else "## 8. Key Citations"
    detail_heading = "### 引用详情" if language == "zh" else "### Citation Details"
    file_label = "来源文件" if language == "zh" else "Source File"
    locator_label = "定位" if language == "zh" else "Locator"
    back_label = "返回引用目录" if language == "zh" else "Back To Citation Index"
    none_text = "- 无" if language == "zh" else "- None"
    lines = [heading, "", '<a id="report-citation-index"></a>']
    if not citations:
        lines.append(none_text)
        return "\n".join(lines)
    for citation in citations:
        lines.append(f"- [引用 {citation.citation_id}](#report-citation-{citation.citation_id}) {format_citation_line(citation)}")
    lines.extend(["", detail_heading, ""])
    for citation in citations:
        locator = citation.page_label or citation.section_label or ("未定位" if language == "zh" else "Unlocated")
        lines.extend(
            [
                f'<a id="report-citation-{citation.citation_id}"></a>',
                f"#### 引用 {citation.citation_id}",
                "",
                f"- {file_label}: {citation.file_name}",
                f"- {locator_label}: {locator}",
                "",
                "```text",
                citation.quote.strip(),
                "```",
                "",
                f"[{back_label}](#report-citation-index)",
                "",
            ]
        )
    return "\n".join(lines).strip()


def append_report_citation_sections(markdown: str, citations: list[ChunkCitation], language: str) -> str:
    """Append clickable citation sections to one generated report."""

    base = markdown.rstrip()
    sections = [base] if base else []
    sections.extend(["", build_report_citation_sections(citations, language)])
    return "\n".join(section for section in sections if section is not None).strip() + "\n"


def append_citations(answer: str, citations: list[ChunkCitation], language: str) -> str:
    """Append a formatted citation block to one final answer."""

    if not citations:
        return answer
    heading = "来源" if language == "zh" else "Sources"
    citation_block = "\n".join(format_citation_line(citation) for citation in citations)
    return f"{answer.strip()}\n\n{heading}:\n{citation_block}"


def build_analysis_markdown(analysis: SingleDocAnalysis, language: str) -> str:
    """Format a single-document analysis object as Markdown."""

    heading = "单文档分析结果" if language == "zh" else "Single Document Analysis"
    sentiment_label = "情感" if language == "zh" else "Sentiment"
    keywords_label = "关键词" if language == "zh" else "Keywords"
    topics_label = "核心主题" if language == "zh" else "Main Topics"
    risks_label = "风险点" if language == "zh" else "Risk Points"
    summary_label = "摘要" if language == "zh" else "Summary"
    return (
        f"# {heading}\n\n"
        f"## {analysis.title}\n\n"
        f"### {summary_label}\n{analysis.summary}\n\n"
        f"### {sentiment_label}\n- {analysis.sentiment}\n\n"
        f"### {keywords_label}\n- " + "\n- ".join(analysis.keywords) + "\n\n"
        f"### {topics_label}\n- " + "\n- ".join(analysis.main_topics) + "\n\n"
        f"### {risks_label}\n- " + "\n- ".join(analysis.risk_points)
    )


def build_comparison_markdown(
    course_id: str,
    analyses: list[SingleDocAnalysis],
    key_differences: list[str],
    inspirations: list[str],
    citations: list[ChunkCitation],
    output_path: str,
    report_id: str,
    language: str,
) -> ComparisonReport:
    """Build the fallback comparison report when no chat model is available."""

    lines = [
        "# 文献对比分析报告" if language == "zh" else "# Literature Comparison Report",
        "",
        "## 1. 数据概览" if language == "zh" else "## 1. Data Overview",
        f"- course_id: {course_id}",
        f"- documents: {len(analyses)}",
        "",
        "## 2. 单文档摘要" if language == "zh" else "## 2. Single Document Summaries",
        "",
    ]
    for item in analyses:
        lines.extend([f"### {item.title}", item.summary, ""])
    lines.extend(["## 3. 主题共性" if language == "zh" else "## 3. Shared Themes", ""])
    common_topics = sorted({topic for item in analyses for topic in item.main_topics})
    lines.extend(f"- {topic}" for topic in common_topics[:10])
    lines.extend(["", "## 4. 关键差异" if language == "zh" else "## 4. Key Differences", ""])
    lines.extend(f"- {item}" for item in key_differences)
    lines.extend(
        [
            "",
            "## 5. 方法/观点对比表" if language == "zh" else "## 5. Methods / Viewpoints Matrix",
            "",
            "| Document | Topics | Keywords | Risks |",
            "| --- | --- | --- | --- |",
        ]
    )
    for item in analyses:
        lines.append(
            "| {title} | {topics} | {keywords} | {risks} |".format(
                title=item.title,
                topics=", ".join(item.main_topics[:3]),
                keywords=", ".join(item.keywords[:5]),
                risks=", ".join(item.risk_points[:3]),
            )
        )
    lines.extend(["", "## 6. 风险点与局限" if language == "zh" else "## 6. Risks And Limitations", ""])
    risk_points = sorted({risk for item in analyses for risk in item.risk_points})
    lines.extend(f"- {risk}" for risk in risk_points[:10])
    lines.extend(["", "## 7. 对课程学习的启发" if language == "zh" else "## 7. Learning Implications", ""])
    lines.extend(f"- {item}" for item in inspirations)
    markdown = postprocess_report_markdown("\n".join(lines), analyses, citations, language)
    markdown = append_report_citation_sections(markdown, citations, language)
    return ComparisonReport(
        report_id=report_id,
        course_id=course_id,
        doc_ids=[item.doc_id for item in analyses],
        markdown=markdown,
        output_path=output_path,
    )


_REPORT_CITATION_REF_RE = re.compile(r"(?<!\!)\[(\s*\d+(?:\s*[,，]\s*\d+)*\s*)\](?!\()")


def _replace_doc_ids_in_report_text(text: str, doc_title_map: dict[str, str], language: str) -> str:
    updated = text
    for doc_id, title in sorted(doc_title_map.items(), key=lambda item: len(item[0]), reverse=True):
        label = f"《{title}》" if language == "zh" else title
        pattern = re.compile(rf"(?<![A-Za-z0-9_-]){re.escape(doc_id)}(?![A-Za-z0-9_-])")
        updated = pattern.sub(label, updated)
    return updated


def _linkify_report_citation_refs(text: str, valid_ids: set[int], language: str) -> str:
    if not valid_ids:
        return text

    def _replace(match: re.Match[str]) -> str:
        raw = match.group(1)
        parts = [item.strip() for item in re.split(r"[,，]", raw) if item.strip()]
        if not parts:
            return match.group(0)
        citation_ids: list[int] = []
        for item in parts:
            try:
                citation_ids.append(int(item))
            except ValueError:
                return match.group(0)
        if any(item not in valid_ids for item in citation_ids):
            return match.group(0)
        separator = "、" if language == "zh" else ", "
        return separator.join(f"[{item}](#report-citation-{item})" for item in citation_ids)

    return _REPORT_CITATION_REF_RE.sub(_replace, text)

def _contains_illegal_control_chars(value: str) -> bool:
    for char in value:
        codepoint = ord(char)
        if codepoint < 32 and char not in "\n\r\t":
            return True
    return False


def _contains_sensitive_content(value: str) -> bool:
    return any(pattern.search(value) for pattern in _SENSITIVE_PATTERNS)


_REPEATED_CHARACTER_RUN_RE = re.compile(r"([^\s])\1{127,}")


def _collapse_repeated_character_runs(value: str, language: Literal["zh", "en"]) -> tuple[str, bool]:
    changed = False

    def _replace(match: re.Match[str]) -> str:
        nonlocal changed
        changed = True
        repeated = match.group(0)
        char = match.group(1)
        keep = min(24, len(repeated))
        omitted = max(0, len(repeated) - keep)
        if omitted <= 0:
            return repeated
        marker = (
            f"[重复 {omitted} 次已省略]"
            if language == "zh"
            else f"[repeated {omitted} times omitted]"
        )
        return f"{char * keep}{marker}"

    compact = _REPEATED_CHARACTER_RUN_RE.sub(_replace, value)
    return compact, changed


def _collapse_repeated_lines(value: str, language: Literal["zh", "en"]) -> tuple[str, bool]:
    lines = value.splitlines()
    if len(lines) < 4:
        return value, False
    changed = False
    output: list[str] = []
    index = 0
    while index < len(lines):
        current = lines[index]
        if not current.strip():
            output.append(current)
            index += 1
            continue
        end = index + 1
        while end < len(lines) and lines[end] == current:
            end += 1
        duplicate_count = end - index
        if duplicate_count > 3:
            changed = True
            output.extend([current, current])
            marker = (
                f"[连续重复 {duplicate_count - 2} 行已折叠]"
                if language == "zh"
                else f"[collapsed {duplicate_count - 2} repeated lines]"
            )
            output.append(marker)
        else:
            output.extend(lines[index:end])
        index = end
    return "\n".join(output), changed


def _guard_message(kind: str, language: Literal["zh", "en"], *, max_chars: int = 0) -> str:
    if language == "zh":
        return {
            "empty": "输入不能为空。",
            "too_long": f"输入过长，请控制在 {max_chars} 个字符以内。",
            "illegal_chars": "输入包含非法控制字符，请删除后重试。",
            "sensitive": "输入疑似包含敏感信息（如密钥、令牌、证件号或银行卡号），已被系统拦截。",
        }[kind]
    return {
        "empty": "The input cannot be empty.",
        "too_long": f"The input is too long. Please keep it within {max_chars} characters.",
        "illegal_chars": "The input contains illegal control characters. Remove them and try again.",
        "sensitive": "The input appears to contain sensitive information such as keys, tokens, IDs, or bank card numbers, so it was blocked.",
    }[kind]


def estimate_token_count(text: str, model_name: str | None = None) -> int:
    """Estimate token usage with `tiktoken`, or fall back to a Chinese/English heuristic."""

    if not text:
        return 0
    encoder = _get_tiktoken_encoder(model_name or "")
    if encoder is not None:
        try:
            return len(encoder.encode(text))
        except Exception:
            pass
    cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    other_chars = max(0, len(text) - cjk_chars)
    newline_bonus = text.count("\n")
    return max(1, cjk_chars + math.ceil(other_chars / 4) + newline_bonus)


def trim_text_to_token_limit(text: str, token_limit: int, model_name: str | None = None) -> str:
    """Keep the beginning of one text within a token budget.

    The function prefers exact token trimming when `tiktoken` is available.
    Otherwise it falls back to binary-searching a character prefix whose
    estimated token count stays under the requested budget.
    """

    if token_limit <= 0 or not text:
        return ""
    if estimate_token_count(text, model_name) <= token_limit:
        return text
    encoder = _get_tiktoken_encoder(model_name or "")
    if encoder is not None:
        try:
            encoded = encoder.encode(text)
            if len(encoded) <= token_limit:
                return text
            trimmed = encoder.decode(encoded[:token_limit])
            return trimmed.rstrip()
        except Exception:
            pass

    low = 0
    high = len(text)
    best = ""
    while low <= high:
        middle = (low + high) // 2
        candidate = text[:middle]
        if estimate_token_count(candidate, model_name) <= token_limit:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    return best.rstrip()


def trim_text_tail_to_token_limit(text: str, token_limit: int, model_name: str | None = None) -> str:
    """Keep the tail of one text within a token budget.

    This is mainly used together with `trim_text_to_token_limit()` when the
    caller wants to preserve both the beginning and the end of a long message.
    """

    if token_limit <= 0 or not text:
        return ""
    if estimate_token_count(text, model_name) <= token_limit:
        return text
    encoder = _get_tiktoken_encoder(model_name or "")
    if encoder is not None:
        try:
            encoded = encoder.encode(text)
            if len(encoded) <= token_limit:
                return text
            trimmed = encoder.decode(encoded[-token_limit:])
            return trimmed.lstrip()
        except Exception:
            pass
    low = 0
    high = len(text)
    best = ""
    while low <= high:
        middle = (low + high) // 2
        candidate = text[len(text) - middle :]
        if estimate_token_count(candidate, model_name) <= token_limit:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    return best.lstrip()


def compress_text_for_prompt(text: str, token_limit: int, model_name: str | None = None) -> str:
    """Compress long text for prompts by preserving both the head and the tail.

    The strategy is:
    1. Keep the full text when it already fits.
    2. For very small budgets, fall back to a plain prefix trim.
    3. Otherwise keep a head segment, a tail segment, and join them with
       a visible `...` marker so the model still sees opening context and
       recent details at the same time.
    """

    normalized = text.strip()
    if token_limit <= 0 or not normalized:
        return ""
    if estimate_token_count(normalized, model_name) <= token_limit:
        return normalized
    if token_limit <= 24:
        return trim_text_to_token_limit(normalized, token_limit, model_name)
    marker = "\n...\n"
    marker_tokens = estimate_token_count(marker, model_name)
    available = max(8, token_limit - marker_tokens)
    head_limit = max(4, int(available * 0.45))
    tail_limit = max(4, available - head_limit)
    head = trim_text_to_token_limit(normalized, head_limit, model_name)
    tail = trim_text_tail_to_token_limit(normalized, tail_limit, model_name)
    if not head or not tail:
        return trim_text_to_token_limit(normalized, token_limit, model_name)
    combined = f"{head}{marker}{tail}"
    if estimate_token_count(combined, model_name) <= token_limit:
        return combined
    return trim_text_to_token_limit(combined, token_limit, model_name)


def split_text_into_token_windows(
    text: str,
    window_tokens: int,
    overlap_tokens: int,
    model_name: str | None = None,
) -> list[str]:
    """Split one long text into overlapping windows measured in tokens.

    This helper is used by the long-document analysis pipeline. When exact
    tokenization is available it uses token windows directly; otherwise it
    approximates window sizes with a character-based fallback.
    """

    normalized = text.strip()
    if not normalized:
        return []
    if window_tokens <= 0:
        return [normalized]
    if estimate_token_count(normalized, model_name) <= window_tokens:
        return [normalized]
    encoder = _get_tiktoken_encoder(model_name or "")
    if encoder is not None:
        try:
            encoded = encoder.encode(normalized)
            if len(encoded) <= window_tokens:
                return [normalized]
            windows: list[str] = []
            start = 0
            step = max(1, window_tokens - max(0, overlap_tokens))
            while start < len(encoded):
                end = min(len(encoded), start + window_tokens)
                chunk = encoder.decode(encoded[start:end]).strip()
                if chunk:
                    windows.append(chunk)
                if end >= len(encoded):
                    break
                start += step
            return windows or [normalized]
        except Exception:
            pass
    approx_chars_per_token = _approx_chars_per_token(normalized)
    window_chars = max(64, window_tokens * approx_chars_per_token)
    overlap_chars = max(0, overlap_tokens * approx_chars_per_token)
    step_chars = max(1, window_chars - overlap_chars)
    windows: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(len(normalized), start + window_chars)
        chunk = normalized[start:end].strip()
        if chunk:
            windows.append(chunk)
        if end >= len(normalized):
            break
        start += step_chars
    return windows or [normalized]


def _approx_chars_per_token(text: str) -> int:
    cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    if cjk_chars >= max(1, len(text) // 3):
        return 2
    return 4


def _get_tiktoken_encoder(model_name: str):
    try:
        import tiktoken
    except ImportError:
        return None
    try:
        if model_name:
            return tiktoken.encoding_for_model(model_name)
    except Exception:
        pass
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None
