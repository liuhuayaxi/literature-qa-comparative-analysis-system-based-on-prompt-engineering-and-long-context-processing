"""Core data models and compatibility helpers used across the project."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, get_type_hints

try:
    from pydantic import BaseModel, ConfigDict, Field
except ImportError:  # pragma: no cover - exercised only when pydantic is absent.
    _MISSING = object()

    class _FieldInfo:
        def __init__(self, default: Any = _MISSING, default_factory: Any = None) -> None:
            self.default = default
            self.default_factory = default_factory

    def Field(default: Any = _MISSING, default_factory: Any = None, **_: Any) -> "_FieldInfo":
        """Provide a tiny fallback replacement for ``pydantic.Field``."""

        return _FieldInfo(default=default, default_factory=default_factory)

    class BaseModel:
        """Small subset of the Pydantic API used by this project."""

        model_config: dict[str, Any] = {}

        def __init__(self, **kwargs: Any) -> None:
            hints = get_type_hints(self.__class__)
            for name in hints:
                if name in kwargs:
                    value = kwargs[name]
                else:
                    default = getattr(self.__class__, name, _MISSING)
                    if isinstance(default, _FieldInfo):
                        if default.default_factory is not None:
                            value = default.default_factory()
                        elif default.default is not _MISSING:
                            value = default.default
                        else:
                            raise TypeError(f"Missing required field: {name}")
                    elif default is not _MISSING:
                        value = default
                    else:
                        raise TypeError(f"Missing required field: {name}")
                setattr(self, name, value)

            extra_keys = set(kwargs) - set(hints)
            if extra_keys and self.model_config.get("extra") != "allow":
                extras = ", ".join(sorted(extra_keys))
                raise TypeError(f"Unexpected fields: {extras}")

        def model_dump(self) -> dict[str, Any]:
            """Return the model as a plain Python dictionary."""

            hints = get_type_hints(self.__class__)
            return {name: getattr(self, name) for name in hints}

        def model_dump_json(self, **kwargs: Any) -> str:
            """Return the model serialized as JSON text."""

            return json.dumps(self.model_dump(), ensure_ascii=False, **kwargs)

        def model_copy(self, update: dict[str, Any] | None = None) -> "BaseModel":
            """Return a shallow copy of the model with optional field overrides."""

            payload = self.model_dump()
            if update:
                payload.update(update)
            return self.__class__(**payload)

        @classmethod
        def model_validate(cls, value: Any) -> "BaseModel":
            """Validate a dict or model instance against the current model type."""

            if isinstance(value, cls):
                return value
            if isinstance(value, dict):
                return cls(**value)
            raise TypeError(f"Cannot validate value for {cls.__name__}: {value!r}")

        @classmethod
        def model_validate_json(cls, value: str) -> "BaseModel":
            """Validate a JSON string against the current model type."""

            return cls.model_validate(json.loads(value))

        def __repr__(self) -> str:
            values = ", ".join(f"{key}={value!r}" for key, value in self.model_dump().items())
            return f"{self.__class__.__name__}({values})"

    ConfigDict = dict


@dataclass(slots=True)
class SourceDocument:
    """Minimal document container shared by ingestion, retrieval, and analysis."""

    page_content: str
    metadata: dict[str, Any]


class DocumentRecord(BaseModel):
    """Document-level metadata shown in the notebook UI and tests."""

    doc_id: str
    course_id: str
    source_type: Literal["lecture", "assignment", "paper"]
    file_name: str
    file_path: str
    file_ext: Literal["pdf", "md", "txt", "docx"]
    language: Literal["zh", "en", "mixed"]
    chunk_count: int = 0
    is_vectorized: bool = False

    @property
    def path(self) -> Path:
        return Path(self.file_path)


class ChunkCitation(BaseModel):
    """Citation information for a retrieved chunk that supports an answer or report."""

    citation_id: int
    doc_id: str
    file_name: str
    page_label: str | None = None
    section_label: str | None = None
    chunk_id: str = ""
    quote: str = ""
    file_path: str = ""
    source_type: str = ""
    locator_text: str = ""
    matched_terms: list[str] = Field(default_factory=list)
    retrieval_score: float | None = None
    score_breakdown: dict[str, float] = Field(default_factory=dict)


class ChatTurn(BaseModel):
    """One chat message stored in memory and rendered in the UI."""

    role: Literal["system", "user", "assistant"]
    content: str
    created_at: datetime


class ChatResponse(BaseModel):
    """Final answer payload returned by the RAG service."""

    answer: str
    citations: list[ChunkCitation] = Field(default_factory=list)
    language: Literal["zh", "en"]
    used_memory_mode: Literal["session", "persistent"]
    session_id: str


class ChatSessionSummary(BaseModel):
    """Lightweight session summary rendered in the left session list."""

    session_id: str
    title: str
    last_updated: datetime | None = None
    turn_count: int = 0
    memory_mode: Literal["session", "persistent"] | str = "session"
    last_prompt_token_estimate: int = 0
    last_context_compressed: bool = False
    last_context_doc_count: int = 0
    last_context_strategies: list[str] = Field(default_factory=list)
    last_candidate_doc_count: int = 0
    last_rerank_kept_count: int = 0
    last_rerank_filtered_count: int = 0
    last_low_score_filtered: bool = False


class SingleDocAnalysis(BaseModel):
    """Structured analysis result for one document."""

    doc_id: str
    title: str
    language: Literal["zh", "en", "mixed"]
    summary: str
    sentiment: Literal["positive", "neutral", "negative", "mixed"]
    keywords: list[str] = Field(default_factory=list)
    main_topics: list[str] = Field(default_factory=list)
    risk_points: list[str] = Field(default_factory=list)


class ExtractionFieldSpec(BaseModel):
    """One user-defined field that should be extracted from selected documents."""

    name: str
    instruction: str = ""
    expected_unit: str = ""


class ExtractedFieldValue(BaseModel):
    """Structured extraction result for one field inside one document."""

    field_name: str
    value: str = ""
    normalized_value: str = ""
    unit: str = ""
    source_unit: str = ""
    converted: bool = False
    status: Literal["found", "not_found", "conflict", "uncertain"] = "not_found"
    notes: str = ""
    source_file: str = ""
    page_label: str | None = None
    section_label: str | None = None
    chunk_id: str = ""
    evidence_quote: str = ""


class DocumentExtractionResult(BaseModel):
    """All extracted field results produced for one selected document."""

    doc_id: str
    title: str
    fields: list[ExtractedFieldValue] = Field(default_factory=list)


class ComparisonReport(BaseModel):
    """Rendered multi-document comparison report persisted to Markdown."""

    report_id: str
    course_id: str
    doc_ids: list[str] = Field(default_factory=list)
    markdown: str
    output_path: str
    csv_output_path: str | None = None
    table_headers: list[str] = Field(default_factory=list)
    table_rows: list[dict[str, str]] = Field(default_factory=list)
    extraction_warnings: list[str] = Field(default_factory=list)
