from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any, Literal

from .app_utils import load_json_mapping, save_json_mapping
from .models import BaseModel, Field
from .llm_tools import (
    DEFAULT_COMPARE_REPORT_PROMPT_EN,
    DEFAULT_COMPARE_REPORT_PROMPT_ZH,
    DEFAULT_DATA_EXTRACTION_PROMPT_EN,
    DEFAULT_DATA_EXTRACTION_PROMPT_ZH,
    DEFAULT_QA_ANSWER_INSTRUCTION_EN,
    DEFAULT_QA_ANSWER_INSTRUCTION_ZH,
    DEFAULT_QA_SYSTEM_PROMPT_EN,
    DEFAULT_QA_SYSTEM_PROMPT_ZH,
    DEFAULT_QUERY_REWRITE_INSTRUCTION_EN,
    DEFAULT_QUERY_REWRITE_INSTRUCTION_ZH,
    DEFAULT_SINGLE_ANALYSIS_PROMPT_EN,
    DEFAULT_SINGLE_ANALYSIS_PROMPT_ZH,
    DEFAULT_TABLE_SUMMARY_PROMPT_EN,
    DEFAULT_TABLE_SUMMARY_PROMPT_ZH,
)


class AppConfig(BaseModel):
    """Central application settings loaded from ``config/app_config.json``."""

    openai_api_key: str = ""
    openai_base_url: str = ""
    chat_api_key: str = ""
    chat_base_url: str = ""
    chat_model: str = ""
    embedding_api_key: str = ""
    embedding_base_url: str = ""
    embedding_model: str = ""
    default_language: Literal["auto", "zh", "en"] = "auto"
    default_memory_mode: Literal["session", "persistent"] = "persistent"
    chat_timeout_seconds: int = 45
    chat_stream_request_timeout_seconds: int = 300
    chat_stream_first_token_timeout_seconds: int = 120
    chat_stream_idle_timeout_seconds: int = 120
    embedding_timeout_seconds: int = 45
    api_retry_attempts: int = 3
    api_retry_backoff_min_seconds: int = 1
    api_retry_backoff_max_seconds: int = 8
    rate_limit_retry_forever: bool = True
    rate_limit_retry_attempts: int = 5
    rate_limit_retry_delay_seconds: int = 1
    chat_provider_concurrency: int = 1
    embedding_provider_concurrency: int = 1
    allow_local_vector_fallback: bool = False
    vector_upsert_batch_size: int = 8
    max_input_chars: int = 4000
    enable_sensitive_input_check: bool = True
    enable_result_cache: bool = True
    model_context_window: int = 32000
    answer_token_reserve: int = 6000
    long_context_window_tokens: int = 2400
    long_context_window_overlap_tokens: int = 240
    recursive_summary_target_tokens: int = 1400
    recursive_summary_batch_size: int = 4
    prompt_compression_turn_token_limit: int = 180
    recent_history_turns: int = 6
    history_summary_token_limit: int = 800
    context_overflow_retries: int = 2
    retrieval_top_k: int = 6
    retrieval_fetch_k: int = 20
    citation_limit: int = 4
    enable_rerank: bool = True
    rerank_min_score: float = 0.28
    rerank_min_keep: int = 1
    rerank_weight_vector: float = 0.45
    rerank_weight_keyword: float = 0.25
    rerank_weight_phrase: float = 0.20
    rerank_weight_metadata: float = 0.10
    chunk_size: int = 800
    chunk_overlap: int = 120
    merge_small_chunks: bool = True
    min_chunk_size: int = 400
    batch_concurrency: int = 5
    enable_query_rewrite: bool = True
    enable_migration_ui: bool = False
    default_streaming_mode: Literal["stream", "non_stream"] = "stream"
    qa_system_prompt_en: str = DEFAULT_QA_SYSTEM_PROMPT_EN
    qa_system_prompt_zh: str = DEFAULT_QA_SYSTEM_PROMPT_ZH
    query_rewrite_instruction_en: str = DEFAULT_QUERY_REWRITE_INSTRUCTION_EN
    query_rewrite_instruction_zh: str = DEFAULT_QUERY_REWRITE_INSTRUCTION_ZH
    qa_answer_instruction_en: str = DEFAULT_QA_ANSWER_INSTRUCTION_EN
    qa_answer_instruction_zh: str = DEFAULT_QA_ANSWER_INSTRUCTION_ZH
    single_analysis_prompt_en: str = DEFAULT_SINGLE_ANALYSIS_PROMPT_EN
    single_analysis_prompt_zh: str = DEFAULT_SINGLE_ANALYSIS_PROMPT_ZH
    compare_report_prompt_en: str = DEFAULT_COMPARE_REPORT_PROMPT_EN
    compare_report_prompt_zh: str = DEFAULT_COMPARE_REPORT_PROMPT_ZH
    data_extraction_prompt_en: str = DEFAULT_DATA_EXTRACTION_PROMPT_EN
    data_extraction_prompt_zh: str = DEFAULT_DATA_EXTRACTION_PROMPT_ZH
    table_summary_prompt_en: str = DEFAULT_TABLE_SUMMARY_PROMPT_EN
    table_summary_prompt_zh: str = DEFAULT_TABLE_SUMMARY_PROMPT_ZH
    db_path: Path = Path("storage/app_state.db")
    vector_dir: Path = Path("storage/chroma")
    api_log_path: Path = Path("logs/runtime_api_traffic.jsonl")
    runtime_api_log_path: Path = Path("logs/runtime_api_traffic.jsonl")
    test_api_log_path: Path = Path("logs/test_api_traffic.jsonl")
    vector_operation_log_path: Path = Path("logs/runtime_vector_operations.jsonl")
    cache_dir: Path = Path("storage/cache")
    analysis_checkpoint_dir: Path = Path("storage/analysis_checkpoints")
    field_template_path: Path = Path("storage/field_templates.json")
    knowledge_base_state_path: Path = Path("storage/knowledge_base_state.json")
    config_path: Path = Path("config/app_config.json")
    project_root: Path = Path(".")
    data_root: Path = Path("data/raw")
    reports_dir: Path = Path("reports")
    migration_messages: list[str] = Field(default_factory=list)
    storage_prepared: bool = False

    @classmethod
    def from_file(
        cls,
        config_path: str | os.PathLike[str] | None = "config/app_config.json",
    ) -> "AppConfig":
        """Load runtime settings from the JSON config file under ``config/``."""

        resolved_config_path = _resolve_config_path(config_path)
        project_root = _project_root_from_config_path(resolved_config_path)
        settings = _load_or_initialize_config(resolved_config_path, project_root)
        legacy_api_key = _read_text_setting(settings, "OPENAI_API_KEY", "")
        legacy_base_url = _normalize_base_url(_read_text_setting(settings, "OPENAI_BASE_URL", ""))
        chat_api_key = _read_with_fallback(settings, "OPENAI_CHAT_API_KEY", legacy_api_key)
        chat_base_url = _normalize_base_url(_read_with_fallback(settings, "OPENAI_CHAT_BASE_URL", legacy_base_url))
        embedding_api_key = _read_with_fallback(settings, "OPENAI_EMBEDDING_API_KEY", legacy_api_key)
        embedding_base_url = _normalize_base_url(
            _read_with_fallback(settings, "OPENAI_EMBEDDING_BASE_URL", legacy_base_url)
        )
        chunk_size = int(_read_setting(settings, "APP_CHUNK_SIZE", 800))
        return cls(
            openai_api_key=legacy_api_key,
            openai_base_url=legacy_base_url,
            chat_api_key=chat_api_key,
            chat_base_url=chat_base_url,
            chat_model=_read_text_setting(settings, "OPENAI_CHAT_MODEL", ""),
            embedding_api_key=embedding_api_key,
            embedding_base_url=embedding_base_url,
            embedding_model=_read_text_setting(settings, "OPENAI_EMBEDDING_MODEL", ""),
            default_language=_read_text_setting(settings, "APP_DEFAULT_LANGUAGE", "auto"),
            default_memory_mode=_read_text_setting(settings, "APP_DEFAULT_MEMORY_MODE", "persistent"),
            chat_timeout_seconds=int(_read_setting(settings, "APP_CHAT_TIMEOUT_SECONDS", 45)),
            chat_stream_request_timeout_seconds=int(
                _read_setting(settings, "APP_CHAT_STREAM_REQUEST_TIMEOUT_SECONDS", 300)
            ),
            chat_stream_first_token_timeout_seconds=int(
                _read_setting(settings, "APP_CHAT_STREAM_FIRST_TOKEN_TIMEOUT_SECONDS", 120)
            ),
            chat_stream_idle_timeout_seconds=int(
                _read_setting(settings, "APP_CHAT_STREAM_IDLE_TIMEOUT_SECONDS", 120)
            ),
            embedding_timeout_seconds=int(_read_setting(settings, "APP_EMBEDDING_TIMEOUT_SECONDS", 45)),
            api_retry_attempts=int(_read_setting(settings, "APP_API_RETRY_ATTEMPTS", 3)),
            api_retry_backoff_min_seconds=int(_read_setting(settings, "APP_API_RETRY_BACKOFF_MIN_SECONDS", 1)),
            api_retry_backoff_max_seconds=int(_read_setting(settings, "APP_API_RETRY_BACKOFF_MAX_SECONDS", 8)),
            rate_limit_retry_forever=_read_bool_setting(settings, "APP_RATE_LIMIT_RETRY_FOREVER", True),
            rate_limit_retry_attempts=int(_read_setting(settings, "APP_RATE_LIMIT_RETRY_ATTEMPTS", 5)),
            rate_limit_retry_delay_seconds=int(_read_setting(settings, "APP_RATE_LIMIT_RETRY_DELAY_SECONDS", 1)),
            chat_provider_concurrency=int(_read_setting(settings, "APP_CHAT_PROVIDER_CONCURRENCY", 1)),
            embedding_provider_concurrency=int(_read_setting(settings, "APP_EMBEDDING_PROVIDER_CONCURRENCY", 1)),
            allow_local_vector_fallback=_read_bool_setting(settings, "APP_ALLOW_LOCAL_VECTOR_FALLBACK", False),
            vector_upsert_batch_size=int(_read_setting(settings, "APP_VECTOR_UPSERT_BATCH_SIZE", 8)),
            max_input_chars=int(_read_setting(settings, "APP_MAX_INPUT_CHARS", 4000)),
            enable_sensitive_input_check=_read_bool_setting(settings, "APP_ENABLE_SENSITIVE_INPUT_CHECK", True),
            enable_result_cache=_read_bool_setting(settings, "APP_ENABLE_RESULT_CACHE", True),
            model_context_window=int(_read_setting(settings, "APP_MODEL_CONTEXT_WINDOW", 32000)),
            answer_token_reserve=int(_read_setting(settings, "APP_ANSWER_TOKEN_RESERVE", 6000)),
            long_context_window_tokens=int(_read_setting(settings, "APP_LONG_CONTEXT_WINDOW_TOKENS", 2400)),
            long_context_window_overlap_tokens=int(
                _read_setting(settings, "APP_LONG_CONTEXT_WINDOW_OVERLAP_TOKENS", 240)
            ),
            recursive_summary_target_tokens=int(
                _read_setting(settings, "APP_RECURSIVE_SUMMARY_TARGET_TOKENS", 1400)
            ),
            recursive_summary_batch_size=int(_read_setting(settings, "APP_RECURSIVE_SUMMARY_BATCH_SIZE", 4)),
            prompt_compression_turn_token_limit=int(
                _read_setting(settings, "APP_PROMPT_COMPRESSION_TURN_TOKEN_LIMIT", 180)
            ),
            recent_history_turns=int(_read_setting(settings, "APP_RECENT_HISTORY_TURNS", 6)),
            history_summary_token_limit=int(_read_setting(settings, "APP_HISTORY_SUMMARY_TOKEN_LIMIT", 800)),
            context_overflow_retries=int(_read_setting(settings, "APP_CONTEXT_OVERFLOW_RETRIES", 2)),
            retrieval_top_k=int(_read_setting(settings, "APP_RETRIEVAL_TOP_K", 6)),
            retrieval_fetch_k=int(_read_setting(settings, "APP_RETRIEVAL_FETCH_K", 20)),
            citation_limit=int(_read_setting(settings, "APP_CITATION_LIMIT", 4)),
            enable_rerank=_read_bool_setting(settings, "APP_ENABLE_RERANK", True),
            rerank_min_score=float(_read_setting(settings, "APP_RERANK_MIN_SCORE", 0.28)),
            rerank_min_keep=int(_read_setting(settings, "APP_RERANK_MIN_KEEP", 1)),
            rerank_weight_vector=float(_read_setting(settings, "APP_RERANK_WEIGHT_VECTOR", 0.45)),
            rerank_weight_keyword=float(_read_setting(settings, "APP_RERANK_WEIGHT_KEYWORD", 0.25)),
            rerank_weight_phrase=float(_read_setting(settings, "APP_RERANK_WEIGHT_PHRASE", 0.20)),
            rerank_weight_metadata=float(_read_setting(settings, "APP_RERANK_WEIGHT_METADATA", 0.10)),
            chunk_size=chunk_size,
            chunk_overlap=int(_read_setting(settings, "APP_CHUNK_OVERLAP", 120)),
            merge_small_chunks=_read_bool_setting(settings, "APP_MERGE_SMALL_CHUNKS", True),
            min_chunk_size=int(_read_setting(settings, "APP_MIN_CHUNK_SIZE", max(200, chunk_size // 2))),
            batch_concurrency=int(_read_setting(settings, "APP_BATCH_CONCURRENCY", 5)),
            enable_query_rewrite=_read_bool_setting(settings, "APP_ENABLE_QUERY_REWRITE", True),
            enable_migration_ui=_read_bool_setting(settings, "APP_ENABLE_MIGRATION_UI", False),
            default_streaming_mode=_read_text_setting(settings, "APP_DEFAULT_STREAMING_MODE", "stream"),
            qa_system_prompt_en=_read_text_setting(settings, "APP_QA_SYSTEM_PROMPT_EN", DEFAULT_QA_SYSTEM_PROMPT_EN),
            qa_system_prompt_zh=_read_text_setting(settings, "APP_QA_SYSTEM_PROMPT_ZH", DEFAULT_QA_SYSTEM_PROMPT_ZH),
            query_rewrite_instruction_en=_read_text_setting(
                settings,
                "APP_QUERY_REWRITE_INSTRUCTION_EN",
                DEFAULT_QUERY_REWRITE_INSTRUCTION_EN,
            ),
            query_rewrite_instruction_zh=_read_text_setting(
                settings,
                "APP_QUERY_REWRITE_INSTRUCTION_ZH",
                DEFAULT_QUERY_REWRITE_INSTRUCTION_ZH,
            ),
            qa_answer_instruction_en=_read_text_setting(
                settings,
                "APP_QA_ANSWER_INSTRUCTION_EN",
                DEFAULT_QA_ANSWER_INSTRUCTION_EN,
            ),
            qa_answer_instruction_zh=_read_text_setting(
                settings,
                "APP_QA_ANSWER_INSTRUCTION_ZH",
                DEFAULT_QA_ANSWER_INSTRUCTION_ZH,
            ),
            single_analysis_prompt_en=_read_text_setting(
                settings,
                "APP_SINGLE_ANALYSIS_PROMPT_EN",
                DEFAULT_SINGLE_ANALYSIS_PROMPT_EN,
            ),
            single_analysis_prompt_zh=_read_text_setting(
                settings,
                "APP_SINGLE_ANALYSIS_PROMPT_ZH",
                DEFAULT_SINGLE_ANALYSIS_PROMPT_ZH,
            ),
            compare_report_prompt_en=_read_text_setting(
                settings,
                "APP_COMPARE_REPORT_PROMPT_EN",
                DEFAULT_COMPARE_REPORT_PROMPT_EN,
            ),
            compare_report_prompt_zh=_read_text_setting(
                settings,
                "APP_COMPARE_REPORT_PROMPT_ZH",
                DEFAULT_COMPARE_REPORT_PROMPT_ZH,
            ),
            data_extraction_prompt_en=_read_text_setting(
                settings,
                "APP_DATA_EXTRACTION_PROMPT_EN",
                DEFAULT_DATA_EXTRACTION_PROMPT_EN,
            ),
            data_extraction_prompt_zh=_read_text_setting(
                settings,
                "APP_DATA_EXTRACTION_PROMPT_ZH",
                DEFAULT_DATA_EXTRACTION_PROMPT_ZH,
            ),
            table_summary_prompt_en=_read_text_setting(
                settings,
                "APP_TABLE_SUMMARY_PROMPT_EN",
                DEFAULT_TABLE_SUMMARY_PROMPT_EN,
            ),
            table_summary_prompt_zh=_read_text_setting(
                settings,
                "APP_TABLE_SUMMARY_PROMPT_ZH",
                DEFAULT_TABLE_SUMMARY_PROMPT_ZH,
            ),
            db_path=_resolve_path_from_root(project_root, _read_setting(settings, "APP_DB_PATH", "storage/app_state.db")),
            vector_dir=_resolve_path_from_root(project_root, _read_setting(settings, "APP_VECTOR_DIR", "storage/chroma")),
            api_log_path=_resolve_path_from_root(
                project_root,
                _read_setting(
                    settings,
                    "APP_RUNTIME_API_LOG_PATH",
                    _read_setting(settings, "APP_API_LOG_PATH", "logs/runtime_api_traffic.jsonl"),
                ),
            ),
            runtime_api_log_path=_resolve_path_from_root(
                project_root,
                _read_setting(settings, "APP_RUNTIME_API_LOG_PATH", "logs/runtime_api_traffic.jsonl"),
            ),
            test_api_log_path=_resolve_path_from_root(
                project_root,
                _read_setting(settings, "APP_TEST_API_LOG_PATH", "logs/test_api_traffic.jsonl"),
            ),
            vector_operation_log_path=_resolve_path_from_root(
                project_root,
                _read_setting(settings, "APP_VECTOR_OPERATION_LOG_PATH", "logs/runtime_vector_operations.jsonl"),
            ),
            cache_dir=_resolve_path_from_root(project_root, _read_setting(settings, "APP_CACHE_DIR", "storage/cache")),
            analysis_checkpoint_dir=_resolve_path_from_root(
                project_root,
                _read_setting(settings, "APP_ANALYSIS_CHECKPOINT_DIR", "storage/analysis_checkpoints"),
            ),
            field_template_path=_resolve_path_from_root(
                project_root,
                _read_setting(settings, "APP_FIELD_TEMPLATE_PATH", "storage/field_templates.json"),
            ),
            knowledge_base_state_path=_resolve_path_from_root(
                project_root,
                _read_setting(settings, "APP_KNOWLEDGE_BASE_STATE_PATH", "storage/knowledge_base_state.json"),
            ),
            config_path=resolved_config_path,
            project_root=project_root,
            data_root=_resolve_path_from_root(project_root, _read_setting(settings, "APP_DATA_ROOT", "data/raw")),
            reports_dir=_resolve_path_from_root(project_root, _read_setting(settings, "APP_REPORTS_DIR", "reports")),
        )

    def to_flat_config(self) -> dict[str, Any]:
        """Serialize runtime settings back into the JSON config file structure."""

        return {
            "OPENAI_API_KEY": self.openai_api_key,
            "OPENAI_BASE_URL": self.openai_base_url,
            "OPENAI_CHAT_API_KEY": self.chat_api_key,
            "OPENAI_CHAT_BASE_URL": self.chat_base_url,
            "OPENAI_CHAT_MODEL": self.chat_model,
            "OPENAI_EMBEDDING_API_KEY": self.embedding_api_key,
            "OPENAI_EMBEDDING_BASE_URL": self.embedding_base_url,
            "OPENAI_EMBEDDING_MODEL": self.embedding_model,
            "APP_DEFAULT_LANGUAGE": self.default_language,
            "APP_DEFAULT_MEMORY_MODE": self.default_memory_mode,
            "APP_CHAT_TIMEOUT_SECONDS": self.chat_timeout_seconds,
            "APP_CHAT_STREAM_REQUEST_TIMEOUT_SECONDS": self.chat_stream_request_timeout_seconds,
            "APP_CHAT_STREAM_FIRST_TOKEN_TIMEOUT_SECONDS": self.chat_stream_first_token_timeout_seconds,
            "APP_CHAT_STREAM_IDLE_TIMEOUT_SECONDS": self.chat_stream_idle_timeout_seconds,
            "APP_EMBEDDING_TIMEOUT_SECONDS": self.embedding_timeout_seconds,
            "APP_API_RETRY_ATTEMPTS": self.api_retry_attempts,
            "APP_API_RETRY_BACKOFF_MIN_SECONDS": self.api_retry_backoff_min_seconds,
            "APP_API_RETRY_BACKOFF_MAX_SECONDS": self.api_retry_backoff_max_seconds,
            "APP_RATE_LIMIT_RETRY_FOREVER": self.rate_limit_retry_forever,
            "APP_RATE_LIMIT_RETRY_ATTEMPTS": self.rate_limit_retry_attempts,
            "APP_RATE_LIMIT_RETRY_DELAY_SECONDS": self.rate_limit_retry_delay_seconds,
            "APP_CHAT_PROVIDER_CONCURRENCY": self.chat_provider_concurrency,
            "APP_EMBEDDING_PROVIDER_CONCURRENCY": self.embedding_provider_concurrency,
            "APP_ALLOW_LOCAL_VECTOR_FALLBACK": self.allow_local_vector_fallback,
            "APP_VECTOR_UPSERT_BATCH_SIZE": self.vector_upsert_batch_size,
            "APP_MAX_INPUT_CHARS": self.max_input_chars,
            "APP_ENABLE_SENSITIVE_INPUT_CHECK": self.enable_sensitive_input_check,
            "APP_ENABLE_RESULT_CACHE": self.enable_result_cache,
            "APP_MODEL_CONTEXT_WINDOW": self.model_context_window,
            "APP_ANSWER_TOKEN_RESERVE": self.answer_token_reserve,
            "APP_LONG_CONTEXT_WINDOW_TOKENS": self.long_context_window_tokens,
            "APP_LONG_CONTEXT_WINDOW_OVERLAP_TOKENS": self.long_context_window_overlap_tokens,
            "APP_RECURSIVE_SUMMARY_TARGET_TOKENS": self.recursive_summary_target_tokens,
            "APP_RECURSIVE_SUMMARY_BATCH_SIZE": self.recursive_summary_batch_size,
            "APP_PROMPT_COMPRESSION_TURN_TOKEN_LIMIT": self.prompt_compression_turn_token_limit,
            "APP_RECENT_HISTORY_TURNS": self.recent_history_turns,
            "APP_HISTORY_SUMMARY_TOKEN_LIMIT": self.history_summary_token_limit,
            "APP_CONTEXT_OVERFLOW_RETRIES": self.context_overflow_retries,
            "APP_RETRIEVAL_TOP_K": self.retrieval_top_k,
            "APP_RETRIEVAL_FETCH_K": self.retrieval_fetch_k,
            "APP_CITATION_LIMIT": self.citation_limit,
            "APP_ENABLE_RERANK": self.enable_rerank,
            "APP_RERANK_MIN_SCORE": self.rerank_min_score,
            "APP_RERANK_MIN_KEEP": self.rerank_min_keep,
            "APP_RERANK_WEIGHT_VECTOR": self.rerank_weight_vector,
            "APP_RERANK_WEIGHT_KEYWORD": self.rerank_weight_keyword,
            "APP_RERANK_WEIGHT_PHRASE": self.rerank_weight_phrase,
            "APP_RERANK_WEIGHT_METADATA": self.rerank_weight_metadata,
            "APP_CHUNK_SIZE": self.chunk_size,
            "APP_CHUNK_OVERLAP": self.chunk_overlap,
            "APP_MERGE_SMALL_CHUNKS": self.merge_small_chunks,
            "APP_MIN_CHUNK_SIZE": self.min_chunk_size,
            "APP_BATCH_CONCURRENCY": self.batch_concurrency,
            "APP_ENABLE_QUERY_REWRITE": self.enable_query_rewrite,
            "APP_ENABLE_MIGRATION_UI": self.enable_migration_ui,
            "APP_DEFAULT_STREAMING_MODE": self.default_streaming_mode,
            "APP_QA_SYSTEM_PROMPT_EN": self.qa_system_prompt_en,
            "APP_QA_SYSTEM_PROMPT_ZH": self.qa_system_prompt_zh,
            "APP_QUERY_REWRITE_INSTRUCTION_EN": self.query_rewrite_instruction_en,
            "APP_QUERY_REWRITE_INSTRUCTION_ZH": self.query_rewrite_instruction_zh,
            "APP_QA_ANSWER_INSTRUCTION_EN": self.qa_answer_instruction_en,
            "APP_QA_ANSWER_INSTRUCTION_ZH": self.qa_answer_instruction_zh,
            "APP_SINGLE_ANALYSIS_PROMPT_EN": self.single_analysis_prompt_en,
            "APP_SINGLE_ANALYSIS_PROMPT_ZH": self.single_analysis_prompt_zh,
            "APP_COMPARE_REPORT_PROMPT_EN": self.compare_report_prompt_en,
            "APP_COMPARE_REPORT_PROMPT_ZH": self.compare_report_prompt_zh,
            "APP_DATA_EXTRACTION_PROMPT_EN": self.data_extraction_prompt_en,
            "APP_DATA_EXTRACTION_PROMPT_ZH": self.data_extraction_prompt_zh,
            "APP_TABLE_SUMMARY_PROMPT_EN": self.table_summary_prompt_en,
            "APP_TABLE_SUMMARY_PROMPT_ZH": self.table_summary_prompt_zh,
            "APP_DB_PATH": _serialize_path_for_config(self.project_root, self.db_path),
            "APP_VECTOR_DIR": _serialize_path_for_config(self.project_root, self.vector_dir),
            "APP_RUNTIME_API_LOG_PATH": _serialize_path_for_config(self.project_root, self.runtime_api_log_path),
            "APP_TEST_API_LOG_PATH": _serialize_path_for_config(self.project_root, self.test_api_log_path),
            "APP_VECTOR_OPERATION_LOG_PATH": _serialize_path_for_config(
                self.project_root,
                self.vector_operation_log_path,
            ),
            "APP_CACHE_DIR": _serialize_path_for_config(self.project_root, self.cache_dir),
            "APP_ANALYSIS_CHECKPOINT_DIR": _serialize_path_for_config(self.project_root, self.analysis_checkpoint_dir),
            "APP_FIELD_TEMPLATE_PATH": _serialize_path_for_config(self.project_root, self.field_template_path),
            "APP_KNOWLEDGE_BASE_STATE_PATH": _serialize_path_for_config(
                self.project_root,
                self.knowledge_base_state_path,
            ),
            "APP_DATA_ROOT": _serialize_path_for_config(self.project_root, self.data_root),
            "APP_REPORTS_DIR": _serialize_path_for_config(self.project_root, self.reports_dir),
        }

    def ensure_directories(self) -> None:
        """Create the local directories required by storage, reports, and raw files."""

        if self.storage_prepared:
            return
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.vector_dir.mkdir(parents=True, exist_ok=True)
        self.active_api_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.runtime_api_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.test_api_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.vector_operation_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.field_template_path.parent.mkdir(parents=True, exist_ok=True)
        self.knowledge_base_state_path.parent.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.migration_messages = _migrate_legacy_notebook_data(self)
        self.storage_prepared = True

    @property
    def resolved_chat_api_key(self) -> str:
        return self.chat_api_key or self.openai_api_key

    @property
    def resolved_chat_base_url(self) -> str:
        return _normalize_base_url(self.chat_base_url or self.openai_base_url)

    @property
    def resolved_embedding_api_key(self) -> str:
        return self.embedding_api_key or self.openai_api_key

    @property
    def resolved_embedding_base_url(self) -> str:
        return _normalize_base_url(self.embedding_base_url or self.openai_base_url)

    @property
    def has_chat_model_credentials(self) -> bool:
        return bool(
            self.resolved_chat_api_key
            and self.resolved_chat_base_url
            and self.chat_model
        )

    @property
    def has_embedding_model_credentials(self) -> bool:
        return bool(
            self.resolved_embedding_api_key
            and self.resolved_embedding_base_url
            and self.embedding_model
        )

    @property
    def has_model_credentials(self) -> bool:
        return self.has_chat_model_credentials and self.has_embedding_model_credentials

    @property
    def active_api_log_path(self) -> Path:
        if _is_test_runtime():
            return self.test_api_log_path
        return self.runtime_api_log_path

    @property
    def default_streaming(self) -> bool:
        return self.default_streaming_mode != "non_stream"

def _normalize_base_url(value: str) -> str:
    normalized = value.strip().rstrip("/")
    if not normalized:
        return ""
    for suffix in ("/chat/completions", "/completions", "/embeddings"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    return normalized


def _read_bool_setting(payload: dict[str, Any], name: str, default: bool) -> bool:
    value = payload.get(name)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _read_text_setting(payload: dict[str, Any], name: str, default: str) -> str:
    value = payload.get(name)
    if value is None:
        return default
    if not isinstance(value, str):
        return str(value)
    return value


def _is_test_runtime() -> bool:
    if os.getenv("PYTEST_CURRENT_TEST"):
        return True
    if "pytest" in sys.modules:
        return True
    if "unittest" in sys.modules and "ipykernel" not in sys.modules:
        return True
    return False


def _read_with_fallback(payload: dict[str, Any], name: str, fallback: str) -> str:
    value = payload.get(name)
    if value is None:
        return fallback
    normalized = str(value).strip()
    return normalized or fallback


def _read_setting(payload: dict[str, Any], name: str, default: Any) -> Any:
    return payload.get(name, default)

def _resolve_config_path(config_path: str | os.PathLike[str] | None) -> Path:
    if config_path:
        candidate = Path(config_path).expanduser()
        if candidate.is_absolute():
            return candidate.resolve(strict=False)
        resolved = candidate.resolve(strict=False)
        fallback = (_default_project_root() / candidate).resolve(strict=False)
        if resolved.exists() or candidate != Path("config/app_config.json"):
            return resolved
        return fallback
    return _default_config_path(_default_project_root())


def _default_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_config_path(project_root: Path) -> Path:
    return (project_root / "config" / "app_config.json").resolve(strict=False)


def _project_root_from_config_path(config_path: Path) -> Path:
    if config_path.parent.name == "config":
        return config_path.parent.parent.resolve(strict=False)
    return config_path.parent.resolve(strict=False)


def _resolve_path_from_root(project_root: Path, value: str | os.PathLike[str]) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve(strict=False)
    return (project_root / candidate).resolve(strict=False)


def _serialize_path_for_config(project_root: Path, value: Path) -> str:
    resolved_project_root = project_root.resolve(strict=False)
    resolved_value = value.resolve(strict=False)
    try:
        return str(resolved_value.relative_to(resolved_project_root))
    except ValueError:
        return str(resolved_value)


def _load_or_initialize_config(config_path: Path, project_root: Path) -> dict[str, Any]:
    payload = load_json_mapping(config_path)
    if not payload:
        payload = _build_default_flat_config(project_root)
        save_json_mapping(config_path, payload)
    _ensure_config_example(project_root)
    return payload


def _ensure_config_example(project_root: Path) -> None:
    example_path = project_root / "config" / "app_config.example.json"
    if example_path.exists():
        return
    save_json_mapping(example_path, _build_default_flat_config(project_root))


def _build_default_flat_config(project_root: Path) -> dict[str, Any]:
    default_config = AppConfig(
        project_root=project_root,
        config_path=_default_config_path(project_root),
        db_path=_resolve_path_from_root(project_root, "storage/app_state.db"),
        vector_dir=_resolve_path_from_root(project_root, "storage/chroma"),
        api_log_path=_resolve_path_from_root(project_root, "logs/runtime_api_traffic.jsonl"),
        runtime_api_log_path=_resolve_path_from_root(project_root, "logs/runtime_api_traffic.jsonl"),
        test_api_log_path=_resolve_path_from_root(project_root, "logs/test_api_traffic.jsonl"),
        cache_dir=_resolve_path_from_root(project_root, "storage/cache"),
        analysis_checkpoint_dir=_resolve_path_from_root(project_root, "storage/analysis_checkpoints"),
        field_template_path=_resolve_path_from_root(project_root, "storage/field_templates.json"),
        knowledge_base_state_path=_resolve_path_from_root(project_root, "storage/knowledge_base_state.json"),
        data_root=_resolve_path_from_root(project_root, "data/raw"),
        reports_dir=_resolve_path_from_root(project_root, "reports"),
    )
    return default_config.to_flat_config()


def _migrate_legacy_notebook_data(config: AppConfig) -> list[str]:
    legacy_root = config.project_root / "notebooks"
    if not legacy_root.exists():
        return []

    messages: list[str] = []
    directory_specs = [
        ("知识库原始文件", legacy_root / "data" / "raw", config.data_root),
        ("分析缓存", legacy_root / "storage" / "cache", config.cache_dir),
        ("分析检查点", legacy_root / "storage" / "analysis_checkpoints", config.analysis_checkpoint_dir),
        ("报告文件", legacy_root / "reports", config.reports_dir),
        ("向量库", legacy_root / "storage" / "chroma", config.vector_dir),
    ]
    for label, source_dir, target_dir in directory_specs:
        copied, backed_up = _merge_directory_non_destructive(source_dir, target_dir)
        if copied or backed_up:
            messages.append(
                f"{label}已从 {source_dir} 合并到 {target_dir}，新增 {copied} 个文件"
                + (f"，保留冲突副本 {backed_up} 个" if backed_up else "")
                + "。"
            )

    file_specs = [
        ("会话数据库", legacy_root / "storage" / "app_state.db", config.db_path),
        ("字段模板", legacy_root / "storage" / "field_templates.json", config.field_template_path),
        ("知识库状态", legacy_root / "storage" / "knowledge_base_state.json", config.knowledge_base_state_path),
        ("运行日志", legacy_root / "logs" / "runtime_api_traffic.jsonl", config.runtime_api_log_path),
        ("测试日志", legacy_root / "logs" / "test_api_traffic.jsonl", config.test_api_log_path),
        ("旧版日志", legacy_root / "logs" / "api_traffic.jsonl", config.project_root / "logs" / "api_traffic_legacy_notebook.jsonl"),
    ]
    for label, source_file, target_file in file_specs:
        status, actual_target = _copy_file_non_destructive(source_file, target_file)
        if status == "copied":
            messages.append(f"{label}已迁移到 {actual_target}。")
        elif status == "backup":
            messages.append(f"{label}与现有文件冲突，已另存到 {actual_target}。")

    rewritten = 0
    for root in (config.cache_dir, config.analysis_checkpoint_dir):
        rewritten += _rewrite_legacy_paths_in_json_tree(
            root,
            legacy_root=legacy_root,
            project_root=config.project_root,
        )
    if rewritten:
        messages.append(f"已修正 {rewritten} 个缓存/检查点文件中的旧 notebook 路径。")
    return messages


def _merge_directory_non_destructive(source_dir: Path, target_dir: Path) -> tuple[int, int]:
    if not source_dir.exists():
        return 0, 0
    copied = 0
    backed_up = 0
    for source_path in sorted(source_dir.rglob("*")):
        relative_path = source_path.relative_to(source_dir)
        target_path = target_dir / relative_path
        if source_path.is_dir():
            target_path.mkdir(parents=True, exist_ok=True)
            continue
        status, _ = _copy_file_non_destructive(source_path, target_path)
        if status == "copied":
            copied += 1
        elif status == "backup":
            backed_up += 1
    return copied, backed_up


def _copy_file_non_destructive(source_file: Path, target_file: Path) -> tuple[str, Path]:
    if not source_file.exists():
        return "missing", target_file
    target_file.parent.mkdir(parents=True, exist_ok=True)
    if not target_file.exists():
        shutil.copy2(source_file, target_file)
        return "copied", target_file
    if _same_file_signature(source_file, target_file):
        return "skipped", target_file
    backup_path = _legacy_backup_path(target_file, source_file)
    if not backup_path.exists():
        shutil.copy2(source_file, backup_path)
    return "backup", backup_path


def _legacy_backup_path(target_file: Path, source_file: Path) -> Path:
    suffix = "".join(target_file.suffixes)
    stem = target_file.name[: -len(suffix)] if suffix else target_file.name
    candidate = target_file.with_name(f"{stem}_legacy_notebook{suffix}")
    index = 2
    while candidate.exists() and not _same_file_signature(source_file, candidate):
        candidate = target_file.with_name(f"{stem}_legacy_notebook_{index}{suffix}")
        index += 1
    return candidate


def _same_file_signature(source_file: Path, target_file: Path) -> bool:
    if not source_file.exists() or not target_file.exists():
        return False
    source_stat = source_file.stat()
    target_stat = target_file.stat()
    return source_stat.st_size == target_stat.st_size and source_stat.st_mtime_ns == target_stat.st_mtime_ns


def _rewrite_legacy_paths_in_json_tree(root_dir: Path, *, legacy_root: Path, project_root: Path) -> int:
    if not root_dir.exists():
        return 0
    replacements = {
        str((legacy_root / "data" / "raw").resolve(strict=False)): str((project_root / "data" / "raw").resolve(strict=False)),
        str((legacy_root / "reports").resolve(strict=False)): str((project_root / "reports").resolve(strict=False)),
        str((legacy_root / "logs").resolve(strict=False)): str((project_root / "logs").resolve(strict=False)),
        str((legacy_root / "storage").resolve(strict=False)): str((project_root / "storage").resolve(strict=False)),
    }
    updated = 0
    for path in root_dir.rglob("*.json"):
        try:
            original = path.read_text(encoding="utf-8")
        except Exception:
            continue
        rewritten = original
        for marker, replacement in replacements.items():
            rewritten = rewritten.replace(marker, replacement)
        if rewritten != original:
            path.write_text(rewritten, encoding="utf-8")
            updated += 1
    return updated
