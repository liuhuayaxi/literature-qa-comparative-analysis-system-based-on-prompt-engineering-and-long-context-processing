"""Vector retrieval, reranking, and chat-oriented RAG orchestration."""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import shutil
import sqlite3
import sys
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Iterable, Literal

from .config import AppConfig
from .errors import AppServiceError, InputValidationError, OperationCancelledError, ProviderRequestError
from .llm_tools import (
    DEFAULT_QA_ANSWER_INSTRUCTION_EN,
    DEFAULT_QA_ANSWER_INSTRUCTION_ZH,
    DEFAULT_QA_SYSTEM_PROMPT_EN,
    DEFAULT_QA_SYSTEM_PROMPT_ZH,
    DEFAULT_QUERY_REWRITE_INSTRUCTION_EN,
    DEFAULT_QUERY_REWRITE_INSTRUCTION_ZH,
    build_embedding_model,
    build_query_rewrite_prompt,
    build_rag_prompt,
    _retry_delay_seconds,
    invoke_chat_text,
    stream_chat_text,
)
from .memory_store import BaseMemoryStore
from .models import ChatResponse, ChatTurn, ChunkCitation, DocumentRecord, SourceDocument
from .app_utils import (
    JsonResultCache,
    append_citations,
    build_cache_key,
    compress_text_for_prompt,
    estimate_token_count,
    resolve_output_language,
    sanitize_user_question_for_prompt,
    trim_text_to_token_limit,
    validate_user_text,
)

_STOPWORDS_EN = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "can",
    "do",
    "does",
    "for",
    "how",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "why",
    "with",
}
_STOPWORDS_ZH = {"什么", "怎么", "如何", "为什么", "哪个", "哪些", "是否", "一下", "这个", "那个"}
_PROMPT_COMPRESSION_TRIGGER_RATIO = 0.90
_PROMPT_COMPRESSION_STRONG_TRIGGER_RATIO = 0.97
_PROMPT_COMPRESSION_MIN_CONTEXT_WINDOW = 20000


@dataclass(slots=True)
class RerankResult:
    """Container for rerank inputs, filtered outputs, and filter statistics."""

    candidates: list[SourceDocument]
    kept_documents: list[SourceDocument]
    filtered_count: int
    low_score_filtered: bool


class LocalRetriever:
    """Minimal async retriever wrapper used when the app falls back to local mode."""

    def __init__(
        self,
        service: "VectorStoreService",
        course_id: str,
        source_types: list[str] | None = None,
        doc_ids: list[str] | None = None,
        k: int | None = None,
        fetch_k: int | None = None,
    ) -> None:
        self.service = service
        self.course_id = course_id
        self.source_types = source_types
        self.doc_ids = doc_ids
        self.k = k
        self.fetch_k = fetch_k

    async def aget_relevant_documents(self, query: str) -> list[SourceDocument]:
        return await self.service.similarity_search(
            course_id=self.course_id,
            query=query,
            source_types=self.source_types,
            doc_ids=self.doc_ids,
            k=self.k,
            fetch_k=self.fetch_k,
        )


class VectorStoreService:
    """Own document storage, recall, metadata updates, and rerank operations."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.config.ensure_directories()
        self._maybe_restore_legacy_vector_store()
        self._backend = None
        self._local_documents: list[SourceDocument] = []
        self._local_backend = True
        self._allow_local_fallback = bool(self.config.allow_local_vector_fallback or _is_test_runtime())
        self._initialize_backend()
        self._repair_legacy_file_paths()

    def _maybe_restore_legacy_vector_store(self) -> None:
        """Prefer the old notebook Chroma store when the project-root store is empty."""

        legacy_dir = self.config.project_root / "notebooks" / "storage" / "chroma"
        target_dir = self.config.vector_dir
        if not legacy_dir.exists() or legacy_dir.resolve(strict=False) == target_dir.resolve(strict=False):
            return
        legacy_count = _count_chroma_embeddings(legacy_dir)
        if legacy_count <= 0:
            return
        target_count = _count_chroma_embeddings(target_dir)
        if target_count >= legacy_count:
            return

        backup_dir = _next_backup_dir(target_dir)
        if target_dir.exists():
            shutil.move(str(target_dir), str(backup_dir))
        shutil.copytree(legacy_dir, target_dir, dirs_exist_ok=True)
        self.config.migration_messages.append(
            f"检测到项目根目录向量库为空，已优先接管旧的 notebook 向量库；原根目录向量库已备份到 {backup_dir}。"
        )

    def _initialize_backend(self) -> None:
        try:
            from langchain_chroma import Chroma
            from chromadb.config import Settings
        except ImportError:
            if self._allow_local_fallback:
                self._local_backend = True
                return
            raise RuntimeError("Chroma backend is unavailable. Please install the vector-store dependencies.")

        if not self.config.has_embedding_model_credentials:
            if self._allow_local_fallback:
                self._local_backend = True
                return
            raise RuntimeError("Embedding model credentials are missing. Vectorization cannot continue.")

        try:
            os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")
            logging.getLogger("chromadb.telemetry.product.posthog").disabled = True
            logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
            embeddings = build_embedding_model(self.config)
            settings = Settings(
                anonymized_telemetry=False,
                is_persistent=True,
                persist_directory=str(self.config.vector_dir),
            )
            self._backend = Chroma(
                collection_name="course_assistant",
                persist_directory=str(self.config.vector_dir),
                embedding_function=embeddings,
                client_settings=settings,
            )
            self._local_backend = False
            _append_vector_log(self.config, {"event": "backend_initialized", "backend_mode": "chroma"})
        except Exception as exc:
            self._backend = None
            self._local_backend = True
            _append_vector_log(
                self.config,
                {"event": "backend_init_error", "backend_mode": "chroma", "detail": str(exc)},
            )
            if not self._allow_local_fallback:
                raise RuntimeError(f"Failed to initialize Chroma vector backend: {exc}") from exc

    def _repair_legacy_file_paths(self) -> None:
        """Normalize only the local fallback paths used inside this process.

        Persistent Chroma metadata is left untouched at startup. Rewriting it
        would require deleting and re-embedding every stored chunk, which is too
        risky during automatic recovery.
        """

        legacy_prefix = str((self.config.project_root / "notebooks" / "data" / "raw").resolve(strict=False))
        current_prefix = str(self.config.data_root.resolve(strict=False))
        if legacy_prefix == current_prefix:
            return
        if self._local_backend:
            self._repair_local_legacy_file_paths(legacy_prefix, current_prefix)
        return

    def _repair_local_legacy_file_paths(self, legacy_prefix: str, current_prefix: str) -> None:
        for index, document in enumerate(self._local_documents):
            metadata = dict(document.metadata)
            file_path = str(metadata.get("file_path", ""))
            if legacy_prefix not in file_path:
                continue
            metadata["file_path"] = file_path.replace(legacy_prefix, current_prefix, 1)
            self._local_documents[index] = SourceDocument(page_content=document.page_content, metadata=metadata)

    @property
    def backend_mode(self) -> str:
        """Return the currently active backend name shown in the notebook UI."""

        return "local" if self._local_backend else "chroma"

    async def upsert_documents(
        self,
        docs: list[SourceDocument],
        progress_callback: Any | None = None,
    ) -> int:
        if self._local_backend:
            if self._allow_local_fallback:
                self._local_upsert(docs)
                return len(docs)
            raise RuntimeError("Vector store is not available. Chroma backend failed to initialize.")
        batch_size = max(1, int(self.config.vector_upsert_batch_size))
        total_batches = max(1, math.ceil(len(docs) / batch_size))
        stored = 0
        inserted_chunk_ids: list[str] = []
        _append_vector_log(
            self.config,
            {
                "event": "upsert_start",
                "backend_mode": self.backend_mode,
                "total_chunks": len(docs),
                "batch_size": batch_size,
                "total_batches": total_batches,
                "course_ids": sorted({str(doc.metadata.get("course_id", "")) for doc in docs if doc.metadata.get("course_id")}),
            },
        )
        for batch_index, start in enumerate(range(0, len(docs), batch_size), start=1):
            batch = docs[start : start + batch_size]
            if progress_callback is not None:
                await _emit_vector_progress(
                    progress_callback,
                    f"正在写入向量库批次 {batch_index}/{total_batches}，本批 {len(batch)} 个切片...",
                )
            _append_vector_log(
                self.config,
                {
                    "event": "upsert_batch_start",
                    "batch_index": batch_index,
                    "total_batches": total_batches,
                    "chunk_count": len(batch),
                    "sample_files": [str(doc.metadata.get("file_name", "")) for doc in batch[:3]],
                },
            )
            try:
                stored += await asyncio.to_thread(self._chroma_upsert_batch, batch)
                inserted_chunk_ids.extend(str(doc.metadata.get("chunk_id", "")) for doc in batch if doc.metadata.get("chunk_id"))
            except Exception as exc:
                if inserted_chunk_ids:
                    try:
                        await asyncio.to_thread(self._chroma_delete_chunk_ids, inserted_chunk_ids)
                        _append_vector_log(
                            self.config,
                            {
                                "event": "upsert_rollback_success",
                                "rolled_back_chunk_count": len(inserted_chunk_ids),
                            },
                        )
                    except Exception as rollback_exc:
                        _append_vector_log(
                            self.config,
                            {
                                "event": "upsert_rollback_error",
                                "rolled_back_chunk_count": len(inserted_chunk_ids),
                                "detail": str(rollback_exc),
                            },
                        )
                _append_vector_log(
                    self.config,
                    {
                        "event": "upsert_batch_error",
                        "batch_index": batch_index,
                        "total_batches": total_batches,
                        "chunk_count": len(batch),
                        "detail": str(exc),
                    },
                )
                raise RuntimeError(
                    f"向量库写入失败，停止在第 {batch_index}/{total_batches} 个批次。详细原因：{exc}"
                ) from exc
            _append_vector_log(
                self.config,
                {
                    "event": "upsert_batch_success",
                    "batch_index": batch_index,
                    "total_batches": total_batches,
                    "chunk_count": len(batch),
                    "stored_so_far": stored,
                },
            )
        _append_vector_log(self.config, {"event": "upsert_complete", "stored": stored, "total_chunks": len(docs)})
        return stored

    def _local_upsert(self, docs: list[SourceDocument]) -> None:
        existing_by_chunk = {doc.metadata.get("chunk_id"): index for index, doc in enumerate(self._local_documents)}
        for doc in docs:
            chunk_id = doc.metadata.get("chunk_id")
            if chunk_id in existing_by_chunk:
                self._local_documents[existing_by_chunk[chunk_id]] = doc
            else:
                self._local_documents.append(doc)

    def _chroma_upsert_batch(self, docs: list[SourceDocument]) -> int:
        from langchain_core.documents import Document

        langchain_docs = [
            Document(page_content=doc.page_content, metadata=doc.metadata)
            for doc in docs
        ]
        ids = [str(doc.metadata["chunk_id"]) for doc in docs]
        self._backend.add_documents(documents=langchain_docs, ids=ids)
        return len(docs)

    def _chroma_delete_chunk_ids(self, chunk_ids: list[str]) -> None:
        if chunk_ids:
            self._backend.delete(ids=chunk_ids)

    def get_retriever(
        self,
        course_id: str,
        source_types: list[str] | None = None,
        doc_ids: list[str] | None = None,
        k: int | None = None,
        fetch_k: int | None = None,
    ):
        """Return a retriever-like object for LangChain-style calling code."""

        if self._local_backend:
            if not self._allow_local_fallback:
                raise RuntimeError("Vector store is unavailable. Retrieval cannot use local fallback.")
            return LocalRetriever(
                self,
                course_id=course_id,
                source_types=source_types,
                doc_ids=doc_ids,
                k=k,
                fetch_k=fetch_k,
            )
        filter_payload = _build_filter(course_id=course_id, source_types=source_types, doc_ids=doc_ids)
        return self._backend.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k or self.config.retrieval_top_k,
                "fetch_k": fetch_k or self.config.retrieval_fetch_k,
                "filter": filter_payload,
            },
        )

    async def similarity_search(
        self,
        course_id: str,
        query: str,
        source_types: list[str] | None = None,
        doc_ids: list[str] | None = None,
        k: int | None = None,
        fetch_k: int | None = None,
    ) -> list[SourceDocument]:
        """Fetch the first-stage recalled chunks without reranking."""

        candidate_k = k or self.config.retrieval_top_k
        recalled = await self.recall_documents(
            course_id=course_id,
            query=query,
            source_types=source_types,
            doc_ids=doc_ids,
            candidate_k=candidate_k,
            fetch_k=fetch_k,
        )
        return recalled[:candidate_k]

    async def recall_documents(
        self,
        course_id: str,
        query: str,
        source_types: list[str] | None = None,
        doc_ids: list[str] | None = None,
        candidate_k: int | None = None,
        fetch_k: int | None = None,
    ) -> list[SourceDocument]:
        """Recall a wider candidate set for downstream reranking and filtering."""

        limit = candidate_k or self.config.retrieval_fetch_k
        if self._local_backend:
            if not self._allow_local_fallback:
                raise RuntimeError("Vector store is unavailable. Recall cannot continue.")
            return self._local_recall_documents(
                course_id,
                query,
                source_types,
                doc_ids,
                limit,
            )
        try:
            return await asyncio.to_thread(
                self._chroma_recall_documents,
                course_id,
                query,
                source_types,
                doc_ids,
                limit,
                fetch_k or self.config.retrieval_fetch_k,
            )
        except Exception as exc:
            _append_vector_log(self.config, {"event": "recall_error", "detail": str(exc), "course_id": course_id})
            if self._allow_local_fallback:
                self._backend = None
                self._local_backend = True
                return self._local_recall_documents(
                    course_id,
                    query,
                    source_types,
                    doc_ids,
                    limit,
                )
            raise

    def rerank_documents(
        self,
        query: str,
        documents: list[SourceDocument],
        *,
        top_k: int | None = None,
        min_score: float | None = None,
        min_keep: int | None = None,
        preferred_doc_ids: list[str] | None = None,
    ) -> RerankResult:
        """Score recalled chunks again and keep only the strongest ones.

        The rerank stage mixes four signals:
        1. First-stage vector or rank score.
        2. Keyword coverage in chunk text and metadata.
        3. Phrase-level exact matches.
        4. Metadata boosts from file names and section labels.
        """

        if not documents:
            return RerankResult(candidates=[], kept_documents=[], filtered_count=0, low_score_filtered=False)
        query_terms = _extract_terms(query)
        normalized_query = _normalize_search_text(query)
        weights = _normalize_weights(
            self.config.rerank_weight_vector,
            self.config.rerank_weight_keyword,
            self.config.rerank_weight_phrase,
            self.config.rerank_weight_metadata,
        )
        scored_documents: list[SourceDocument] = []
        preferred_set = {str(item) for item in (preferred_doc_ids or []) if str(item).strip()}
        for rank, document in enumerate(documents, start=1):
            metadata = document.metadata.copy()
            chunk_text = _normalize_search_text(document.page_content)
            metadata_text = _normalize_search_text(
                " ".join(
                    [
                        str(metadata.get("file_name", "")),
                        str(metadata.get("section_label", "")),
                        str(metadata.get("section", "")),
                        str(metadata.get("page_label", "")),
                    ]
                )
            )
            vector_score = _resolve_vector_score(metadata, rank, len(documents))
            keyword_score = _keyword_coverage_score(query_terms, chunk_text, metadata_text)
            phrase_score = _phrase_match_score(query_terms, normalized_query, chunk_text, metadata_text)
            metadata_score = _metadata_boost_score(query_terms, metadata_text)
            focus_score = 1.0 if preferred_set and str(metadata.get("doc_id", "")) in preferred_set else 0.0
            matched_terms = [
                term
                for term in query_terms
                if term and (term in chunk_text or term in metadata_text)
            ]
            final_score = (
                weights[0] * vector_score
                + weights[1] * keyword_score
                + weights[2] * phrase_score
                + weights[3] * metadata_score
            )
            if focus_score:
                final_score = min(1.0, final_score + 0.15)
            metadata["retrieval_score"] = round(final_score, 4)
            metadata["score_breakdown"] = {
                "vector": round(vector_score, 4),
                "keyword": round(keyword_score, 4),
                "phrase": round(phrase_score, 4),
                "metadata": round(metadata_score, 4),
                "focus": round(focus_score, 4),
            }
            metadata["matched_terms"] = matched_terms[:8]
            scored_documents.append(SourceDocument(page_content=document.page_content, metadata=metadata))
        scored_documents.sort(
            key=lambda item: (
                float(item.metadata.get("retrieval_score", 0.0) or 0.0),
                float(item.metadata.get("retrieval_vector_score", 0.0) or 0.0),
            ),
            reverse=True,
        )
        result_limit = max(1, top_k or self.config.retrieval_top_k)
        threshold = min_score if min_score is not None else self.config.rerank_min_score
        minimum_keep = max(0, min_keep if min_keep is not None else self.config.rerank_min_keep)
        thresholded = [
            document
            for document in scored_documents
            if float(document.metadata.get("retrieval_score", 0.0) or 0.0) >= threshold
        ]
        kept_documents = thresholded[:result_limit]
        required_keep = min(len(scored_documents), max(0, min(result_limit, minimum_keep)))
        if len(kept_documents) < required_keep:
            kept_documents = scored_documents[:required_keep]
        if not kept_documents and scored_documents:
            kept_documents = scored_documents[:1]
        filtered_count = max(0, len(scored_documents) - len(kept_documents))
        low_score_filtered = len(thresholded) < len(scored_documents)
        return RerankResult(
            candidates=scored_documents,
            kept_documents=kept_documents,
            filtered_count=filtered_count,
            low_score_filtered=low_score_filtered,
        )

    def _local_recall_documents(
        self,
        course_id: str,
        query: str,
        source_types: list[str] | None,
        doc_ids: list[str] | None,
        candidate_k: int,
    ) -> list[SourceDocument]:
        candidates = [
            doc
            for doc in self._local_documents
            if doc.metadata.get("course_id") == course_id
            and (not source_types or doc.metadata.get("source_type") in source_types)
            and (not doc_ids or doc.metadata.get("doc_id") in doc_ids)
        ]
        query_tokens = _tokenize(query)
        scored = [
            (self._score_tokens(query_tokens, _tokenize(doc.page_content)), doc)
            for doc in candidates
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        recalled: list[SourceDocument] = []
        for rank, (score, doc) in enumerate(scored[:candidate_k], start=1):
            metadata = doc.metadata.copy()
            metadata["retrieval_rank"] = rank
            metadata["retrieval_vector_score"] = round(_clamp_score(score), 4)
            recalled.append(SourceDocument(page_content=doc.page_content, metadata=metadata))
        return recalled

    def _score_tokens(self, query_tokens: Counter[str], doc_tokens: Counter[str]) -> float:
        dot = sum(query_tokens[token] * doc_tokens[token] for token in query_tokens)
        query_norm = math.sqrt(sum(value * value for value in query_tokens.values()))
        doc_norm = math.sqrt(sum(value * value for value in doc_tokens.values()))
        if not dot or not query_norm or not doc_norm:
            return 0.0
        return dot / (query_norm * doc_norm)

    def _chroma_recall_documents(
        self,
        course_id: str,
        query: str,
        source_types: list[str] | None,
        doc_ids: list[str] | None,
        candidate_k: int,
        fetch_k: int,
    ) -> list[SourceDocument]:
        filter_payload = _build_filter(course_id=course_id, source_types=source_types, doc_ids=doc_ids)
        try:
            scored_results = self._backend.similarity_search_with_relevance_scores(
                query=query,
                k=candidate_k,
                filter=filter_payload,
            )
            recalled: list[SourceDocument] = []
            for rank, (result, score) in enumerate(scored_results, start=1):
                metadata = dict(result.metadata)
                metadata["retrieval_rank"] = rank
                metadata["retrieval_vector_score"] = round(_clamp_score(score), 4)
                recalled.append(SourceDocument(page_content=result.page_content, metadata=metadata))
            if recalled:
                return recalled
        except Exception:
            pass
        results = self._backend.max_marginal_relevance_search(
            query=query,
            k=candidate_k,
            fetch_k=max(candidate_k, fetch_k),
            filter=filter_payload,
        )
        recalled = []
        for rank, result in enumerate(results, start=1):
            metadata = dict(result.metadata)
            metadata["retrieval_rank"] = rank
            metadata["retrieval_vector_score"] = round(_rank_to_score(rank, len(results)), 4)
            recalled.append(SourceDocument(page_content=result.page_content, metadata=metadata))
        return recalled

    async def get_document_chunks(self, course_id: str, doc_id: str) -> list[SourceDocument]:
        if self._local_backend:
            if not self._allow_local_fallback:
                raise RuntimeError("Vector store is unavailable. Chunk inspection cannot continue.")
            return [
                doc
                for doc in self._local_documents
                if doc.metadata.get("course_id") == course_id and doc.metadata.get("doc_id") == doc_id
            ]
        try:
            return await asyncio.to_thread(self._chroma_get_document_chunks, course_id, doc_id)
        except Exception as exc:
            _append_vector_log(self.config, {"event": "get_document_chunks_error", "detail": str(exc), "course_id": course_id, "doc_id": doc_id})
            if self._allow_local_fallback:
                self._backend = None
                self._local_backend = True
                return [
                    doc
                    for doc in self._local_documents
                    if doc.metadata.get("course_id") == course_id and doc.metadata.get("doc_id") == doc_id
                ]
            raise

    async def delete_documents(self, course_id: str, doc_ids: list[str]) -> int:
        if self._local_backend:
            if not self._allow_local_fallback:
                raise RuntimeError("Vector store is unavailable. Delete cannot continue.")
            return self._local_delete_documents(course_id, doc_ids)
        try:
            return await asyncio.to_thread(self._chroma_delete_documents, course_id, doc_ids)
        except Exception as exc:
            _append_vector_log(self.config, {"event": "delete_documents_error", "detail": str(exc), "course_id": course_id, "doc_ids": doc_ids})
            if self._allow_local_fallback:
                self._backend = None
                self._local_backend = True
                return self._local_delete_documents(course_id, doc_ids)
            raise

    async def update_document_metadata(
        self,
        course_id: str,
        doc_id: str,
        updates: dict[str, object],
    ) -> DocumentRecord:
        if self._local_backend:
            if not self._allow_local_fallback:
                raise RuntimeError("Vector store is unavailable. Metadata update cannot continue.")
            return self._local_update_document_metadata(course_id, doc_id, updates)
        try:
            return await asyncio.to_thread(self._chroma_update_document_metadata, course_id, doc_id, updates)
        except Exception as exc:
            _append_vector_log(self.config, {"event": "update_document_metadata_error", "detail": str(exc), "course_id": course_id, "doc_id": doc_id})
            if self._allow_local_fallback:
                self._backend = None
                self._local_backend = True
                return self._local_update_document_metadata(course_id, doc_id, updates)
            raise

    async def rename_course(self, old_course_id: str, new_course_id: str) -> int:
        if self._local_backend:
            if not self._allow_local_fallback:
                raise RuntimeError("Vector store is unavailable. Knowledge-base rename cannot continue.")
            return self._local_rename_course(old_course_id, new_course_id)
        try:
            return await asyncio.to_thread(self._chroma_rename_course, old_course_id, new_course_id)
        except Exception as exc:
            _append_vector_log(self.config, {"event": "rename_course_error", "detail": str(exc), "old_course_id": old_course_id, "new_course_id": new_course_id})
            if self._allow_local_fallback:
                self._backend = None
                self._local_backend = True
                return self._local_rename_course(old_course_id, new_course_id)
            raise

    def _chroma_get_document_chunks(self, course_id: str, doc_id: str) -> list[SourceDocument]:
        payload = self._backend.get(
            where={"$and": [{"course_id": course_id}, {"doc_id": doc_id}]},
            include=["documents", "metadatas"],
        )
        return [
            SourceDocument(page_content=page_content, metadata=metadata)
            for page_content, metadata in zip(payload.get("documents", []), payload.get("metadatas", []))
        ]

    async def list_documents(self, course_id: str) -> list[DocumentRecord]:
        if self._local_backend:
            if not self._allow_local_fallback:
                raise RuntimeError("Vector store is unavailable. Document list cannot continue.")
            return _dedupe_records(
                doc.metadata
                for doc in self._local_documents
                if doc.metadata.get("course_id") == course_id
            )
        try:
            return await asyncio.to_thread(self._chroma_list_documents, course_id)
        except Exception as exc:
            _append_vector_log(self.config, {"event": "list_documents_error", "detail": str(exc), "course_id": course_id})
            if self._allow_local_fallback:
                self._backend = None
                self._local_backend = True
                return _dedupe_records(
                    doc.metadata
                    for doc in self._local_documents
                    if doc.metadata.get("course_id") == course_id
                )
            raise

    async def list_course_ids(self) -> list[str]:
        if self._local_backend:
            if not self._allow_local_fallback:
                raise RuntimeError("Vector store is unavailable. Knowledge-base list cannot continue.")
            return sorted(
                {str(doc.metadata.get("course_id", "")) for doc in self._local_documents if doc.metadata.get("course_id")}
            )
        try:
            return await asyncio.to_thread(self._chroma_list_course_ids)
        except Exception as exc:
            _append_vector_log(self.config, {"event": "list_course_ids_error", "detail": str(exc)})
            if self._allow_local_fallback:
                self._backend = None
                self._local_backend = True
                return sorted(
                    {str(doc.metadata.get("course_id", "")) for doc in self._local_documents if doc.metadata.get("course_id")}
                )
            raise

    def _chroma_list_documents(self, course_id: str) -> list[DocumentRecord]:
        payload = self._backend.get(
            where={"course_id": course_id},
            include=["metadatas"],
        )
        return _dedupe_records(payload.get("metadatas", []))

    def _chroma_list_course_ids(self) -> list[str]:
        payload = self._backend.get(include=["metadatas"])
        return sorted(
            {
                str(metadata.get("course_id", ""))
                for metadata in payload.get("metadatas", [])
                if metadata.get("course_id")
            }
        )

    async def reset_course(self, course_id: str) -> None:
        if self._local_backend:
            if not self._allow_local_fallback:
                raise RuntimeError("Vector store is unavailable. Reset cannot continue.")
            self._local_documents = [
                doc for doc in self._local_documents if doc.metadata.get("course_id") != course_id
            ]
            return
        try:
            await asyncio.to_thread(self._chroma_reset_course, course_id)
        except Exception as exc:
            _append_vector_log(self.config, {"event": "reset_course_error", "detail": str(exc), "course_id": course_id})
            if self._allow_local_fallback:
                self._backend = None
                self._local_backend = True
                self._local_documents = [
                    doc for doc in self._local_documents if doc.metadata.get("course_id") != course_id
                ]
                return
            raise

    def _chroma_reset_course(self, course_id: str) -> None:
        payload = self._backend.get(where={"course_id": course_id}, include=[])
        ids = payload.get("ids", [])
        if ids:
            self._backend.delete(ids=ids)

    def _local_delete_documents(self, course_id: str, doc_ids: list[str]) -> int:
        before = len({doc.metadata.get("doc_id") for doc in self._local_documents if doc.metadata.get("course_id") == course_id})
        self._local_documents = [
            doc
            for doc in self._local_documents
            if not (doc.metadata.get("course_id") == course_id and doc.metadata.get("doc_id") in doc_ids)
        ]
        after = len({doc.metadata.get("doc_id") for doc in self._local_documents if doc.metadata.get("course_id") == course_id})
        return before - after

    def _chroma_delete_documents(self, course_id: str, doc_ids: list[str]) -> int:
        payload = self._backend.get(
            where=_build_filter(course_id=course_id, doc_ids=doc_ids),
            include=[],
        )
        ids = payload.get("ids", [])
        if ids:
            self._backend.delete(ids=ids)
        return len(doc_ids)

    def _local_update_document_metadata(
        self,
        course_id: str,
        doc_id: str,
        updates: dict[str, object],
    ) -> DocumentRecord:
        updated_metadatas = []
        for index, document in enumerate(self._local_documents):
            metadata = document.metadata
            if metadata.get("course_id") == course_id and metadata.get("doc_id") == doc_id:
                new_metadata = metadata.copy()
                new_metadata.update(updates)
                self._local_documents[index] = SourceDocument(
                    page_content=document.page_content,
                    metadata=new_metadata,
                )
                updated_metadatas.append(new_metadata)
        if not updated_metadatas:
            raise ValueError(f"Document not found: course_id={course_id}, doc_id={doc_id}")
        return _record_from_metadata(updated_metadatas[0], chunk_count=len(updated_metadatas))

    def _chroma_update_document_metadata(
        self,
        course_id: str,
        doc_id: str,
        updates: dict[str, object],
    ) -> DocumentRecord:
        payload = self._backend.get(
            where={"$and": [{"course_id": course_id}, {"doc_id": doc_id}]},
            include=["metadatas"],
        )
        ids = payload.get("ids", [])
        metadatas = payload.get("metadatas", [])
        if not ids:
            raise ValueError(f"Document not found: course_id={course_id}, doc_id={doc_id}")
        new_metadatas = []
        for metadata in metadatas:
            new_metadata = dict(metadata)
            new_metadata.update(updates)
            new_metadatas.append(new_metadata)
        self._backend._collection.update(ids=ids, metadatas=new_metadatas)
        return _record_from_metadata(new_metadatas[0], chunk_count=len(new_metadatas))

    def _local_rename_course(self, old_course_id: str, new_course_id: str) -> int:
        updated = 0
        marker = str((self.config.data_root / old_course_id).resolve(strict=False))
        replacement = str((self.config.data_root / new_course_id).resolve(strict=False))
        for index, document in enumerate(self._local_documents):
            metadata = document.metadata
            if metadata.get("course_id") != old_course_id:
                continue
            new_metadata = metadata.copy()
            new_metadata["course_id"] = new_course_id
            file_path = str(new_metadata.get("file_path", ""))
            if marker in file_path:
                new_metadata["file_path"] = file_path.replace(marker, replacement, 1)
            self._local_documents[index] = SourceDocument(page_content=document.page_content, metadata=new_metadata)
            updated += 1
        return updated

    def _chroma_rename_course(self, old_course_id: str, new_course_id: str) -> int:
        payload = self._backend.get(
            where={"course_id": old_course_id},
            include=["metadatas"],
        )
        ids = payload.get("ids", [])
        metadatas = payload.get("metadatas", [])
        if not ids:
            return 0
        marker = str((self.config.data_root / old_course_id).resolve(strict=False))
        replacement = str((self.config.data_root / new_course_id).resolve(strict=False))
        new_metadatas = []
        for metadata in metadatas:
            new_metadata = dict(metadata)
            new_metadata["course_id"] = new_course_id
            file_path = str(new_metadata.get("file_path", ""))
            if marker in file_path:
                new_metadata["file_path"] = file_path.replace(marker, replacement, 1)
            new_metadatas.append(new_metadata)
        self._backend._collection.update(ids=ids, metadatas=new_metadatas)
        return len(ids)


def _build_filter(
    course_id: str,
    source_types: list[str] | None = None,
    doc_ids: list[str] | None = None,
) -> dict:
    conditions: list[dict] = [{"course_id": course_id}]
    if source_types:
        conditions.append({"source_type": {"$in": source_types}})
    if doc_ids:
        conditions.append({"doc_id": {"$in": doc_ids}})
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _dedupe_records(metadatas: Iterable[dict]) -> list[DocumentRecord]:
    grouped: dict[str, list[dict]] = {}
    for metadata in metadatas:
        doc_id = metadata.get("doc_id")
        if not doc_id:
            continue
        grouped.setdefault(str(doc_id), []).append(metadata)
    records: list[DocumentRecord] = []
    for _, group in grouped.items():
        records.append(_record_from_metadata(group[0], chunk_count=len(group)))
    return records


def _tokenize(text: str) -> Counter[str]:
    tokens = re.findall(r"[\w\u4e00-\u9fff]+", text.lower())
    return Counter(tokens)


def _extract_terms(text: str) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for token in re.findall(r"[\w\u4e00-\u9fff]+", text.lower()):
        normalized = token.strip("_")
        if not normalized:
            continue
        if normalized.isascii() and len(normalized) < 2:
            continue
        if normalized in _STOPWORDS_EN or normalized in _STOPWORDS_ZH:
            continue
        if normalized not in seen:
            terms.append(normalized)
            seen.add(normalized)
    return terms


def _normalize_search_text(text: str) -> str:
    return " ".join(re.findall(r"[\w\u4e00-\u9fff]+", text.lower()))


def _normalize_weights(*weights: float) -> tuple[float, float, float, float]:
    total = sum(max(0.0, float(weight)) for weight in weights)
    if total <= 0:
        return 0.45, 0.25, 0.20, 0.10
    normalized = [max(0.0, float(weight)) / total for weight in weights]
    return normalized[0], normalized[1], normalized[2], normalized[3]


def _clamp_score(value: float | int | None) -> float:
    if value is None:
        return 0.0
    score = float(value)
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


def _rank_to_score(rank: int, total: int) -> float:
    if total <= 1:
        return 1.0
    return max(0.0, 1.0 - ((rank - 1) / total))


def _resolve_vector_score(metadata: dict[str, object], rank: int, total: int) -> float:
    if "retrieval_vector_score" in metadata:
        return _clamp_score(metadata.get("retrieval_vector_score"))
    return _rank_to_score(rank, total)


def _keyword_coverage_score(query_terms: list[str], chunk_text: str, metadata_text: str) -> float:
    if not query_terms:
        return 0.0
    haystack = f"{chunk_text} {metadata_text}".strip()
    matched = sum(1 for term in query_terms if term in haystack)
    return matched / len(query_terms)


def _phrase_match_score(
    query_terms: list[str],
    normalized_query: str,
    chunk_text: str,
    metadata_text: str,
) -> float:
    haystack = f"{chunk_text} {metadata_text}".strip()
    if not haystack:
        return 0.0
    exact_match = 1.0 if normalized_query and normalized_query in haystack else 0.0
    phrases: list[str] = []
    for size in (2, 3):
        if len(query_terms) < size:
            continue
        for index in range(len(query_terms) - size + 1):
            phrases.append(" ".join(query_terms[index : index + size]))
    if not phrases:
        return exact_match
    unique_phrases = list(dict.fromkeys(phrases))
    matched = sum(1 for phrase in unique_phrases if phrase and phrase in haystack)
    phrase_ratio = matched / len(unique_phrases)
    return min(1.0, max(exact_match, (0.6 * exact_match) + (0.4 * phrase_ratio)))


def _metadata_boost_score(query_terms: list[str], metadata_text: str) -> float:
    if not query_terms or not metadata_text:
        return 0.0
    matched = sum(1 for term in query_terms if term in metadata_text)
    return matched / len(query_terms)


def _record_from_metadata(metadata: dict, chunk_count: int) -> DocumentRecord:
    file_ext = str(metadata.get("file_ext", "txt")).lower()
    if file_ext not in {"pdf", "md", "txt", "docx"}:
        file_ext = "txt"
    source_type = str(metadata.get("source_type", "lecture"))
    if source_type not in {"lecture", "assignment", "paper"}:
        source_type = "lecture"
    language = str(metadata.get("language", "en"))
    if language not in {"zh", "en", "mixed"}:
        language = "en"
    return DocumentRecord(
        doc_id=str(metadata.get("doc_id", "")),
        course_id=str(metadata.get("course_id", "")),
        source_type=source_type,
        file_name=str(metadata.get("file_name", "")),
        file_path=str(metadata.get("file_path", "")),
        file_ext=file_ext,
        language=language,
        chunk_count=chunk_count,
        is_vectorized=True,
    )


@dataclass(slots=True)
class HistoryContext:
    """Compact history bundle used during query rewrite and answer generation."""

    recent_turns: list[ChatTurn]
    history_summary: str
    focus_doc_ids: list[str]
    focus_file_names: list[str]
    selected_doc_ids: list[str]
    selected_doc_titles: list[str]


@dataclass(slots=True)
class RetrievalStats:
    """Runtime retrieval statistics shown in the notebook session panel."""

    candidate_doc_count: int = 0
    kept_doc_count: int = 0
    filtered_doc_count: int = 0
    low_score_filtered: bool = False


@dataclass(slots=True)
class PromptPlan:
    """Snapshot of the final prompt and its inputs after compression logic runs."""

    prompt: str
    prompt_docs: list[SourceDocument]
    citation_docs: list[SourceDocument]
    recent_turns: list[ChatTurn]
    history_summary: str
    token_estimate: int
    compressed: bool
    strategies_used: list[str]
    focus_sources: list[str]
    selected_document_titles: list[str]


class RAGChatService:
    """Drive multi-turn RAG chat on top of the vector store and memory stores."""

    def __init__(
        self,
        config: AppConfig,
        vector_store: VectorStoreService,
        session_memory_store: BaseMemoryStore,
        persistent_memory_store: BaseMemoryStore,
    ) -> None:
        self.config = config
        self.vector_store = vector_store
        self.session_memory_store = session_memory_store
        self.persistent_memory_store = persistent_memory_store
        self.result_cache = JsonResultCache(config.cache_dir)

    async def stream_answer(
        self,
        session_id: str,
        course_id: str,
        question: str,
        memory_mode: Literal["session", "persistent"],
        language: Literal["auto", "zh", "en"] = "auto",
        doc_ids: list[str] | None = None,
        enable_query_rewrite: bool = True,
        qa_system_prompt_override: str | dict[str, str] | None = None,
        rewrite_instruction_override: str | dict[str, str] | None = None,
        answer_instruction_override: str | dict[str, str] | None = None,
        retrieval_top_k: int | None = None,
        retrieval_fetch_k: int | None = None,
        citation_limit: int = 4,
        streaming_mode: Literal["stream", "non_stream"] = "stream",
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream a RAG answer while updating UI-friendly status and citation events."""

        resolved_language = resolve_output_language(language, question)
        try:
            validated_question = validate_user_text(
                question,
                language=resolved_language,
                max_chars=self.config.max_input_chars,
                enable_sensitive_check=self.config.enable_sensitive_input_check,
            )
        except InputValidationError as exc:
            yield {"type": "error", "content": exc.user_message}
            return
        effective_question, question_strategies = sanitize_user_question_for_prompt(
            validated_question,
            language=resolved_language,
            model_name=self.config.chat_model,
            token_limit=max(256, min(8192, self.config.model_context_window // 4)),
        )
        if question_strategies:
            yield {
                "type": "status",
                "content": (
                    "检测到输入中存在大量重复内容，已自动做问题去噪/压缩。"
                    if resolved_language == "zh"
                    else "Detected heavy repetitive input; automatically denoised/compressed the question."
                ),
            }
        memory_store = self._select_memory_store(memory_mode)
        yield {
            "type": "status",
            "content": "正在读取会话历史..." if resolved_language == "zh" else "Loading conversation history...",
        }
        selected_doc_titles = await self._load_selected_document_titles(course_id, doc_ids)
        history_context = await self._prepare_history_context(
            memory_store,
            session_id,
            resolved_language,
            selected_doc_ids=doc_ids,
            selected_doc_titles=selected_doc_titles,
        )
        focus_doc_ids = _resolve_follow_up_focus_doc_ids(
            question=effective_question,
            language=resolved_language,
            selected_doc_ids=doc_ids or [],
            history_context=history_context,
        )
        if enable_query_rewrite:
            yield {
                "type": "status",
                "content": "正在改写问题..." if resolved_language == "zh" else "Rewriting the question...",
            }
        resolved_qa_system_prompt = _resolve_prompt_override(qa_system_prompt_override, resolved_language)
        resolved_rewrite_instruction = _resolve_prompt_override(rewrite_instruction_override, resolved_language)
        resolved_answer_instruction = _resolve_prompt_override(answer_instruction_override, resolved_language)
        result_top_k = retrieval_top_k or self.config.retrieval_top_k
        candidate_k = max(result_top_k, retrieval_fetch_k or self.config.retrieval_fetch_k)
        rewritten_query = await self._rewrite_query(
            course_id=course_id,
            question=effective_question,
            history=history_context.recent_turns,
            language=resolved_language,
            enable_query_rewrite=enable_query_rewrite,
            instruction_override=resolved_rewrite_instruction,
            history_summary=history_context.history_summary,
            focus_doc_ids=focus_doc_ids,
            focus_file_names=history_context.focus_file_names,
            selected_document_titles=selected_doc_titles,
        )
        yield {
            "type": "status",
            "content": "正在检索相关片段..." if resolved_language == "zh" else "Retrieving relevant chunks...",
        }
        candidate_docs = await self.vector_store.recall_documents(
            course_id=course_id,
            query=rewritten_query,
            source_types=["lecture", "assignment", "paper"],
            doc_ids=doc_ids,
            candidate_k=candidate_k,
            fetch_k=retrieval_fetch_k or self.config.retrieval_fetch_k,
        )
        retrieval_stats = RetrievalStats(candidate_doc_count=len(candidate_docs))
        if self.config.enable_rerank:
            yield {
                "type": "status",
                "content": (
                    f"正在对 {len(candidate_docs)} 个候选切片评分排序..."
                    if resolved_language == "zh"
                    else f"Scoring and reranking {len(candidate_docs)} candidate chunks..."
                ),
            }
            rerank_result = self.vector_store.rerank_documents(
                rewritten_query,
                candidate_docs,
                top_k=result_top_k,
                min_score=self.config.rerank_min_score,
                min_keep=self.config.rerank_min_keep,
                preferred_doc_ids=focus_doc_ids,
            )
            retrieved_docs = rerank_result.kept_documents
            retrieval_stats = RetrievalStats(
                candidate_doc_count=len(rerank_result.candidates),
                kept_doc_count=len(rerank_result.kept_documents),
                filtered_doc_count=rerank_result.filtered_count,
                low_score_filtered=rerank_result.low_score_filtered,
            )
        else:
            retrieved_docs = candidate_docs[:result_top_k]
            retrieval_stats = RetrievalStats(
                candidate_doc_count=len(candidate_docs),
                kept_doc_count=len(retrieved_docs),
                filtered_doc_count=max(0, len(candidate_docs) - len(retrieved_docs)),
                low_score_filtered=False,
            )
        if _is_collection_scope_question(effective_question, doc_ids or []):
            retrieved_docs = _ensure_selected_doc_coverage(
                candidate_docs=candidate_docs if not self.config.enable_rerank else rerank_result.candidates,
                kept_documents=retrieved_docs,
                selected_doc_ids=doc_ids or [],
                result_limit=max(result_top_k, min(len(doc_ids or []), 8)),
            )
            retrieval_stats.kept_doc_count = len(retrieved_docs)
        if not retrieved_docs:
            message = _no_evidence_message(resolved_language)
            await self._persist_turns(
                memory_store,
                session_id,
                validated_question,
                message,
                course_id,
                resolved_language,
                memory_mode,
                prompt_plan=None,
                retrieval_stats=retrieval_stats,
                selected_doc_ids=doc_ids,
                selected_doc_titles=selected_doc_titles,
                citations=[],
            )
            yield {"type": "token", "content": message}
            yield {"type": "done", "content": {"answer": message, "citations": []}}
            return

        answer_text = ""
        used_prompt_plan: PromptPlan | None = None
        if not self.config.has_chat_model_credentials:
            used_prompt_plan = self._build_answer_prompt_plan(
                question=effective_question,
                retrieved_docs=retrieved_docs,
                language=resolved_language,
                history_context=history_context,
                qa_system_prompt_override=resolved_qa_system_prompt,
                answer_instruction_override=resolved_answer_instruction,
                degrade_level=0,
                focus_sources=history_context.focus_file_names,
                selected_document_titles=selected_doc_titles,
            )
            for strategy in question_strategies:
                if strategy not in used_prompt_plan.strategies_used:
                    used_prompt_plan.strategies_used.append(strategy)
            answer_text = _offline_answer(effective_question, retrieved_docs, resolved_language)
            citations = _build_citations(used_prompt_plan.citation_docs[: max(1, citation_limit)])
            for token in _chunk_text(answer_text):
                yield {"type": "token", "content": token}
        else:
            citations: list[ChunkCitation] = []
            retry_limit = max(0, self.config.context_overflow_retries)
            rate_limit_retry_forever = bool(self.config.rate_limit_retry_forever)
            rate_limit_retry_total = max(0, int(self.config.rate_limit_retry_attempts))
            general_retry_total = max(0, int(self.config.api_retry_attempts) - 1)
            for degrade_level in range(retry_limit + 1):
                plan = self._build_answer_prompt_plan(
                    question=effective_question,
                    retrieved_docs=retrieved_docs,
                    language=resolved_language,
                    history_context=history_context,
                    qa_system_prompt_override=resolved_qa_system_prompt,
                    answer_instruction_override=resolved_answer_instruction,
                    degrade_level=degrade_level,
                    focus_sources=history_context.focus_file_names,
                    selected_document_titles=selected_doc_titles,
                )
                used_prompt_plan = plan
                for strategy in question_strategies:
                    if strategy not in used_prompt_plan.strategies_used:
                        used_prompt_plan.strategies_used.append(strategy)
                citations = _build_citations(plan.citation_docs[: max(1, citation_limit)])
                status_text = "正在生成回答..." if resolved_language == "zh" else "Generating the answer..."
                if plan.compressed:
                    status_text = (
                        f"正在压缩上下文并生成回答... 预计输入 {plan.token_estimate} tokens"
                        if resolved_language == "zh"
                        else f"Compressing context before answering... estimated input {plan.token_estimate} tokens"
                    )
                if degrade_level > 0:
                    status_text = (
                        f"上下文仍偏长，正在降级重试（第 {degrade_level + 1} 次）..."
                        if resolved_language == "zh"
                        else f"Context still too long, retrying with a smaller prompt (attempt {degrade_level + 1})..."
                    )
                yield {"type": "status", "content": status_text}
                rate_limit_retries = 0
                general_retries = 0
                while True:
                    answer_text = ""
                    try:
                        if streaming_mode == "non_stream":
                            answer_text = await self._invoke_non_stream_with_events(
                                prompt=plan.prompt,
                                language=resolved_language,
                            )
                            for chunk in _chunk_text(answer_text):
                                if chunk:
                                    yield {"type": "token", "content": chunk}
                        else:
                            async for chunk in stream_chat_text(
                                self.config,
                                plan.prompt,
                                resolved_language,
                            ):
                                if chunk:
                                    answer_text += chunk
                                    yield {"type": "token", "content": chunk}
                        break
                    except OperationCancelledError as exc:
                        yield {"type": "error", "content": exc.user_message}
                        return
                    except ProviderRequestError as exc:  # pragma: no cover - network and provider behavior.
                        if exc.code == "context_length" and degrade_level < retry_limit:
                            yield {
                                "type": "status",
                                "content": "检测到上下文超限，正在自动缩短输入..."
                                if resolved_language == "zh"
                                else "Context window exceeded. Automatically shrinking the prompt...",
                            }
                            break
                        if exc.code == "rate_limit" and (rate_limit_retry_forever or rate_limit_retries < rate_limit_retry_total):
                            rate_limit_retries += 1
                            wait_seconds = max(1, int(self.config.rate_limit_retry_delay_seconds))
                            if answer_text.strip():
                                yield {
                                    "type": "status",
                                    "content": (
                                        f"生成过程中触发频率限制，{wait_seconds} 秒后将从头重试回答（第 {rate_limit_retries} 次）..."
                                        if resolved_language == "zh"
                                        else f"Rate limited during streaming. Restarting the answer in {wait_seconds} seconds (attempt {rate_limit_retries})..."
                                    ),
                                }
                                yield {"type": "restart_answer", "content": ""}
                            yield {
                                "type": "status",
                                "content": (
                                    f"触发频率限制，{wait_seconds} 秒后继续自动重试（第 {rate_limit_retries} 次）..."
                                    if resolved_language == "zh" and rate_limit_retry_forever
                                    else (
                                        f"触发频率限制，{wait_seconds} 秒后自动重试（第 {rate_limit_retries}/{rate_limit_retry_total} 次）..."
                                        if resolved_language == "zh"
                                        else (
                                            f"Rate limited. Retrying again in {wait_seconds} seconds (attempt {rate_limit_retries})..."
                                            if rate_limit_retry_forever
                                            else f"Rate limited. Retrying in {wait_seconds} seconds ({rate_limit_retries}/{rate_limit_retry_total})..."
                                        )
                                    )
                                ),
                            }
                            await asyncio.sleep(wait_seconds)
                            continue
                        if answer_text.strip():
                            yield {"type": "error", "content": exc.user_message}
                            return
                        if exc.retryable and general_retries < general_retry_total:
                            general_retries += 1
                            wait_seconds = _retry_delay_seconds(self.config, general_retries)
                            yield {
                                "type": "status",
                                "content": (
                                    f"{exc.user_message} 系统将在 {int(wait_seconds)} 秒后自动重试（第 {general_retries}/{general_retry_total} 次）..."
                                    if resolved_language == "zh"
                                    else f"{exc.user_message} Retrying in {int(wait_seconds)} seconds ({general_retries}/{general_retry_total})..."
                                ),
                            }
                            await asyncio.sleep(wait_seconds)
                            continue
                        yield {"type": "error", "content": exc.user_message}
                        return
                if answer_text.strip():
                    break

        if not answer_text.strip():
            answer_text = _no_evidence_message(resolved_language)

        for citation in citations:
            yield {"type": "citation", "content": citation.model_dump()}

        final_answer = append_citations(answer_text, citations, language=resolved_language)
        yield {
            "type": "status",
            "content": "正在写入会话记录..." if resolved_language == "zh" else "Saving the conversation...",
        }
        await self._persist_turns(
            memory_store,
            session_id,
            validated_question,
            final_answer,
            course_id,
            resolved_language,
            memory_mode,
            prompt_plan=used_prompt_plan,
            retrieval_stats=retrieval_stats,
            selected_doc_ids=doc_ids,
            selected_doc_titles=selected_doc_titles,
            citations=citations,
        )
        yield {
            "type": "done",
            "content": ChatResponse(
                answer=final_answer,
                citations=citations,
                language=resolved_language,
                used_memory_mode=memory_mode,
                session_id=session_id,
            ).model_dump(),
        }

    async def answer(
        self,
        session_id: str,
        course_id: str,
        question: str,
        memory_mode: Literal["session", "persistent"],
        language: Literal["auto", "zh", "en"] = "auto",
        doc_ids: list[str] | None = None,
        enable_query_rewrite: bool = True,
        qa_system_prompt_override: str | dict[str, str] | None = None,
        rewrite_instruction_override: str | dict[str, str] | None = None,
        answer_instruction_override: str | dict[str, str] | None = None,
        retrieval_top_k: int | None = None,
        retrieval_fetch_k: int | None = None,
        citation_limit: int = 4,
        streaming_mode: Literal["stream", "non_stream"] = "stream",
    ) -> ChatResponse:
        """Collect the stream-based answer flow into one final response object."""

        final_payload: dict[str, Any] | None = None
        streamed_text = ""
        error_message: str | None = None
        async for event in self.stream_answer(
            session_id=session_id,
            course_id=course_id,
            question=question,
            memory_mode=memory_mode,
            language=language,
            doc_ids=doc_ids,
            enable_query_rewrite=enable_query_rewrite,
            qa_system_prompt_override=qa_system_prompt_override,
            rewrite_instruction_override=rewrite_instruction_override,
            answer_instruction_override=answer_instruction_override,
            retrieval_top_k=retrieval_top_k,
            retrieval_fetch_k=retrieval_fetch_k,
            citation_limit=citation_limit,
            streaming_mode=streaming_mode,
        ):
            if event["type"] == "token":
                streamed_text += str(event["content"])
            elif event["type"] == "done":
                final_payload = dict(event["content"])
            elif event["type"] == "error":
                error_message = str(event["content"])
        if error_message:
            raise AppServiceError(error_message)
        if final_payload is None:
            resolved_language = resolve_output_language(language, question)
            return ChatResponse(
                answer=streamed_text or _no_evidence_message(resolved_language),
                citations=[],
                language=resolved_language,
                used_memory_mode=memory_mode,
                session_id=session_id,
            )
        return ChatResponse.model_validate(final_payload)

    def _select_memory_store(self, memory_mode: Literal["session", "persistent"]) -> BaseMemoryStore:
        if memory_mode == "persistent":
            return self.persistent_memory_store
        return self.session_memory_store

    async def _load_selected_document_titles(
        self,
        course_id: str,
        doc_ids: list[str] | None,
    ) -> list[str]:
        """Load selected file titles for prompt grounding when the scope is small."""

        if not doc_ids or len(doc_ids) > 12:
            return []
        records = await self.vector_store.list_documents(course_id)
        title_map = {record.doc_id: record.file_name for record in records}
        return [str(title_map.get(doc_id, doc_id)) for doc_id in doc_ids if str(title_map.get(doc_id, doc_id)).strip()]

    async def _rewrite_query(
        self,
        course_id: str,
        question: str,
        history: list[ChatTurn],
        language: Literal["zh", "en"],
        enable_query_rewrite: bool = True,
        instruction_override: str | None = None,
        history_summary: str = "",
        focus_doc_ids: list[str] | None = None,
        focus_file_names: list[str] | None = None,
        selected_document_titles: list[str] | None = None,
    ) -> str:
        if not enable_query_rewrite or (not history and not history_summary.strip()):
            return question
        cache_namespace = f"rewrite_{course_id}"
        cache_key = build_cache_key(
            language,
            question,
            history_summary,
            [(turn.role, turn.content) for turn in history],
            instruction_override or "",
            focus_doc_ids or [],
            focus_file_names or [],
            selected_document_titles or [],
        )
        if self.config.enable_result_cache:
            cached = self.result_cache.get(cache_namespace, cache_key)
            if isinstance(cached, dict) and str(cached.get("rewritten_query", "")).strip():
                return str(cached.get("rewritten_query", "")).strip()
        if not self.config.has_chat_model_credentials:
            previous = history[-2:] if len(history) > 1 else history
            prefix = " ".join(turn.content for turn in previous)
            return f"{prefix} {question}".strip()
        prompt = self._build_rewrite_prompt(
            question=question,
            history=history,
            language=language,
            instruction_override=instruction_override,
            history_summary=history_summary,
            focus_file_names=focus_file_names,
            selected_document_titles=selected_document_titles,
        )
        try:
            result = await invoke_chat_text(self.config, prompt, language)
        except ProviderRequestError:
            previous = history[-2:] if len(history) > 1 else history
            prefix = " ".join(turn.content for turn in previous)
            return f"{prefix} {question}".strip()
        rewritten = result.strip() or question
        if self.config.enable_result_cache:
            self.result_cache.set(cache_namespace, cache_key, {"rewritten_query": rewritten})
        return rewritten

    async def _invoke_non_stream_with_events(
        self,
        *,
        prompt: str,
        language: Literal["zh", "en"],
    ) -> str:
        """Run one non-streaming answer request for users who disable streaming."""

        return await invoke_chat_text(self.config, prompt, language)

    async def _persist_turns(
        self,
        memory_store: BaseMemoryStore,
        session_id: str,
        question: str,
        answer: str,
        course_id: str,
        language: str,
        memory_mode: str,
        prompt_plan: PromptPlan | None = None,
        retrieval_stats: RetrievalStats | None = None,
        selected_doc_ids: list[str] | None = None,
        selected_doc_titles: list[str] | None = None,
        citations: list[ChunkCitation] | None = None,
    ) -> None:
        timestamp = datetime.now(timezone.utc)
        await memory_store.append_turn(
            session_id=session_id,
            turn=ChatTurn(role="user", content=question, created_at=timestamp),
        )
        await memory_store.append_turn(
            session_id=session_id,
            turn=ChatTurn(role="assistant", content=answer, created_at=timestamp),
        )
        existing_profile = await memory_store.get_session_profile(session_id=session_id)
        session_title = str(existing_profile.get("session_title") or "").strip() or _build_session_title(question, language)
        summary_text, summary_turn_count, total_turn_count = await self._roll_history_summary(
            memory_store=memory_store,
            session_id=session_id,
            existing_profile=existing_profile,
            language=language,
        )
        await memory_store.set_session_profile(
            session_id=session_id,
            profile={
                "session_title": session_title,
                "preferred_language": language,
                "last_course_id": course_id,
                "memory_mode": memory_mode,
                "conversation_summary": summary_text,
                "history_summary_source_turns": summary_turn_count,
                "turn_count": total_turn_count,
                "last_prompt_token_estimate": int(prompt_plan.token_estimate if prompt_plan else 0),
                "last_context_compressed": bool(prompt_plan.compressed) if prompt_plan else False,
                "last_context_doc_count": len(prompt_plan.citation_docs) if prompt_plan else int(retrieval_stats.kept_doc_count if retrieval_stats else 0),
                "last_context_strategies": list(prompt_plan.strategies_used) if prompt_plan else [],
                "last_candidate_doc_count": int(retrieval_stats.candidate_doc_count if retrieval_stats else 0),
                "last_rerank_kept_count": int(retrieval_stats.kept_doc_count if retrieval_stats else 0),
                "last_rerank_filtered_count": int(retrieval_stats.filtered_doc_count if retrieval_stats else 0),
                "last_low_score_filtered": bool(retrieval_stats.low_score_filtered) if retrieval_stats else False,
                "last_citation_doc_ids": [str(item.doc_id) for item in (citations or []) if str(item.doc_id).strip()],
                "last_citation_file_names": [str(item.file_name) for item in (citations or []) if str(item.file_name).strip()],
                "last_selected_doc_ids": [str(item) for item in (selected_doc_ids or []) if str(item).strip()],
                "last_selected_doc_titles": [str(item) for item in (selected_doc_titles or []) if str(item).strip()],
            },
        )

    async def _prepare_history_context(
        self,
        memory_store: BaseMemoryStore,
        session_id: str,
        language: Literal["zh", "en"],
        selected_doc_ids: list[str] | None = None,
        selected_doc_titles: list[str] | None = None,
    ) -> HistoryContext:
        profile = await memory_store.get_session_profile(session_id=session_id)
        history_limit = max(self.config.recent_history_turns + 12, 24)
        turns = await memory_store.get_recent_turns(session_id=session_id, limit=history_limit)
        recent_turns = turns[-self.config.recent_history_turns :] if self.config.recent_history_turns > 0 else []
        history_summary = trim_text_to_token_limit(
            str(profile.get("conversation_summary", "")),
            self.config.history_summary_token_limit,
            self.config.chat_model,
        )
        return HistoryContext(
            recent_turns=recent_turns,
            history_summary=history_summary,
            focus_doc_ids=[str(item) for item in profile.get("last_citation_doc_ids", []) if str(item).strip()],
            focus_file_names=[str(item) for item in profile.get("last_citation_file_names", []) if str(item).strip()],
            selected_doc_ids=[
                str(item)
                for item in (selected_doc_ids or profile.get("last_selected_doc_ids", []))
                if str(item).strip()
            ],
            selected_doc_titles=[
                str(item)
                for item in (selected_doc_titles or profile.get("last_selected_doc_titles", []))
                if str(item).strip()
            ],
        )

    async def _roll_history_summary(
        self,
        memory_store: BaseMemoryStore,
        session_id: str,
        existing_profile: dict[str, Any],
        language: str,
    ) -> tuple[str, int, int]:
        keep_turns = max(0, self.config.recent_history_turns)
        existing_turn_count = int(existing_profile.get("turn_count", 0) or 0)
        summarized_turns = int(existing_profile.get("history_summary_source_turns", 0) or 0)
        target_turn_count = max(existing_turn_count + 2, 2)
        target_summary_turns = max(0, target_turn_count - keep_turns)
        new_summary_turns = max(0, target_summary_turns - summarized_turns)
        summary_limit = self.config.history_summary_token_limit
        summary_text = trim_text_to_token_limit(
            str(existing_profile.get("conversation_summary", "")),
            summary_limit,
            self.config.chat_model,
        )
        if new_summary_turns <= 0:
            return summary_text, summarized_turns, target_turn_count

        summary_window = await memory_store.get_recent_turns(
            session_id=session_id,
            limit=max(keep_turns + new_summary_turns, keep_turns + 2),
        )
        target_turn_count = max(existing_turn_count + 2, len(summary_window))
        target_summary_turns = max(0, target_turn_count - keep_turns)
        new_summary_turns = max(0, target_summary_turns - summarized_turns)
        if new_summary_turns <= 0:
            return summary_text, summarized_turns, target_turn_count
        if len(summary_window) > keep_turns:
            turns_to_summarize = summary_window[: len(summary_window) - keep_turns]
            if len(turns_to_summarize) > new_summary_turns:
                turns_to_summarize = turns_to_summarize[-new_summary_turns:]
            summary_text = _merge_history_summary(
                summary_text,
                turns_to_summarize,
                language,
                summary_limit,
                self.config.chat_model,
            )
            summarized_turns += len(turns_to_summarize)
            target_turn_count = max(target_turn_count, len(summary_window))
        return summary_text, summarized_turns, target_turn_count

    def _build_rewrite_prompt(
        self,
        question: str,
        history: list[ChatTurn],
        language: Literal["zh", "en"],
        instruction_override: str | None,
        history_summary: str,
        focus_file_names: list[str] | None = None,
        selected_document_titles: list[str] | None = None,
    ) -> str:
        """Build a rewrite prompt and shrink old history until it fits its budget."""

        budget = max(1024, self.config.model_context_window // 3)
        working_history = list(history)
        working_summary = history_summary
        prompt = build_query_rewrite_prompt(
            question=question,
            history=working_history,
            language=language,
            instruction_override=instruction_override,
            history_summary=working_summary,
            focus_sources=focus_file_names,
            selected_document_titles=selected_document_titles,
        )
        while estimate_token_count(prompt, self.config.chat_model) > budget:
            if working_history:
                working_history = working_history[1:]
            elif working_summary:
                working_summary = trim_text_to_token_limit(
                    working_summary,
                    max(80, estimate_token_count(working_summary, self.config.chat_model) - 80),
                    self.config.chat_model,
                )
            else:
                break
            prompt = build_query_rewrite_prompt(
                question=question,
                history=working_history,
                language=language,
                instruction_override=instruction_override,
                history_summary=working_summary,
                focus_sources=focus_file_names,
                selected_document_titles=selected_document_titles,
            )
        return prompt

    def _build_answer_prompt_plan(
        self,
        question: str,
        retrieved_docs: list[SourceDocument],
        language: Literal["zh", "en"],
        history_context: HistoryContext,
        qa_system_prompt_override: str | None,
        answer_instruction_override: str | None,
        degrade_level: int = 0,
        focus_sources: list[str] | None = None,
        selected_document_titles: list[str] | None = None,
    ) -> PromptPlan:
        """Assemble the final answer prompt under a strict context budget.

        Compression order is intentionally fixed so behavior is predictable:
        1. Build one prompt with original history and summary.
        2. Run compression gate:
           - <20k window: disable proactive history compression,
           - >=20k and >=90% budget: enable soft history compression,
           - >=20k and >=97% budget: enable stronger history compression.
        3. Drop oldest turns on retry rounds.
        4. Trim the rolling history summary.
        5. If still over budget, iteratively compress history (when enabled),
           drop low-priority
           turns, drop trailing chunks, trim the largest chunk, and finally trim
           the summary until the prompt fits or no more reductions are possible.
        """

        input_budget = max(1024, self.config.model_context_window - self.config.answer_token_reserve - (degrade_level * 1500))
        per_turn_limit = max(48, self.config.prompt_compression_turn_token_limit - (degrade_level * 20))
        working_history = list(history_context.recent_turns)
        if degrade_level > 0 and working_history:
            drop_count = min(len(working_history), degrade_level * 2)
            working_history = working_history[drop_count:]
        working_summary = trim_text_to_token_limit(
            history_context.history_summary,
            max(80, self.config.history_summary_token_limit - (degrade_level * 120)),
            self.config.chat_model,
        )
        active_focus_sources = list(focus_sources or history_context.focus_file_names)
        active_selected_titles = list(selected_document_titles or history_context.selected_doc_titles)
        selected_original_docs = list(retrieved_docs[: max(1, len(retrieved_docs) - degrade_level)])
        prompt_docs = [_clone_source_document(doc) for doc in selected_original_docs]
        prompt = build_rag_prompt(
            question=question,
            context_docs=prompt_docs,
            language=language,
            system_prompt_override=qa_system_prompt_override,
            answer_instruction_override=answer_instruction_override,
            recent_history=working_history,
            history_summary=working_summary,
            focus_sources=active_focus_sources,
            selected_document_titles=active_selected_titles,
        )
        token_estimate = estimate_token_count(prompt, self.config.chat_model)
        stage, enable_history_compression, aggressive_history_compression, soft_trigger, strong_trigger = _compression_gate(
            model_context_window=self.config.model_context_window,
            input_budget=input_budget,
            token_estimate=token_estimate,
        )
        strategies_used: list[str] = []
        if stage == "disabled_small_window":
            strategies_used.append("压缩门控:窗口<20k，禁用预压缩")
        elif stage == "watch_only":
            strategies_used.append(
                f"压缩门控:输入未达90%阈值({token_estimate}/{soft_trigger})"
            )
        elif stage == "near_limit":
            strategies_used.append(
                f"压缩门控:达到90%阈值，启用历史压缩({token_estimate}/{input_budget})"
            )
        else:
            strategies_used.append(
                f"压缩门控:达到97%阈值，启用强化压缩({token_estimate}/{input_budget})"
            )
        history_token_limit = per_turn_limit
        if aggressive_history_compression:
            history_token_limit = max(48, int(per_turn_limit * 0.8))
        history_compressed = False
        if enable_history_compression and working_history:
            working_history, history_compressed = _compress_history_turns(
                working_history,
                token_limit=history_token_limit,
                model_name=self.config.chat_model,
            )
            if history_compressed:
                prompt = build_rag_prompt(
                    question=question,
                    context_docs=prompt_docs,
                    language=language,
                    system_prompt_override=qa_system_prompt_override,
                    answer_instruction_override=answer_instruction_override,
                    recent_history=working_history,
                    history_summary=working_summary,
                    focus_sources=active_focus_sources,
                    selected_document_titles=active_selected_titles,
                )
                token_estimate = estimate_token_count(prompt, self.config.chat_model)
        if history_compressed:
            strategies_used.append(f"历史压缩:每轮限制{history_token_limit} tokens")
        compressed = token_estimate > input_budget or degrade_level > 0 or history_compressed
        guard = 0
        while token_estimate > input_budget and guard < 24:
            guard += 1
            if enable_history_compression and _compress_oldest_history_turn(
                working_history,
                per_turn_limit=max(24, history_token_limit - (guard * 8)),
                model_name=self.config.chat_model,
            ):
                if "历史压缩:渐进收缩" not in strategies_used:
                    strategies_used.append("历史压缩:渐进收缩")
            elif _drop_low_priority_history_turn(working_history):
                if "动态截断:删除低优先级历史" not in strategies_used:
                    strategies_used.append("动态截断:删除低优先级历史")
            elif len(prompt_docs) > 1:
                prompt_docs = prompt_docs[:-1]
                selected_original_docs = selected_original_docs[: len(prompt_docs)]
                if "动态截断:减少检索片段数" not in strategies_used:
                    strategies_used.append("动态截断:减少检索片段数")
            elif _trim_largest_document(prompt_docs, self.config.chat_model):
                if "提示词压缩:截短最大文档片段" not in strategies_used:
                    strategies_used.append("提示词压缩:截短最大文档片段")
            elif working_summary:
                working_summary = trim_text_to_token_limit(
                    working_summary,
                    max(60, estimate_token_count(working_summary, self.config.chat_model) - 120),
                    self.config.chat_model,
                )
                if "提示词压缩:收缩会话摘要" not in strategies_used:
                    strategies_used.append("提示词压缩:收缩会话摘要")
            else:
                break
            prompt = build_rag_prompt(
                question=question,
                context_docs=prompt_docs,
                language=language,
                system_prompt_override=qa_system_prompt_override,
                answer_instruction_override=answer_instruction_override,
                recent_history=working_history,
                history_summary=working_summary,
                focus_sources=active_focus_sources,
                selected_document_titles=active_selected_titles,
            )
            token_estimate = estimate_token_count(prompt, self.config.chat_model)
            compressed = True
        return PromptPlan(
            prompt=prompt,
            prompt_docs=prompt_docs,
            citation_docs=selected_original_docs,
            recent_turns=working_history,
            history_summary=working_summary,
            token_estimate=token_estimate,
            compressed=compressed,
            strategies_used=strategies_used,
            focus_sources=active_focus_sources,
            selected_document_titles=active_selected_titles,
        )


def _build_citations(documents: list[SourceDocument]) -> list[ChunkCitation]:
    citations: list[ChunkCitation] = []
    for index, document in enumerate(documents, start=1):
        metadata = document.metadata
        raw_breakdown = metadata.get("score_breakdown")
        breakdown = raw_breakdown if isinstance(raw_breakdown, dict) else {}
        citations.append(
            ChunkCitation(
                citation_id=index,
                doc_id=str(metadata.get("doc_id", "")),
                file_name=str(metadata.get("file_name", "")),
                file_path=str(metadata.get("file_path", "")),
                source_type=str(metadata.get("source_type", "")),
                page_label=str(metadata.get("page_label")) if metadata.get("page_label") else None,
                section_label=str(metadata.get("section_label")) if metadata.get("section_label") else None,
                locator_text=str(
                    metadata.get("page_label")
                    or metadata.get("section_label")
                    or metadata.get("section")
                    or ""
                ),
                chunk_id=str(metadata.get("chunk_id", "")),
                quote=document.page_content,
                matched_terms=[str(item) for item in metadata.get("matched_terms", []) if str(item).strip()],
                retrieval_score=float(metadata.get("retrieval_score")) if metadata.get("retrieval_score") is not None else None,
                score_breakdown={
                    str(key): float(value)
                    for key, value in breakdown.items()
                    if isinstance(value, (int, float))
                },
            )
        )
    return citations


def _no_evidence_message(language: Literal["zh", "en"]) -> str:
    if language == "zh":
        return "未在当前知识库中找到可引用依据。"
    return "No citable evidence was found in the current knowledge base."


def _build_session_title(question: str, language: str) -> str:
    normalized = " ".join(question.strip().split())
    if not normalized:
        return "新会话" if language == "zh" else "New chat"
    first_line = normalized.splitlines()[0]
    if language == "zh":
        return first_line[:18] + ("..." if len(first_line) > 18 else "")
    words = first_line.split()
    title = " ".join(words[:6])
    if len(words) > 6:
        title += "..."
    return title


def _resolve_follow_up_focus_doc_ids(
    *,
    question: str,
    language: Literal["zh", "en"],
    selected_doc_ids: list[str],
    history_context: HistoryContext,
) -> list[str]:
    """Prefer the previously cited document when the new question is a follow-up reference."""

    candidate_focus = [
        doc_id
        for doc_id in history_context.focus_doc_ids
        if not selected_doc_ids or doc_id in set(selected_doc_ids)
    ]
    if not candidate_focus:
        return []
    if _looks_like_follow_up_reference(question, language):
        return candidate_focus
    if len(candidate_focus) == 1 and len(question.strip()) <= 24:
        return candidate_focus
    return []


def _looks_like_follow_up_reference(question: str, language: Literal["zh", "en"]) -> bool:
    normalized = " ".join(question.strip().lower().split())
    if not normalized:
        return False
    zh_markers = ("它", "这篇", "该文", "该论文", "这项工作", "前者", "后者", "这篇论文", "这项研究")
    en_markers = (
        "it ",
        "it?",
        "it",
        "this paper",
        "that paper",
        "the paper",
        "the study",
        "former",
        "latter",
    )
    if language == "zh":
        return any(marker in question for marker in zh_markers)
    return any(marker in normalized for marker in en_markers)


def _is_collection_scope_question(question: str, selected_doc_ids: list[str]) -> bool:
    """Detect set-level questions such as 'which of these papers'."""

    if len(selected_doc_ids) < 2:
        return False
    normalized = " ".join(question.strip().lower().split())
    markers = (
        "哪些",
        "哪几篇",
        "这几篇",
        "这些论文",
        "这些文献",
        "which of these",
        "among these",
        "among the selected",
        "these papers",
        "selected papers",
    )
    return any(marker in question or marker in normalized for marker in markers)


def _ensure_selected_doc_coverage(
    *,
    candidate_docs: list[SourceDocument],
    kept_documents: list[SourceDocument],
    selected_doc_ids: list[str],
    result_limit: int,
) -> list[SourceDocument]:
    """Keep one strong chunk per selected document for small set-comparison questions."""

    if not candidate_docs or not selected_doc_ids:
        return kept_documents
    preferred_limit = max(1, min(result_limit, len(selected_doc_ids)))
    by_doc: dict[str, SourceDocument] = {}
    for document in candidate_docs:
        doc_id = str(document.metadata.get("doc_id", ""))
        if doc_id and doc_id in selected_doc_ids and doc_id not in by_doc:
            by_doc[doc_id] = document
    combined: list[SourceDocument] = []
    seen_doc_ids: set[str] = set()
    for document in kept_documents:
        doc_id = str(document.metadata.get("doc_id", ""))
        if doc_id and doc_id in seen_doc_ids:
            continue
        combined.append(document)
        if doc_id:
            seen_doc_ids.add(doc_id)
        if len(combined) >= preferred_limit:
            return combined
    for doc_id in selected_doc_ids:
        if doc_id in seen_doc_ids:
            continue
        document = by_doc.get(doc_id)
        if document is None:
            continue
        combined.append(document)
        seen_doc_ids.add(doc_id)
        if len(combined) >= preferred_limit:
            break
    return combined or kept_documents


def _merge_history_summary(
    existing_summary: str,
    turns: list[ChatTurn],
    language: str,
    token_limit: int,
    model_name: str,
) -> str:
    if not turns:
        return trim_text_to_token_limit(existing_summary, token_limit, model_name)
    role_map = {
        "user": "用户" if language == "zh" else "User",
        "assistant": "助手" if language == "zh" else "Assistant",
        "system": "系统" if language == "zh" else "System",
    }
    lines: list[str] = []
    if existing_summary.strip():
        lines.append(existing_summary.strip())
    for turn in turns:
        one_line = " ".join(turn.content.split())
        lines.append(f"{role_map.get(turn.role, turn.role)}: {one_line[:160]}")
    return trim_text_to_token_limit("\n".join(lines), token_limit, model_name)


def _clone_source_document(document: SourceDocument) -> SourceDocument:
    return SourceDocument(page_content=document.page_content, metadata=document.metadata.copy())


def _trim_largest_document(documents: list[SourceDocument], model_name: str) -> bool:
    if not documents:
        return False
    token_sizes = [estimate_token_count(doc.page_content, model_name) for doc in documents]
    largest_index = max(range(len(documents)), key=lambda idx: token_sizes[idx])
    largest_tokens = token_sizes[largest_index]
    if largest_tokens <= 120:
        return False
    new_limit = max(80, int(largest_tokens * 0.8))
    trimmed_text = trim_text_to_token_limit(documents[largest_index].page_content, new_limit, model_name)
    if not trimmed_text or trimmed_text == documents[largest_index].page_content:
        return False
    metadata = documents[largest_index].metadata.copy()
    documents[largest_index] = SourceDocument(page_content=trimmed_text, metadata=metadata)
    return True


def _compress_history_turns(
    turns: list[ChatTurn],
    token_limit: int,
    model_name: str,
) -> tuple[list[ChatTurn], bool]:
    compressed: list[ChatTurn] = []
    changed = False
    for turn in turns:
        content = compress_text_for_prompt(turn.content, token_limit, model_name)
        if content != turn.content:
            changed = True
        compressed.append(ChatTurn(role=turn.role, content=content, created_at=turn.created_at))
    return compressed, changed


def _compress_oldest_history_turn(
    turns: list[ChatTurn],
    per_turn_limit: int,
    model_name: str,
) -> bool:
    for index, turn in enumerate(turns):
        if estimate_token_count(turn.content, model_name) <= per_turn_limit:
            continue
        compressed = compress_text_for_prompt(turn.content, per_turn_limit, model_name)
        if compressed and compressed != turn.content:
            turns[index] = ChatTurn(role=turn.role, content=compressed, created_at=turn.created_at)
            return True
    return False


def _drop_low_priority_history_turn(turns: list[ChatTurn]) -> bool:
    if not turns:
        return False
    for index, turn in enumerate(turns):
        if turn.role == "assistant":
            del turns[index]
            return True
    turns.pop(0)
    return True


def _compression_gate(
    *,
    model_context_window: int,
    input_budget: int,
    token_estimate: int,
) -> tuple[str, bool, bool, int, int]:
    """Return compression stage + parameters for deterministic prompt behavior.

    Stages:
    - disabled_small_window: when model window < 20k, disable proactive history compression.
    - watch_only: window is large enough, but current prompt is still below the soft trigger.
    - near_limit: current prompt crossed 90% trigger, enable soft history compression.
    - critical_limit: current prompt crossed 97% trigger, enable stronger history compression.
    """

    soft_trigger = max(
        1024,
        min(input_budget, int(model_context_window * _PROMPT_COMPRESSION_TRIGGER_RATIO)),
    )
    strong_trigger = max(
        soft_trigger,
        min(input_budget, int(model_context_window * _PROMPT_COMPRESSION_STRONG_TRIGGER_RATIO)),
    )
    if model_context_window < _PROMPT_COMPRESSION_MIN_CONTEXT_WINDOW:
        return "disabled_small_window", False, False, soft_trigger, strong_trigger
    if token_estimate < soft_trigger:
        return "watch_only", False, False, soft_trigger, strong_trigger
    if token_estimate < strong_trigger:
        return "near_limit", True, False, soft_trigger, strong_trigger
    return "critical_limit", True, True, soft_trigger, strong_trigger


def _offline_answer(
    question: str,
    documents: list[SourceDocument],
    language: Literal["zh", "en"],
) -> str:
    snippets = []
    for document in documents[:2]:
        snippet = " ".join(document.page_content.split())
        snippets.append(snippet[:220])
    evidence = "\n".join(f"- {snippet}" for snippet in snippets if snippet)
    if language == "zh":
        return f"基于当前检索到的资料，和问题“{question}”最相关的内容如下：\n{evidence}"
    return f'Based on the retrieved materials, the most relevant evidence for "{question}" is:\n{evidence}'


def _chunk_text(text: str, chunk_size: int = 40) -> list[str]:
    return [text[index : index + chunk_size] for index in range(0, len(text), chunk_size)]


def _resolve_prompt_override(
    override: str | dict[str, str] | None,
    language: Literal["zh", "en"],
) -> str | None:
    if override is None:
        return None
    if isinstance(override, dict):
        value = str(override.get(language, "")).strip()
        return value or None
    value = str(override).strip()
    return value or None


def default_prompt_bundle(language: Literal["zh", "en"]) -> dict[str, str]:
    if language == "zh":
        return {
            "qa_system_prompt": DEFAULT_QA_SYSTEM_PROMPT_ZH,
            "rewrite_instruction": DEFAULT_QUERY_REWRITE_INSTRUCTION_ZH,
            "answer_instruction": DEFAULT_QA_ANSWER_INSTRUCTION_ZH,
        }
        return {
            "qa_system_prompt": DEFAULT_QA_SYSTEM_PROMPT_EN,
            "rewrite_instruction": DEFAULT_QUERY_REWRITE_INSTRUCTION_EN,
            "answer_instruction": DEFAULT_QA_ANSWER_INSTRUCTION_EN,
        }


def _count_chroma_embeddings(vector_dir: Path) -> int:
    sqlite_path = vector_dir / "chroma.sqlite3"
    if not sqlite_path.exists():
        return 0
    try:
        connection = sqlite3.connect(str(sqlite_path))
    except Exception:
        return 0
    try:
        cursor = connection.cursor()
        cursor.execute("select count(*) from embeddings")
        row = cursor.fetchone()
    except Exception:
        return 0
    finally:
        connection.close()
    if not row:
        return 0
    return int(row[0] or 0)


def _next_backup_dir(target_dir: Path) -> Path:
    candidate = target_dir.with_name(f"{target_dir.name}_pre_legacy_backup")
    index = 2
    while candidate.exists():
        candidate = target_dir.with_name(f"{target_dir.name}_pre_legacy_backup_{index}")
        index += 1
    return candidate


async def _emit_vector_progress(callback: Any | None, message: str) -> None:
    if callback is None:
        return
    result = callback(message)
    if asyncio.iscoroutine(result):
        await result


def _append_vector_log(config: AppConfig, payload: dict[str, Any]) -> None:
    path = Path(config.vector_operation_log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_payload = {"timestamp": datetime.now(timezone.utc).isoformat(), **payload}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(safe_payload, ensure_ascii=False, default=str) + "\n")


def _is_test_runtime() -> bool:
    if os.getenv("PYTEST_CURRENT_TEST"):
        return True
    if "pytest" in sys.modules:
        return True
    if "unittest" in sys.modules and "ipykernel" not in sys.modules:
        return True
    return False
