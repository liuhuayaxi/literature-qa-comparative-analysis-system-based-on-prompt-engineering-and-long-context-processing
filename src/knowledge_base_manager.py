from __future__ import annotations

import asyncio
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable

from .config import AppConfig
from .knowledge_ingestion import DocumentIndexer, build_chunk_config_signature
from .retrieval_engine import VectorStoreService
from .models import DocumentRecord, SourceDocument
from .app_utils import (
    JsonResultCache,
    build_course_signature,
    build_file_signature,
    detect_language,
    load_json_mapping,
    save_json_mapping,
)


ProgressCallback = Callable[[str], Awaitable[None] | None]


class KnowledgeBaseManager:
    """Coordinate file-level operations on one knowledge base and its vector index."""

    def __init__(
        self,
        config: AppConfig,
        vector_store: VectorStoreService,
        indexer: DocumentIndexer,
    ) -> None:
        self.config = config
        self.vector_store = vector_store
        self.indexer = indexer
        self.result_cache = JsonResultCache(config.cache_dir)

    async def list_knowledge_bases(self) -> list[str]:
        """Return all knowledge base names found either on disk or in the vector store."""

        await self._repair_legacy_vector_aliases()
        return sorted(set(_scan_data_course_ids(self.config.data_root)) | set(await self.vector_store.list_course_ids()))

    async def create_knowledge_base(self, course_id: str) -> str:
        """Create the directory layout for a new knowledge base."""

        normalized = course_id.strip()
        if not normalized:
            raise ValueError("Knowledge base name is required.")
        root = self.config.data_root / normalized
        for _, folder_name in _source_type_pairs():
            (root / folder_name).mkdir(parents=True, exist_ok=True)
        await self.record_knowledge_base_state(normalized)
        return normalized

    async def list_documents(self, course_id: str) -> list[DocumentRecord]:
        """List vectorized documents known to the retrieval layer."""

        await self._repair_course_vectors(course_id)
        records = await self.vector_store.list_documents(course_id)
        return [_normalize_record_file_path(record, self.config) for record in records]

    async def list_manageable_files(self, course_id: str) -> list[DocumentRecord]:
        """List both vectorized and not-yet-vectorized files under one knowledge base."""

        await self._repair_course_vectors(course_id)
        vectorized_records = await self.vector_store.list_documents(course_id)
        by_path = {
            str(Path(_normalize_record_file_path(record, self.config).file_path).resolve(strict=False)): _normalize_record_file_path(record, self.config)
            for record in vectorized_records
        }
        records: list[DocumentRecord] = []
        seen_paths: set[str] = set()
        base_dir = self.config.data_root / course_id
        for source_type, folder_name in _source_type_pairs():
            folder = base_dir / folder_name
            if not folder.exists():
                continue
            for path in sorted(item for item in folder.iterdir() if item.is_file()):
                normalized_path = str(path.resolve(strict=False))
                record = by_path.get(normalized_path)
                if record is None:
                    record = _build_unvectorized_record(course_id=course_id, file_path=path, source_type=source_type)
                records.append(record)
                seen_paths.add(normalized_path)
        for normalized_path, record in by_path.items():
            if normalized_path not in seen_paths:
                records.append(record)
        return sorted(records, key=lambda item: (item.source_type, item.file_name.lower()))

    async def add_files(
        self,
        course_id: str,
        file_paths: list[Path],
        source_type: str,
        rebuild_index: bool = False,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        merge_small_chunks: bool = False,
        min_chunk_size: int | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, int]:
        """Add one batch of files into the knowledge base and build their index chunks."""

        result = await self.indexer.ingest_paths(
            course_id=course_id,
            file_paths=file_paths,
            source_type=source_type,
            rebuild_index=rebuild_index,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            merge_small_chunks=merge_small_chunks,
            min_chunk_size=min_chunk_size,
            progress_callback=progress_callback,
        )
        await self.record_knowledge_base_state(course_id)
        self.invalidate_course_caches(course_id)
        return result

    async def delete_files(self, course_id: str, doc_ids: list[str]) -> dict[str, int]:
        """Delete indexed documents by document id and remove their raw files if present."""

        records = await self.list_documents(course_id)
        to_delete = [record for record in records if record.doc_id in doc_ids]
        deleted_files = 0
        for record in to_delete:
            path = record.path
            if path.exists():
                path.unlink()
                deleted_files += 1
        deleted_docs = await self.vector_store.delete_documents(course_id=course_id, doc_ids=doc_ids)
        await self.record_knowledge_base_state(course_id)
        self.invalidate_course_caches(course_id)
        return {"deleted_files": deleted_files, "deleted_docs": deleted_docs}

    async def delete_file_paths(self, course_id: str, file_paths: list[str]) -> dict[str, int]:
        """Delete files by absolute path and remove their vector documents too."""

        managed_files = await self.list_manageable_files(course_id)
        records_by_path = {
            str(Path(record.file_path).resolve(strict=False)): record
            for record in managed_files
        }
        deleted_files = 0
        doc_ids_to_delete: list[str] = []
        for raw_path in file_paths:
            normalized = str(Path(raw_path).resolve(strict=False))
            record = records_by_path.get(normalized)
            if not record:
                continue
            path = Path(normalized)
            if path.exists():
                path.unlink()
                deleted_files += 1
            if record.doc_id:
                doc_ids_to_delete.append(record.doc_id)
        deleted_docs = 0
        if doc_ids_to_delete:
            deleted_docs = await self.vector_store.delete_documents(course_id=course_id, doc_ids=doc_ids_to_delete)
        await self.record_knowledge_base_state(course_id)
        self.invalidate_course_caches(course_id)
        return {"deleted_files": deleted_files, "deleted_docs": deleted_docs}

    async def rename_file(self, course_id: str, doc_id: str, new_file_name: str) -> DocumentRecord:
        """Rename one indexed file and sync its metadata in the vector store."""

        record = await self.get_document(course_id, doc_id)
        target_name = _normalize_file_name(record, new_file_name)
        source_path = record.path
        target_path = source_path.with_name(target_name)
        if target_path.exists() and target_path != source_path:
            raise FileExistsError(f"Target file already exists: {target_path.name}")
        if source_path.exists() and target_path != source_path:
            source_path.rename(target_path)
        try:
            updated = await self.vector_store.update_document_metadata(
                course_id=course_id,
                doc_id=doc_id,
                updates={
                    "file_name": target_path.name,
                    "file_path": str(target_path.resolve()),
                },
            )
        except Exception:
            if target_path.exists() and target_path != source_path:
                target_path.rename(source_path)
            raise
        await self.record_knowledge_base_state(course_id)
        self.invalidate_course_caches(course_id)
        return updated

    async def rename_file_by_path(self, course_id: str, file_path: str, new_file_name: str) -> DocumentRecord:
        """Rename one file selected from the file list, even if not yet vectorized."""

        record = await self.get_manageable_file(course_id, file_path)
        target_name = _normalize_file_name(record, new_file_name)
        source_path = Path(file_path).resolve(strict=False)
        target_path = source_path.with_name(target_name)
        if target_path.exists() and target_path != source_path:
            raise FileExistsError(f"Target file already exists: {target_path.name}")
        if source_path.exists() and target_path != source_path:
            source_path.rename(target_path)
        if record.doc_id:
            try:
                updated = await self.vector_store.update_document_metadata(
                    course_id=course_id,
                    doc_id=record.doc_id,
                    updates={
                        "file_name": target_path.name,
                        "file_path": str(target_path.resolve(strict=False)),
                    },
                )
            except Exception:
                if target_path.exists() and target_path != source_path:
                    target_path.rename(source_path)
                raise
            await self.record_knowledge_base_state(course_id)
            self.invalidate_course_caches(course_id)
            return updated
        updated = _build_unvectorized_record(course_id=course_id, file_path=target_path, source_type=record.source_type)
        await self.record_knowledge_base_state(course_id)
        self.invalidate_course_caches(course_id)
        return updated

    async def rename_knowledge_base(self, old_course_id: str, new_course_id: str) -> dict[str, int]:
        """Rename a knowledge base on disk and in the vector metadata."""

        old_path = self.config.data_root / old_course_id
        new_path = self.config.data_root / new_course_id
        if not old_course_id:
            raise ValueError("Source knowledge base name is required.")
        if not new_course_id:
            raise ValueError("Target knowledge base name is required.")
        if new_path.exists():
            raise FileExistsError(f"Knowledge base already exists: {new_course_id}")
        if old_path.exists():
            old_path.rename(new_path)
        try:
            updated_chunks = await self.vector_store.rename_course(old_course_id=old_course_id, new_course_id=new_course_id)
        except Exception:
            if new_path.exists() and not old_path.exists():
                new_path.rename(old_path)
            raise
        self.invalidate_course_caches(old_course_id)
        self.invalidate_course_caches(new_course_id)
        self.clear_knowledge_base_state(old_course_id)
        await self.record_knowledge_base_state(new_course_id)
        return {"updated_chunks": updated_chunks}

    async def delete_knowledge_base(self, course_id: str) -> dict[str, int]:
        """Delete one knowledge base without touching other knowledge bases or sessions."""

        normalized = course_id.strip()
        if not normalized:
            raise ValueError("Knowledge base name is required.")
        base_dir = self.config.data_root / normalized
        file_count = 0
        if base_dir.exists():
            file_count = sum(1 for path in base_dir.rglob("*") if path.is_file())
            shutil.rmtree(base_dir)
        records = await self.list_documents(normalized)
        vector_doc_ids = [record.doc_id for record in records if record.doc_id]
        deleted_vector_docs = 0
        if vector_doc_ids:
            deleted_vector_docs = await self.vector_store.delete_documents(
                course_id=normalized,
                doc_ids=vector_doc_ids,
            )
        else:
            await self.vector_store.reset_course(normalized)
        self.invalidate_course_caches(normalized)
        self.clear_knowledge_base_state(normalized)
        return {
            "deleted_files": file_count,
            "deleted_vector_docs": deleted_vector_docs,
        }

    async def get_document(self, course_id: str, doc_id: str) -> DocumentRecord:
        """Return one indexed document by id."""

        records = await self.list_documents(course_id)
        for record in records:
            if record.doc_id == doc_id:
                return record
        raise ValueError(f"Document not found: {doc_id}")

    async def get_manageable_file(self, course_id: str, file_path: str) -> DocumentRecord:
        """Return one file record from the management view by its absolute path."""

        normalized = str(Path(file_path).resolve(strict=False))
        records = await self.list_manageable_files(course_id)
        for record in records:
            if str(Path(record.file_path).resolve(strict=False)) == normalized:
                return record
        raise ValueError(f"File not found: {file_path}")

    async def get_chunk_details(self, course_id: str, file_path: str) -> list[dict[str, object]]:
        """Return chunk-level details used in the knowledge-base detail panel."""

        record = await self.get_manageable_file(course_id, file_path)
        if not record.doc_id:
            return []
        chunks = await self.vector_store.get_document_chunks(course_id=course_id, doc_id=record.doc_id)
        details: list[dict[str, object]] = []
        for index, chunk in enumerate(chunks, start=1):
            metadata = chunk.metadata
            details.append(
                {
                    "chunk_index": int(metadata.get("chunk_index", index)),
                    "chunk_id": str(metadata.get("chunk_id", "")),
                    "page_label": metadata.get("page_label"),
                    "section_label": metadata.get("section_label"),
                    "merged_from_count": int(metadata.get("merged_from_count", 1)),
                    "length": len(chunk.page_content),
                    "content": chunk.page_content,
                }
            )
        return details

    async def vectorize_files(
        self,
        course_id: str,
        file_paths: list[str],
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        merge_small_chunks: bool = False,
        min_chunk_size: int | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, int]:
        """Re-vectorize the selected raw files with the current chunk settings."""

        managed_files = await self.list_manageable_files(course_id)
        records_by_path = {
            str(Path(record.file_path).resolve(strict=False)): record
            for record in managed_files
        }
        resolved_chunk_size = int(chunk_size or self.config.chunk_size)
        resolved_chunk_overlap = int(chunk_overlap or self.config.chunk_overlap)
        resolved_merge_small_chunks = bool(merge_small_chunks)
        current_chunk_config_signature = build_chunk_config_signature(
            chunk_size=resolved_chunk_size,
            chunk_overlap=resolved_chunk_overlap,
            merge_small_chunks=resolved_merge_small_chunks,
            min_chunk_size=min_chunk_size,
        )
        grouped_paths: dict[str, list[Path]] = {source_type: [] for source_type, _ in _source_type_pairs()}
        delete_doc_ids: list[str] = []
        skipped_files = 0
        selected_entries: list[tuple[DocumentRecord, Path]] = []
        for raw_path in file_paths:
            normalized = str(Path(raw_path).resolve(strict=False))
            record = records_by_path.get(normalized)
            if not record:
                continue
            selected_entries.append((record, Path(normalized)))

        async def inspect_selected_file(record: DocumentRecord, normalized_path: Path) -> tuple[DocumentRecord, Path, bool]:
            if not record.doc_id:
                return record, normalized_path, False
            chunks = await self.vector_store.get_document_chunks(course_id=course_id, doc_id=record.doc_id)
            if not chunks:
                return record, normalized_path, False
            first_metadata = chunks[0].metadata
            return (
                record,
                normalized_path,
                str(first_metadata.get("file_signature", "")) == build_file_signature(normalized_path)
                and str(first_metadata.get("chunk_config_signature", "")) == current_chunk_config_signature,
            )

        inspection_results = await asyncio.gather(
            *(inspect_selected_file(record, normalized_path) for record, normalized_path in selected_entries)
        )
        for record, normalized_path, can_skip in inspection_results:
            if can_skip:
                skipped_files += 1
                await _emit_progress(
                    progress_callback,
                    f"跳过未变化文件：{normalized_path.name}（文件内容和切片参数都未变化）",
                )
                continue
            grouped_paths[record.source_type].append(normalized_path)
            if record.doc_id:
                delete_doc_ids.append(record.doc_id)
        if delete_doc_ids:
            await self.vector_store.delete_documents(course_id=course_id, doc_ids=delete_doc_ids)
        total_raw = 0
        total_chunks = 0
        total_stored = 0
        for source_type, paths in grouped_paths.items():
            if not paths:
                continue
            await _emit_progress(
                progress_callback,
                f"开始处理类型“{_translate_source_type(source_type)}”，共 {len(paths)} 个文件。",
            )
            result = await self.indexer.ingest_paths(
                course_id=course_id,
                file_paths=paths,
                source_type=source_type,
                rebuild_index=False,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                merge_small_chunks=merge_small_chunks,
                min_chunk_size=min_chunk_size,
                progress_callback=progress_callback,
            )
            total_raw += result["raw_documents"]
            total_chunks += result["chunks"]
            total_stored += result["stored"]
        vectorized_files = sum(len(paths) for paths in grouped_paths.values())
        if vectorized_files:
            await self.record_knowledge_base_state(course_id)
            self.invalidate_course_caches(course_id)
        elif skipped_files:
            await _emit_progress(progress_callback, "所选文件都已是最新向量，无需重复向量化。")
        return {
            "raw_documents": total_raw,
            "chunks": total_chunks,
            "stored": total_stored,
            "vectorized_files": vectorized_files,
            "skipped_files": skipped_files,
        }

    async def move_file_paths(
        self,
        source_course_id: str,
        file_paths: list[str],
        target_course_id: str,
        target_source_type: str,
    ) -> dict[str, int]:
        """Move selected files to another knowledge base and roll back on failure."""

        if not source_course_id.strip() or not target_course_id.strip():
            raise ValueError("Source and target knowledge base names are required.")
        if source_course_id == target_course_id and not target_source_type.strip():
            raise ValueError("Please choose a different target knowledge base or target file type.")
        await self.create_knowledge_base(target_course_id)
        managed_files = await self.list_manageable_files(source_course_id)
        records_by_path = {
            str(Path(record.file_path).resolve(strict=False)): record
            for record in managed_files
        }
        moved_entries: list[tuple[Path, Path, DocumentRecord]] = []
        updated_doc_ids: list[DocumentRecord] = []
        target_dir = self.config.data_root / target_course_id / f"{target_source_type}s"
        target_dir.mkdir(parents=True, exist_ok=True)
        try:
            for raw_path in file_paths:
                normalized = str(Path(raw_path).resolve(strict=False))
                record = records_by_path.get(normalized)
                if record is None:
                    continue
                source_path = Path(normalized)
                destination = target_dir / source_path.name
                if destination.exists() and destination != source_path:
                    raise FileExistsError(f"Target file already exists: {destination.name}")
                if source_path.exists() and destination != source_path:
                    source_path.rename(destination)
                moved_entries.append((source_path, destination, record))
                if record.doc_id:
                    await self.vector_store.update_document_metadata(
                        course_id=source_course_id,
                        doc_id=record.doc_id,
                        updates={
                            "course_id": target_course_id,
                            "source_type": target_source_type,
                            "file_name": destination.name,
                            "file_path": str(destination.resolve(strict=False)),
                        },
                    )
                    updated_doc_ids.append(record)
        except Exception:
            for original_path, destination_path, record in reversed(moved_entries):
                if destination_path.exists() and destination_path != original_path:
                    original_path.parent.mkdir(parents=True, exist_ok=True)
                    destination_path.rename(original_path)
            for record in reversed(updated_doc_ids):
                await self.vector_store.update_document_metadata(
                    course_id=target_course_id,
                    doc_id=record.doc_id,
                    updates={
                        "course_id": source_course_id,
                        "source_type": record.source_type,
                        "file_name": Path(record.file_path).name,
                        "file_path": record.file_path,
                    },
                )
            raise
        await self.record_knowledge_base_state(source_course_id)
        await self.record_knowledge_base_state(target_course_id)
        self.invalidate_course_caches(source_course_id)
        self.invalidate_course_caches(target_course_id)
        return {
            "moved_files": len(moved_entries),
            "updated_vector_docs": len(updated_doc_ids),
        }

    async def rename_session_title(
        self,
        memory_store,
        session_id: str,
        title: str,
    ) -> None:
        profile = await memory_store.get_session_profile(session_id)
        await memory_store.set_session_profile(
            session_id,
            {
                **profile,
                "session_title": title.strip() or profile.get("session_title") or "新会话",
            },
        )

    async def detect_knowledge_base_changes(self, course_id: str) -> dict[str, object]:
        state_map = load_json_mapping(self.config.knowledge_base_state_path)
        saved = state_map.get(course_id, {}) if isinstance(state_map.get(course_id), dict) else {}
        current_signature = build_course_signature(course_id, root_dir=self.config.data_root)
        last_signature = str(saved.get("course_signature", ""))
        changed = current_signature != last_signature
        return {
            "changed": changed,
            "course_signature": current_signature,
            "last_signature": last_signature,
            "last_recorded_at": str(saved.get("recorded_at", "")),
        }

    async def record_knowledge_base_state(self, course_id: str) -> None:
        state_map = load_json_mapping(self.config.knowledge_base_state_path)
        state_map[course_id] = {
            "course_signature": build_course_signature(course_id, root_dir=self.config.data_root),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        save_json_mapping(self.config.knowledge_base_state_path, state_map)

    def clear_knowledge_base_state(self, course_id: str) -> None:
        state_map = load_json_mapping(self.config.knowledge_base_state_path)
        if course_id in state_map:
            state_map.pop(course_id, None)
            save_json_mapping(self.config.knowledge_base_state_path, state_map)

    def invalidate_course_caches(self, course_id: str) -> None:
        for namespace in (
            f"analysis_{course_id}",
            f"extraction_{course_id}",
            f"report_{course_id}",
            f"rewrite_{course_id}",
        ):
            self.result_cache.delete_namespace(namespace)

    async def _repair_legacy_vector_aliases(self) -> None:
        """Reattach vector-only legacy course ids back to raw-file knowledge bases.

        This repairs the partial rename case where raw files were moved to a new
        knowledge-base name but the vector metadata stayed under the old name.
        """

        raw_course_ids = _scan_data_course_ids(self.config.data_root)
        vector_course_ids = await self.vector_store.list_course_ids()
        vector_only_ids = [course_id for course_id in vector_course_ids if course_id not in raw_course_ids]
        if not vector_only_ids:
            return
        for course_id in raw_course_ids:
            await self._repair_course_vectors(course_id, candidate_ids=vector_only_ids)

    async def _repair_course_vectors(
        self,
        course_id: str,
        candidate_ids: list[str] | None = None,
    ) -> int:
        """Repair missing vectors for one course by matching a legacy vector-only alias."""

        normalized = course_id.strip()
        if not normalized:
            return 0
        existing_records = await self.vector_store.list_documents(normalized)
        if existing_records:
            return 0
        current_keys = _scan_course_file_keys(self.config.data_root / normalized)
        if not current_keys:
            return 0

        candidates = candidate_ids or await self.vector_store.list_course_ids()
        best_candidate = ""
        best_overlap = 0
        best_current_ratio = 0.0
        best_candidate_ratio = 0.0

        for candidate in candidates:
            if candidate == normalized:
                continue
            candidate_dir = self.config.data_root / candidate
            if _directory_has_files(candidate_dir):
                continue
            candidate_records = await self.vector_store.list_documents(candidate)
            if not candidate_records:
                continue
            candidate_keys = _record_identity_keys(candidate_records)
            overlap = len(current_keys & candidate_keys)
            if overlap == 0:
                continue
            current_ratio = overlap / max(len(current_keys), 1)
            candidate_ratio = overlap / max(len(candidate_keys), 1)
            if current_ratio < 0.8 or candidate_ratio < 0.8:
                continue
            if (
                overlap > best_overlap
                or (overlap == best_overlap and candidate_ratio > best_candidate_ratio)
                or (
                    overlap == best_overlap
                    and candidate_ratio == best_candidate_ratio
                    and current_ratio > best_current_ratio
                )
            ):
                best_candidate = candidate
                best_overlap = overlap
                best_current_ratio = current_ratio
                best_candidate_ratio = candidate_ratio

        if not best_candidate:
            return 0

        updated_chunks = await self.vector_store.rename_course(best_candidate, normalized)
        if updated_chunks > 0:
            self.invalidate_course_caches(best_candidate)
            self.invalidate_course_caches(normalized)
            self.clear_knowledge_base_state(best_candidate)
            await self.record_knowledge_base_state(normalized)
        return updated_chunks


def _normalize_file_name(record: DocumentRecord, new_file_name: str) -> str:
    candidate = new_file_name.strip()
    if not candidate:
        raise ValueError("New file name is required.")
    suffix = record.path.suffix
    if Path(candidate).suffix.lower() != suffix.lower():
        candidate = f"{candidate}{suffix}"
    return candidate


def _source_type_pairs() -> list[tuple[str, str]]:
    return [
        ("lecture", "lectures"),
        ("assignment", "assignments"),
        ("paper", "papers"),
    ]


def _scan_data_course_ids(root: Path) -> list[str]:
    if not root.exists():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def _directory_has_files(path: Path) -> bool:
    if not path.exists():
        return False
    return any(item.is_file() for item in path.rglob("*"))


def _scan_course_file_keys(base_dir: Path) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for source_type, folder_name in _source_type_pairs():
        folder = base_dir / folder_name
        if not folder.exists():
            continue
        for path in folder.iterdir():
            if path.is_file():
                keys.add((source_type, path.name))
    return keys


def _record_identity_keys(records: list[DocumentRecord]) -> set[tuple[str, str]]:
    return {
        (record.source_type, record.file_name)
        for record in records
        if record.file_name
    }


def _build_unvectorized_record(course_id: str, file_path: Path, source_type: str) -> DocumentRecord:
    file_ext = file_path.suffix.lower().lstrip(".")
    if file_ext not in {"pdf", "md", "txt", "docx"}:
        file_ext = "txt"
    return DocumentRecord(
        doc_id="",
        course_id=course_id,
        source_type=source_type,
        file_name=file_path.name,
        file_path=str(file_path.resolve(strict=False)),
        file_ext=file_ext,
        language=detect_language(file_path.stem),
        chunk_count=0,
        is_vectorized=False,
    )


async def _emit_progress(callback: ProgressCallback | None, message: str) -> None:
    """Forward knowledge-base operation progress to an optional notebook callback."""

    if callback is None:
        return
    result = callback(message)
    if hasattr(result, "__await__"):
        await result


def _translate_source_type(value: str) -> str:
    return {
        "lecture": "讲义",
        "assignment": "作业",
        "paper": "论文",
    }.get(value, value)


def _normalize_record_file_path(record: DocumentRecord, config: AppConfig) -> DocumentRecord:
    legacy_root = (config.project_root / "notebooks" / "data" / "raw").resolve(strict=False)
    current_root = config.data_root.resolve(strict=False)
    original_path = Path(record.file_path).resolve(strict=False)
    try:
        relative = original_path.relative_to(legacy_root)
    except ValueError:
        return record
    canonical_path = current_root / relative
    return record.model_copy(update={"file_path": str(canonical_path.resolve(strict=False))})
