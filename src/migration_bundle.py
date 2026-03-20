"""Export and import migration bundles for project-scoped runtime data."""

from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
import errno
from pathlib import Path, PurePosixPath
from typing import Any

from .config import AppConfig

BUNDLE_SCHEMA_VERSION = 1
BUNDLE_MANIFEST_NAME = "migration_bundle_manifest.json"
_PAYLOAD_PREFIX = "payload"
_EXPORT_EXCLUDED_SUFFIXES = {".pdf"}


def collect_migration_roots(config: AppConfig) -> list[Path]:
    """Return the minimal set of project-relative roots required for migration."""

    project_root = config.project_root.resolve(strict=False)
    candidates = [
        config.config_path.parent,
        config.db_path.parent,
        config.vector_dir,
        config.cache_dir,
        config.analysis_checkpoint_dir,
        config.field_template_path.parent,
        config.knowledge_base_state_path.parent,
        config.data_root,
        config.reports_dir,
    ]
    relative_candidates: list[Path] = []
    outside_paths: list[str] = []
    for candidate in candidates:
        resolved = Path(candidate).resolve(strict=False)
        try:
            relative = resolved.relative_to(project_root)
        except ValueError:
            outside_paths.append(str(resolved))
            continue
        if not relative.parts:
            raise ValueError("迁移包不能直接覆盖整个项目根目录。")
        relative_candidates.append(relative)
    if outside_paths:
        joined = ", ".join(sorted(outside_paths))
        raise ValueError(f"迁移包暂不支持项目目录之外的数据路径: {joined}")

    unique_candidates = sorted(
        {candidate for candidate in relative_candidates},
        key=lambda item: (len(item.parts), item.as_posix()),
    )
    collapsed: list[Path] = []
    for candidate in unique_candidates:
        if any(_path_contains(existing, candidate) for existing in collapsed):
            continue
        collapsed.append(candidate)
    return collapsed


def export_migration_bundle(
    config: AppConfig,
    bundle_path: str | Path | None = None,
    *,
    bundle_name: str = "",
) -> dict[str, Any]:
    """Create a zip bundle containing config and runtime data, excluding logs."""

    project_root = config.project_root.resolve(strict=False)
    roots = collect_migration_roots(config)
    target_bundle = _resolve_bundle_output_path(project_root, bundle_path=bundle_path, bundle_name=bundle_name)
    entries: list[dict[str, Any]] = []
    total_bytes = 0

    with zipfile.ZipFile(target_bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for root in roots:
            root_relative = root.as_posix().rstrip("/")
            archive.writestr(f"{_PAYLOAD_PREFIX}/{root_relative}/", "")
            root_path = project_root / root
            if not root_path.exists():
                continue
            for file_path in _iter_root_files(root_path):
                relative_path = file_path.resolve(strict=False).relative_to(project_root)
                if _should_skip_export_bundle_file(relative_path):
                    continue
                archive_name = f"{_PAYLOAD_PREFIX}/{relative_path.as_posix()}"
                archive.write(file_path, archive_name)
                size = file_path.stat().st_size
                entries.append(
                    {
                        "relative_path": relative_path.as_posix(),
                        "size_bytes": size,
                    }
                )
                total_bytes += size
        manifest = {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "project_root": ".",
            "roots": [root.as_posix() for root in roots],
            "files": entries,
            "file_count": len(entries),
            "total_size_bytes": total_bytes,
            "excluded_categories": ["logs", "pdf"],
        }
        archive.writestr(BUNDLE_MANIFEST_NAME, json.dumps(manifest, ensure_ascii=False, indent=2))

    return {
        "bundle_path": str(target_bundle),
        "roots": manifest["roots"],
        "file_count": len(entries),
        "total_size_bytes": total_bytes,
    }


def inspect_migration_bundle(bundle_path: str | Path) -> dict[str, Any]:
    """Read and validate the manifest from one migration bundle zip file."""

    bundle = Path(bundle_path)
    if not bundle.exists():
        raise FileNotFoundError(f"迁移包不存在: {bundle}")
    with zipfile.ZipFile(bundle, "r") as archive:
        try:
            raw_manifest = archive.read(BUNDLE_MANIFEST_NAME)
        except KeyError as exc:
            raise ValueError("迁移包缺少清单文件，无法导入。") from exc
    try:
        manifest = json.loads(raw_manifest.decode("utf-8"))
    except Exception as exc:
        raise ValueError("迁移包清单不是有效的 JSON。") from exc
    if not isinstance(manifest, dict):
        raise ValueError("迁移包清单格式不正确。")
    if int(manifest.get("schema_version", 0) or 0) != BUNDLE_SCHEMA_VERSION:
        raise ValueError("迁移包版本不兼容。")
    roots = manifest.get("roots")
    files = manifest.get("files")
    if not isinstance(roots, list) or not isinstance(files, list):
        raise ValueError("迁移包清单缺少 roots/files 信息。")
    for root in roots:
        _validate_relative_path(str(root))
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("迁移包文件条目格式不正确。")
        _validate_relative_path(str(item.get("relative_path", "")))
    return manifest


def import_migration_bundle(
    config: AppConfig,
    bundle_path: str | Path,
    *,
    create_backup: bool = True,
    backup_path: str | Path | None = None,
) -> dict[str, Any]:
    """Restore config and runtime data from one migration bundle zip file."""

    project_root = config.project_root.resolve(strict=False)
    source_bundle = Path(bundle_path)
    if not source_bundle.is_absolute():
        source_bundle = (project_root / source_bundle).resolve(strict=False)
    if not source_bundle.exists():
        raise FileNotFoundError(f"迁移包不存在: {source_bundle}")

    manifest = inspect_migration_bundle(source_bundle)
    with tempfile.TemporaryDirectory(prefix="migration_bundle_") as tmpdir:
        safe_bundle = Path(tmpdir) / source_bundle.name
        shutil.copy2(source_bundle, safe_bundle)
        backup_info: dict[str, Any] | None = None
        if create_backup:
            resolved_backup_path = (
                Path(backup_path)
                if backup_path is not None
                else project_root / "release" / "migration_backups" / _default_backup_name()
            )
            backup_info = export_migration_bundle(
                config,
                bundle_path=resolved_backup_path,
                bundle_name="",
            )
        _clear_existing_roots(project_root, [Path(root) for root in manifest["roots"]])
        _extract_payload(project_root, safe_bundle)
        for root in manifest["roots"]:
            (project_root / Path(root)).mkdir(parents=True, exist_ok=True)

    return {
        "bundle_path": str(source_bundle),
        "roots": list(manifest["roots"]),
        "file_count": int(manifest.get("file_count", 0) or len(manifest.get("files", []))),
        "total_size_bytes": int(manifest.get("total_size_bytes", 0)),
        "backup_path": "" if backup_info is None else str(backup_info["bundle_path"]),
        "created_at": str(manifest.get("created_at", "")),
    }


def _resolve_bundle_output_path(
    project_root: Path,
    *,
    bundle_path: str | Path | None,
    bundle_name: str,
) -> Path:
    if bundle_path is not None:
        target = Path(bundle_path)
        if not target.is_absolute():
            target = project_root / target
        target.parent.mkdir(parents=True, exist_ok=True)
        return target
    target_dir = project_root / "release" / "migration_bundles"
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / _normalize_bundle_name(bundle_name)


def _normalize_bundle_name(bundle_name: str) -> str:
    raw_name = bundle_name.strip() or f"migration_bundle_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    cleaned = raw_name.replace("\\", "_").replace("/", "_")
    return cleaned if cleaned.endswith(".zip") else f"{cleaned}.zip"


def _default_backup_name() -> str:
    return f"migration_backup_before_import_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.zip"


def _iter_root_files(root_path: Path) -> list[Path]:
    if root_path.is_file():
        return [root_path]
    if not root_path.exists():
        return []
    return [path for path in sorted(root_path.rglob("*")) if path.is_file()]


def _clear_existing_roots(project_root: Path, roots: list[Path]) -> None:
    ordered_roots = sorted(
        {_validate_relative_path(root.as_posix()) for root in roots},
        key=lambda item: (len(item.parts), item.as_posix()),
        reverse=True,
    )
    for relative_root in ordered_roots:
        target = project_root / relative_root
        if not target.exists():
            continue
        if target.is_dir():
            _remove_directory_with_fallback(target)
        else:
            target.unlink()


def _extract_payload(project_root: Path, bundle_path: Path) -> None:
    with zipfile.ZipFile(bundle_path, "r") as archive:
        for member in archive.infolist():
            if member.filename == BUNDLE_MANIFEST_NAME:
                continue
            if not member.filename.startswith(f"{_PAYLOAD_PREFIX}/"):
                continue
            relative_name = member.filename[len(_PAYLOAD_PREFIX) + 1 :]
            if not relative_name:
                continue
            relative_path = _validate_relative_path(relative_name)
            if _should_skip_import_bundle_file(relative_path):
                continue
            target_path = project_root / relative_path
            if member.is_dir():
                target_path.mkdir(parents=True, exist_ok=True)
                continue
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member, "r") as source, target_path.open("wb") as target:
                shutil.copyfileobj(source, target)


def _validate_relative_path(value: str) -> Path:
    normalized = str(PurePosixPath(value)).strip("/")
    if not normalized:
        raise ValueError("迁移包中的路径不能为空。")
    relative_path = Path(normalized)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"迁移包中包含非法路径: {value}")
    return relative_path


def _path_contains(parent: Path, child: Path) -> bool:
    return child == parent or parent in child.parents


def _should_skip_export_bundle_file(relative_path: Path) -> bool:
    return _is_logs_path(relative_path) or relative_path.suffix.lower() in _EXPORT_EXCLUDED_SUFFIXES


def _should_skip_import_bundle_file(relative_path: Path) -> bool:
    return _is_logs_path(relative_path)


def _is_logs_path(relative_path: Path) -> bool:
    return bool(relative_path.parts) and relative_path.parts[0] == "logs"


def _remove_directory_with_fallback(target: Path) -> None:
    try:
        shutil.rmtree(target)
        return
    except OSError as exc:
        if exc.errno != errno.ENOTEMPTY:
            raise
    staged = _stage_directory_for_removal(target)
    try:
        shutil.rmtree(staged)
    except OSError:
        # Some notebook-hosted filesystems can briefly keep a renamed directory
        # non-empty while handles are being released. Import can continue once
        # the original path has been freed for extraction.
        pass


def _stage_directory_for_removal(target: Path) -> Path:
    parent = target.parent
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    candidate = parent / f".{target.name}_migration_replace_{timestamp}"
    index = 1
    while candidate.exists():
        candidate = parent / f".{target.name}_migration_replace_{timestamp}_{index}"
        index += 1
    target.rename(candidate)
    return candidate
