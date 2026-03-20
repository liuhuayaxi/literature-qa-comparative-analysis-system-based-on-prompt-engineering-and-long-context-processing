from __future__ import annotations

import errno
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from src.app_utils import update_json_config_file
from src.config import AppConfig
from src.migration_bundle import (
    BUNDLE_MANIFEST_NAME,
    _clear_existing_roots,
    collect_migration_roots,
    export_migration_bundle,
    import_migration_bundle,
    inspect_migration_bundle,
)


class MigrationBundleTests(unittest.TestCase):
    def _make_config(self, root: Path, *, chat_model: str = "") -> AppConfig:
        config_path = root / "config" / "app_config.json"
        update_json_config_file(
            config_path,
            {
                "OPENAI_CHAT_MODEL": chat_model,
            },
        )
        config = AppConfig.from_file(config_path)
        config.ensure_directories()
        return config

    def test_collect_migration_roots_collapses_runtime_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_config(Path(tmpdir))

            roots = {path.as_posix() for path in collect_migration_roots(config)}

            self.assertEqual(roots, {"config", "storage", "data/raw", "reports"})

    def test_export_bundle_excludes_logs_and_pdf_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = self._make_config(root, chat_model="demo-chat")
            (config.db_path).write_text("sqlite payload", encoding="utf-8")
            (config.vector_dir / "chroma.sqlite3").write_text("vector payload", encoding="utf-8")
            (config.cache_dir / "analysis" / "demo.json").parent.mkdir(parents=True, exist_ok=True)
            (config.cache_dir / "analysis" / "demo.json").write_text('{"ok": true}', encoding="utf-8")
            (config.data_root / "demo-course" / "lectures" / "demo.pdf").parent.mkdir(parents=True, exist_ok=True)
            (config.data_root / "demo-course" / "lectures" / "demo.pdf").write_bytes(b"%PDF-demo")
            (config.data_root / "demo-course" / "notes" / "demo.txt").parent.mkdir(parents=True, exist_ok=True)
            (config.data_root / "demo-course" / "notes" / "demo.txt").write_text("demo text", encoding="utf-8")
            (config.reports_dir / "demo.md").write_text("# report", encoding="utf-8")
            config.runtime_api_log_path.write_text("runtime log", encoding="utf-8")
            config.vector_operation_log_path.write_text("vector log", encoding="utf-8")

            result = export_migration_bundle(config, bundle_name="demo_migration")
            bundle_path = Path(result["bundle_path"])

            self.assertTrue(bundle_path.exists())
            manifest = inspect_migration_bundle(bundle_path)
            self.assertEqual(set(manifest["roots"]), {"config", "storage", "data/raw", "reports"})
            self.assertEqual(manifest["file_count"], result["file_count"])

            with zipfile.ZipFile(bundle_path, "r") as archive:
                names = set(archive.namelist())

            self.assertIn(BUNDLE_MANIFEST_NAME, names)
            self.assertIn("payload/config/app_config.json", names)
            self.assertIn("payload/storage/app_state.db", names)
            self.assertIn("payload/storage/chroma/chroma.sqlite3", names)
            self.assertIn("payload/data/raw/demo-course/notes/demo.txt", names)
            self.assertIn("payload/reports/demo.md", names)
            self.assertNotIn("payload/data/raw/demo-course/lectures/demo.pdf", names)
            self.assertNotIn("payload/logs/runtime_api_traffic.jsonl", names)
            self.assertNotIn("payload/logs/runtime_vector_operations.jsonl", names)

    def test_import_bundle_restores_files_and_creates_backup(self) -> None:
        with tempfile.TemporaryDirectory() as source_tmpdir, tempfile.TemporaryDirectory() as target_tmpdir:
            source_root = Path(source_tmpdir)
            target_root = Path(target_tmpdir)

            source_config = self._make_config(source_root, chat_model="source-model")
            source_config.runtime_api_log_path.write_text("source runtime log", encoding="utf-8")
            (source_config.db_path).write_text("source db", encoding="utf-8")
            (source_config.cache_dir / "analysis" / "fresh.json").parent.mkdir(parents=True, exist_ok=True)
            (source_config.cache_dir / "analysis" / "fresh.json").write_text('{"fresh": true}', encoding="utf-8")
            (source_config.data_root / "source-course" / "lectures" / "source.pdf").parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            (source_config.data_root / "source-course" / "lectures" / "source.pdf").write_bytes(b"source pdf")
            (source_config.data_root / "source-course" / "notes" / "source.txt").parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            (source_config.data_root / "source-course" / "notes" / "source.txt").write_text(
                "source text",
                encoding="utf-8",
            )
            (source_config.reports_dir / "source.md").write_text("source report", encoding="utf-8")
            source_bundle = Path(export_migration_bundle(source_config, bundle_name="source_bundle")["bundle_path"])

            target_config = self._make_config(target_root, chat_model="target-model")
            target_config.runtime_api_log_path.write_text("keep target log", encoding="utf-8")
            (target_config.db_path).write_text("target db", encoding="utf-8")
            (target_config.cache_dir / "analysis" / "stale.json").parent.mkdir(parents=True, exist_ok=True)
            (target_config.cache_dir / "analysis" / "stale.json").write_text('{"stale": true}', encoding="utf-8")
            (target_config.data_root / "target-course" / "lectures" / "stale.pdf").parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            (target_config.data_root / "target-course" / "lectures" / "stale.pdf").write_bytes(b"stale pdf")
            (target_config.data_root / "target-course" / "notes" / "stale.txt").parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            (target_config.data_root / "target-course" / "notes" / "stale.txt").write_text("stale text", encoding="utf-8")

            result = import_migration_bundle(target_config, source_bundle)
            backup_path = Path(result["backup_path"])

            self.assertTrue(backup_path.exists())
            self.assertEqual(AppConfig.from_file(target_config.config_path).chat_model, "source-model")
            self.assertEqual(target_config.runtime_api_log_path.read_text(encoding="utf-8"), "keep target log")
            self.assertEqual(target_config.db_path.read_text(encoding="utf-8"), "source db")
            self.assertFalse((target_config.cache_dir / "analysis" / "stale.json").exists())
            self.assertEqual(
                (target_config.cache_dir / "analysis" / "fresh.json").read_text(encoding="utf-8"),
                '{"fresh": true}',
            )
            self.assertFalse((target_config.data_root / "target-course" / "lectures" / "stale.pdf").exists())
            self.assertFalse((target_config.data_root / "source-course" / "lectures" / "source.pdf").exists())
            self.assertFalse((target_config.data_root / "target-course" / "notes" / "stale.txt").exists())
            self.assertEqual(
                (target_config.data_root / "source-course" / "notes" / "source.txt").read_text(encoding="utf-8"),
                "source text",
            )
            self.assertEqual(
                (target_config.reports_dir / "source.md").read_text(encoding="utf-8"),
                "source report",
            )

            backup_manifest = inspect_migration_bundle(backup_path)
            self.assertIn("storage", backup_manifest["roots"])
            with zipfile.ZipFile(backup_path, "r") as archive:
                names = set(archive.namelist())
            self.assertIn("payload/storage/cache/analysis/stale.json", names)
            self.assertIn("payload/data/raw/target-course/notes/stale.txt", names)
            self.assertNotIn("payload/data/raw/target-course/lectures/stale.pdf", names)

    def test_import_bundle_skips_pdf_from_legacy_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as source_tmpdir, tempfile.TemporaryDirectory() as target_tmpdir:
            source_root = Path(source_tmpdir)
            target_root = Path(target_tmpdir)

            source_config = self._make_config(source_root, chat_model="source-model")
            (source_config.data_root / "source-course" / "lectures" / "legacy.pdf").parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            (source_config.data_root / "source-course" / "lectures" / "legacy.pdf").write_bytes(b"legacy pdf")
            bundle_path = Path(export_migration_bundle(source_config, bundle_name="legacy_bundle")["bundle_path"])

            with zipfile.ZipFile(bundle_path, "a", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("payload/data/raw/source-course/lectures/legacy.pdf", b"legacy pdf from old bundle")

            target_config = self._make_config(target_root, chat_model="target-model")
            import_migration_bundle(target_config, bundle_path)

            self.assertFalse((target_config.data_root / "source-course" / "lectures" / "legacy.pdf").exists())

    def test_clear_existing_roots_handles_enotempty_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            target = project_root / "storage" / "chroma"
            target.mkdir(parents=True, exist_ok=True)
            (target / "busy.bin").write_text("busy", encoding="utf-8")

            real_rmtree = shutil.rmtree
            first_call = True

            def flaky_rmtree(path: str | Path, *args: object, **kwargs: object) -> None:
                nonlocal first_call
                candidate = Path(path)
                if first_call and candidate == target:
                    first_call = False
                    raise OSError(errno.ENOTEMPTY, "Directory not empty", str(candidate))
                real_rmtree(path, *args, **kwargs)

            with patch("src.migration_bundle.shutil.rmtree", side_effect=flaky_rmtree):
                _clear_existing_roots(project_root, [Path("storage/chroma")])

            self.assertFalse(target.exists())
            staged = list((project_root / "storage").glob(".chroma_migration_replace_*"))
            self.assertEqual(staged, [])


if __name__ == "__main__":
    unittest.main()
