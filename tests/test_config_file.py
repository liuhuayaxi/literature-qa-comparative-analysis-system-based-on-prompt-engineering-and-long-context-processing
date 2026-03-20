from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.app_utils import update_json_config_file
from src.config import AppConfig


class ConfigFileTests(unittest.TestCase):
    def test_config_file_round_trip_preserves_multiline_prompts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "config" / "app_config.json"
            update_json_config_file(
                config_path,
                {
                    "APP_CHAT_TIMEOUT_SECONDS": 33,
                    "APP_EMBEDDING_TIMEOUT_SECONDS": 29,
                    "APP_API_RETRY_ATTEMPTS": 4,
                    "APP_API_RETRY_BACKOFF_MIN_SECONDS": 2,
                    "APP_API_RETRY_BACKOFF_MAX_SECONDS": 9,
                    "APP_RATE_LIMIT_RETRY_FOREVER": False,
                    "APP_RATE_LIMIT_RETRY_ATTEMPTS": 5,
                    "APP_RATE_LIMIT_RETRY_DELAY_SECONDS": 1,
                    "APP_CHAT_PROVIDER_CONCURRENCY": 2,
                    "APP_EMBEDDING_PROVIDER_CONCURRENCY": 3,
                    "APP_MAX_INPUT_CHARS": 2800,
                    "APP_ENABLE_SENSITIVE_INPUT_CHECK": True,
                    "APP_ENABLE_RESULT_CACHE": False,
                    "APP_DEFAULT_STREAMING_MODE": "non_stream",
                    "APP_MODEL_CONTEXT_WINDOW": 32000,
                    "APP_ANSWER_TOKEN_RESERVE": 5000,
                    "APP_LONG_CONTEXT_WINDOW_TOKENS": 2200,
                    "APP_LONG_CONTEXT_WINDOW_OVERLAP_TOKENS": 220,
                    "APP_RECURSIVE_SUMMARY_TARGET_TOKENS": 1300,
                    "APP_RECURSIVE_SUMMARY_BATCH_SIZE": 5,
                    "APP_PROMPT_COMPRESSION_TURN_TOKEN_LIMIT": 160,
                    "APP_RECENT_HISTORY_TURNS": 4,
                    "APP_RETRIEVAL_TOP_K": 7,
                    "APP_RETRIEVAL_FETCH_K": 18,
                    "APP_CITATION_LIMIT": 3,
                    "APP_ENABLE_RERANK": True,
                    "APP_RERANK_MIN_SCORE": 0.31,
                    "APP_RERANK_MIN_KEEP": 2,
                    "APP_RERANK_WEIGHT_VECTOR": 0.5,
                    "APP_RERANK_WEIGHT_KEYWORD": 0.2,
                    "APP_RERANK_WEIGHT_PHRASE": 0.2,
                    "APP_RERANK_WEIGHT_METADATA": 0.1,
                    "APP_MERGE_SMALL_CHUNKS": True,
                    "APP_MIN_CHUNK_SIZE": 360,
                    "APP_RUNTIME_API_LOG_PATH": "logs/runtime_custom.jsonl",
                    "APP_TEST_API_LOG_PATH": "logs/test_custom.jsonl",
                    "APP_CACHE_DIR": "storage/custom_cache",
                    "APP_ANALYSIS_CHECKPOINT_DIR": "storage/custom_checkpoints",
                    "APP_FIELD_TEMPLATE_PATH": "storage/custom_templates.json",
                    "APP_KNOWLEDGE_BASE_STATE_PATH": "storage/custom_kb_state.json",
                    "APP_ENABLE_QUERY_REWRITE": False,
                    "APP_ENABLE_MIGRATION_UI": False,
                    "APP_QA_SYSTEM_PROMPT_EN": "System line 1\nSystem line 2",
                    "APP_QA_SYSTEM_PROMPT_ZH": "第一行提示词\n第二行提示词",
                    "APP_QUERY_REWRITE_INSTRUCTION_EN": "Rewrite A\nRewrite B",
                    "APP_QUERY_REWRITE_INSTRUCTION_ZH": "改写A\n改写B",
                    "APP_QA_ANSWER_INSTRUCTION_EN": "Answer format A\nAnswer format B",
                    "APP_QA_ANSWER_INSTRUCTION_ZH": "回答格式A\n回答格式B",
                    "APP_SINGLE_ANALYSIS_PROMPT_ZH": "分析提示A\n分析提示B",
                    "APP_COMPARE_REPORT_PROMPT_ZH": "报告提示A\n报告提示B",
                    "APP_SINGLE_ANALYSIS_PROMPT_EN": "Analysis prompt A\nAnalysis prompt B",
                    "APP_COMPARE_REPORT_PROMPT_EN": "Report prompt A\nReport prompt B",
                },
            )

            config = AppConfig.from_file(config_path)
            resolved_root = config.project_root

            self.assertEqual(config.project_root, root.resolve(strict=False))
            self.assertEqual(config.data_root, resolved_root / "data/raw")
            self.assertEqual(config.reports_dir, resolved_root / "reports")
            self.assertEqual(config.chat_timeout_seconds, 33)
            self.assertEqual(config.embedding_timeout_seconds, 29)
            self.assertEqual(config.api_retry_attempts, 4)
            self.assertEqual(config.api_retry_backoff_min_seconds, 2)
            self.assertEqual(config.api_retry_backoff_max_seconds, 9)
            self.assertFalse(config.rate_limit_retry_forever)
            self.assertEqual(config.rate_limit_retry_attempts, 5)
            self.assertEqual(config.rate_limit_retry_delay_seconds, 1)
            self.assertEqual(config.chat_provider_concurrency, 2)
            self.assertEqual(config.embedding_provider_concurrency, 3)
            self.assertEqual(config.max_input_chars, 2800)
            self.assertTrue(config.enable_sensitive_input_check)
            self.assertFalse(config.enable_result_cache)
            self.assertEqual(config.default_streaming_mode, "non_stream")
            self.assertEqual(config.model_context_window, 32000)
            self.assertEqual(config.answer_token_reserve, 5000)
            self.assertEqual(config.long_context_window_tokens, 2200)
            self.assertEqual(config.long_context_window_overlap_tokens, 220)
            self.assertEqual(config.recursive_summary_target_tokens, 1300)
            self.assertEqual(config.recursive_summary_batch_size, 5)
            self.assertEqual(config.prompt_compression_turn_token_limit, 160)
            self.assertEqual(config.recent_history_turns, 4)
            self.assertEqual(config.retrieval_top_k, 7)
            self.assertEqual(config.retrieval_fetch_k, 18)
            self.assertEqual(config.citation_limit, 3)
            self.assertTrue(config.enable_rerank)
            self.assertAlmostEqual(config.rerank_min_score, 0.31)
            self.assertEqual(config.rerank_min_keep, 2)
            self.assertAlmostEqual(config.rerank_weight_vector, 0.5)
            self.assertAlmostEqual(config.rerank_weight_keyword, 0.2)
            self.assertAlmostEqual(config.rerank_weight_phrase, 0.2)
            self.assertAlmostEqual(config.rerank_weight_metadata, 0.1)
            self.assertTrue(config.merge_small_chunks)
            self.assertEqual(config.min_chunk_size, 360)
            self.assertEqual(config.api_log_path, resolved_root / "logs/runtime_custom.jsonl")
            self.assertEqual(config.runtime_api_log_path, resolved_root / "logs/runtime_custom.jsonl")
            self.assertEqual(config.test_api_log_path, resolved_root / "logs/test_custom.jsonl")
            self.assertEqual(config.cache_dir, resolved_root / "storage/custom_cache")
            self.assertEqual(config.analysis_checkpoint_dir, resolved_root / "storage/custom_checkpoints")
            self.assertEqual(config.field_template_path, resolved_root / "storage/custom_templates.json")
            self.assertEqual(config.knowledge_base_state_path, resolved_root / "storage/custom_kb_state.json")
            self.assertFalse(config.enable_query_rewrite)
            self.assertFalse(config.enable_migration_ui)
            self.assertEqual(config.qa_system_prompt_en, "System line 1\nSystem line 2")
            self.assertEqual(config.qa_system_prompt_zh, "第一行提示词\n第二行提示词")
            self.assertEqual(config.query_rewrite_instruction_en, "Rewrite A\nRewrite B")
            self.assertEqual(config.query_rewrite_instruction_zh, "改写A\n改写B")
            self.assertEqual(config.qa_answer_instruction_en, "Answer format A\nAnswer format B")
            self.assertEqual(config.qa_answer_instruction_zh, "回答格式A\n回答格式B")
            self.assertEqual(config.single_analysis_prompt_zh, "分析提示A\n分析提示B")
            self.assertEqual(config.compare_report_prompt_zh, "报告提示A\n报告提示B")
            self.assertEqual(config.single_analysis_prompt_en, "Analysis prompt A\nAnalysis prompt B")
            self.assertEqual(config.compare_report_prompt_en, "Report prompt A\nReport prompt B")

    def test_missing_config_file_is_initialized_and_notebook_data_is_merged(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            legacy_raw_file = root / "notebooks/data/raw/lunwen/papers/demo.txt"
            legacy_report = root / "notebooks/reports/lunwen_progress.md"
            legacy_runtime_log = root / "notebooks/logs/runtime_api_traffic.jsonl"
            legacy_db = root / "notebooks/storage/app_state.db"
            legacy_checkpoint = root / "notebooks/storage/analysis_checkpoints/demo.json"
            legacy_cache = root / "notebooks/storage/cache/report_lunwen/demo.json"
            legacy_vector_file = root / "notebooks/storage/chroma/chroma.sqlite3"
            legacy_raw_file.parent.mkdir(parents=True, exist_ok=True)
            legacy_report.parent.mkdir(parents=True, exist_ok=True)
            legacy_runtime_log.parent.mkdir(parents=True, exist_ok=True)
            legacy_db.parent.mkdir(parents=True, exist_ok=True)
            legacy_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            legacy_cache.parent.mkdir(parents=True, exist_ok=True)
            legacy_vector_file.parent.mkdir(parents=True, exist_ok=True)
            legacy_raw_file.write_text("legacy raw", encoding="utf-8")
            legacy_report.write_text("legacy report", encoding="utf-8")
            legacy_runtime_log.write_text("legacy runtime log", encoding="utf-8")
            legacy_db.write_text("legacy db", encoding="utf-8")
            legacy_checkpoint.write_text(
                '{"progress_output_path": "'
                + str((root / "notebooks/reports/lunwen_progress.md").resolve(strict=False))
                + '"}',
                encoding="utf-8",
            )
            legacy_cache.write_text(
                '{"output_path": "'
                + str((root / "notebooks/reports/lunwen_report.md").resolve(strict=False))
                + '"}',
                encoding="utf-8",
            )
            legacy_vector_file.write_text("legacy vector", encoding="utf-8")

            config_path = root / "config" / "app_config.json"
            config = AppConfig.from_file(config_path)
            config.ensure_directories()
            resolved_root = config.project_root

            self.assertTrue(config_path.exists())
            self.assertFalse(config.enable_migration_ui)
            self.assertTrue((resolved_root / "data/raw/lunwen/papers/demo.txt").exists())
            self.assertTrue((resolved_root / "reports/lunwen_progress.md").exists())
            self.assertTrue((resolved_root / "logs/runtime_api_traffic.jsonl").exists())
            self.assertTrue((resolved_root / "storage/app_state.db").exists())
            self.assertTrue((resolved_root / "storage/chroma/chroma.sqlite3").exists())
            self.assertTrue((resolved_root / "storage/analysis_checkpoints/demo.json").exists())
            self.assertTrue((resolved_root / "storage/cache/report_lunwen/demo.json").exists())
            self.assertIn(
                str((resolved_root / "reports").resolve(strict=False)),
                (resolved_root / "storage/analysis_checkpoints/demo.json").read_text(encoding="utf-8"),
            )
            self.assertTrue(config.migration_messages)
            self.assertTrue(any("知识库原始文件" in message for message in config.migration_messages))
            self.assertTrue((resolved_root / "notebooks/data/raw/lunwen/papers/demo.txt").exists())

    def test_separate_chat_and_embedding_provider_settings_take_priority(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config" / "app_config.json"
            update_json_config_file(
                config_path,
                {
                    "OPENAI_API_KEY": "legacy-key",
                    "OPENAI_BASE_URL": "https://legacy.example.com/v1",
                    "OPENAI_CHAT_API_KEY": "chat-key",
                    "OPENAI_CHAT_BASE_URL": "https://chat.example.com/v1",
                    "OPENAI_CHAT_MODEL": "chat-model",
                    "OPENAI_EMBEDDING_API_KEY": "embed-key",
                    "OPENAI_EMBEDDING_BASE_URL": "https://embed.example.com/v1",
                    "OPENAI_EMBEDDING_MODEL": "embed-model",
                },
            )

            config = AppConfig.from_file(config_path)

            self.assertEqual(config.resolved_chat_api_key, "chat-key")
            self.assertEqual(config.resolved_chat_base_url, "https://chat.example.com/v1")
            self.assertEqual(config.resolved_embedding_api_key, "embed-key")
            self.assertEqual(config.resolved_embedding_base_url, "https://embed.example.com/v1")
            self.assertTrue(config.has_chat_model_credentials)
            self.assertTrue(config.has_embedding_model_credentials)


if __name__ == "__main__":
    unittest.main()
