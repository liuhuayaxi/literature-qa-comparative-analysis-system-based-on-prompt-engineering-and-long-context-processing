from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from src.config import AppConfig
from src.errors import ProviderRequestError
from src.llm_tools import invoke_chat_text


class _AlwaysRateLimitedLlm:
    async def ainvoke(self, prompt: str):
        raise RuntimeError("429 Too Many Requests")


class _FakeQueueSlot:
    def __init__(self) -> None:
        self.release_calls = 0

    def release(self) -> None:
        self.release_calls += 1


class _UsageLlm:
    async def ainvoke(self, _prompt: str):
        return type(
            "Resp",
            (),
            {
                "content": "ok",
                "usage_metadata": {
                    "input_tokens": 123,
                    "output_tokens": 45,
                    "total_tokens": 168,
                },
            },
        )()


class LlmToolsRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_invoke_chat_text_retries_rate_limit_five_times(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AppConfig(
                chat_api_key="chat-key",
                chat_base_url="https://api.example.com/v1",
                chat_model="demo-model",
                api_retry_attempts=3,
                rate_limit_retry_forever=False,
                rate_limit_retry_attempts=5,
                rate_limit_retry_delay_seconds=30,
                api_log_path=Path(tmpdir) / "logs" / "runtime_api.jsonl",
                runtime_api_log_path=Path(tmpdir) / "logs" / "runtime_api.jsonl",
                test_api_log_path=Path(tmpdir) / "logs" / "test_api.jsonl",
            )
            sleep_mock = AsyncMock()

            with patch("src.llm_tools.build_chat_llm", return_value=_AlwaysRateLimitedLlm()), patch(
                "src.llm_tools.asyncio.sleep",
                new=sleep_mock,
            ):
                with self.assertRaises(ProviderRequestError) as ctx:
                    await invoke_chat_text(config, "hello", "zh")

            self.assertEqual(ctx.exception.code, "rate_limit")
            self.assertIn("连续重试 5 次仍失败", ctx.exception.user_message)
            self.assertEqual(sleep_mock.await_count, 5)
            sleep_mock.assert_called_with(30)
            self.assertTrue(config.active_api_log_path.exists())

    async def test_invoke_chat_text_releases_provider_slot_before_rate_limit_sleep(self) -> None:
        config = AppConfig(
            chat_api_key="chat-key",
            chat_base_url="https://api.example.com/v1",
            chat_model="demo-model",
            api_retry_attempts=1,
            rate_limit_retry_forever=False,
            rate_limit_retry_attempts=1,
            rate_limit_retry_delay_seconds=1,
        )
        queue_slot = _FakeQueueSlot()

        async def fake_sleep(_seconds: float) -> None:
            self.assertEqual(queue_slot.release_calls, 1)

        with patch("src.llm_tools._acquire_chat_slot", AsyncMock(return_value=queue_slot)), patch(
            "src.llm_tools.build_chat_llm",
            return_value=_AlwaysRateLimitedLlm(),
        ), patch("src.llm_tools.sleep_with_task_control", new=fake_sleep):
            with self.assertRaises(ProviderRequestError):
                await invoke_chat_text(config, "hello", "zh")

        self.assertGreaterEqual(queue_slot.release_calls, 2)

    async def test_invoke_chat_text_logs_prompt_estimate_and_provider_usage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "logs" / "runtime_api.jsonl"
            config = AppConfig(
                chat_api_key="chat-key",
                chat_base_url="https://api.example.com/v1",
                chat_model="demo-model",
                api_log_path=log_path,
                runtime_api_log_path=log_path,
                test_api_log_path=log_path,
            )
            with patch("src.llm_tools.build_chat_llm", return_value=_UsageLlm()):
                result = await invoke_chat_text(config, "hello prompt", "zh")
            self.assertEqual(result, "ok")
            lines = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
            request_event = next(item for item in lines if item.get("event") == "request")
            response_event = next(item for item in lines if item.get("event") == "response")
            self.assertGreater(int(request_event.get("prompt_tokens_estimate", 0)), 0)
            self.assertGreater(int(response_event.get("completion_tokens_estimate", 0)), 0)
            self.assertEqual(response_event.get("provider_prompt_tokens"), 123)
            self.assertEqual(response_event.get("provider_completion_tokens"), 45)
            self.assertEqual(response_event.get("provider_total_tokens"), 168)


if __name__ == "__main__":
    unittest.main()
