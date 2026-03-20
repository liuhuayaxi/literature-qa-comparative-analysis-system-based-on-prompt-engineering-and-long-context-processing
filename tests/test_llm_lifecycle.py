from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

from src.config import AppConfig
from src.errors import ProviderRequestError
from src.llm_tools import invoke_chat_text


class _TimeoutLLM:
    async def ainvoke(self, _prompt: str):
        raise TimeoutError("request timed out")


class _OkLLM:
    async def ainvoke(self, _prompt: str):
        return type("Resp", (), {"content": "ok"})()


class _RateLimitedLLM:
    async def ainvoke(self, _prompt: str):
        raise RuntimeError("429 rate limit exceeded")


class LlmLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_invoke_chat_text_retries_after_timeout(self) -> None:
        config = AppConfig(
            chat_timeout_seconds=1,
            api_retry_attempts=2,
            api_retry_backoff_min_seconds=0,
            api_retry_backoff_max_seconds=0,
        )
        with patch("src.llm_tools.build_chat_llm", side_effect=[_TimeoutLLM(), _OkLLM()]):
            result = await invoke_chat_text(config, "prompt", "zh")
        self.assertEqual(result, "ok")

    async def test_invoke_chat_text_raises_rate_limit_error(self) -> None:
        config = AppConfig(
            chat_timeout_seconds=1,
            api_retry_attempts=1,
            api_retry_backoff_min_seconds=0,
            api_retry_backoff_max_seconds=0,
            rate_limit_retry_forever=False,
            rate_limit_retry_attempts=0,
        )
        with patch("src.llm_tools.build_chat_llm", return_value=_RateLimitedLLM()):
            with self.assertRaises(ProviderRequestError) as context:
                await invoke_chat_text(config, "prompt", "zh")
        self.assertEqual(context.exception.code, "rate_limit")


if __name__ == "__main__":
    unittest.main()
