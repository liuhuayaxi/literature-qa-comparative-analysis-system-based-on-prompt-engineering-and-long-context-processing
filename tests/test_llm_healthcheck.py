from __future__ import annotations

import unittest
from unittest.mock import patch

from src.config import AppConfig
from src.llm_tools import run_chat_healthcheck


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeLLM:
    def __init__(self, streaming: bool) -> None:
        self.streaming = streaming

    async def ainvoke(self, _prompt: str):
        return _FakeResponse("OK")

    async def astream(self, _prompt: str):
        yield _FakeResponse("O")
        yield _FakeResponse("K")


class ChatHealthcheckTests(unittest.IsolatedAsyncioTestCase):
    async def test_healthcheck_reports_non_stream_and_stream_success(self) -> None:
        config = AppConfig(
            openai_api_key="key",
            openai_base_url="https://example.com/v1",
            chat_model="demo-chat",
            embedding_model="demo-embed",
        )

        with patch("src.llm_tools.build_chat_llm", side_effect=lambda cfg, streaming=False: _FakeLLM(streaming)):
            result = await run_chat_healthcheck(config, timeout_seconds=1.0)

        self.assertTrue(result["ok"])
        self.assertTrue(result["non_stream"]["ok"])
        self.assertTrue(result["stream"]["ok"])
        self.assertEqual(result["non_stream"]["preview"], "OK")
        self.assertEqual(result["stream"]["preview"], "OK")


if __name__ == "__main__":
    unittest.main()
