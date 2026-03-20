from __future__ import annotations

import unittest

from src.app_utils import (
    compress_text_for_prompt,
    estimate_token_count,
    split_text_into_token_windows,
    trim_text_to_token_limit,
)


class TokenUtilsTests(unittest.TestCase):
    def test_estimate_token_count_returns_positive_value(self) -> None:
        self.assertGreater(estimate_token_count("Transformer attention"), 0)
        self.assertGreater(estimate_token_count("中文上下文预算测试"), 0)

    def test_trim_text_to_token_limit_shrinks_text(self) -> None:
        text = "attention " * 200
        trimmed = trim_text_to_token_limit(text, 20)

        self.assertLessEqual(estimate_token_count(trimmed), 20)
        self.assertLess(len(trimmed), len(text))

    def test_split_text_into_token_windows_creates_overlapping_windows(self) -> None:
        text = " ".join(f"token{i}" for i in range(200))
        windows = split_text_into_token_windows(text, window_tokens=30, overlap_tokens=10)

        self.assertGreater(len(windows), 1)
        self.assertTrue(any("token20" in window for window in windows[1:]))

    def test_compress_text_for_prompt_preserves_head_and_tail(self) -> None:
        text = "HEAD " + ("middle " * 200) + "TAIL"
        compressed = compress_text_for_prompt(text, 30)

        self.assertLessEqual(estimate_token_count(compressed), 30)
        self.assertIn("HEAD", compressed)
        self.assertIn("TAIL", compressed)


if __name__ == "__main__":
    unittest.main()
