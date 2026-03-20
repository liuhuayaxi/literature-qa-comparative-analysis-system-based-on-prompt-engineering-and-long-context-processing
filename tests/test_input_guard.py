from __future__ import annotations

import unittest

from src.errors import InputValidationError
from src.app_utils import sanitize_user_question_for_prompt, validate_user_text


class InputGuardTests(unittest.TestCase):
    def test_validate_user_text_rejects_empty_input(self) -> None:
        with self.assertRaises(InputValidationError) as context:
            validate_user_text("   ", language="zh", max_chars=100)
        self.assertEqual(context.exception.code, "empty_input")

    def test_validate_user_text_rejects_illegal_control_chars(self) -> None:
        with self.assertRaises(InputValidationError) as context:
            validate_user_text("hello\x00world", language="en", max_chars=100)
        self.assertEqual(context.exception.code, "illegal_characters")

    def test_validate_user_text_rejects_sensitive_content(self) -> None:
        with self.assertRaises(InputValidationError) as context:
            validate_user_text("api_key=sk-secret-secret-secret-123456", language="zh", max_chars=200)
        self.assertEqual(context.exception.code, "sensitive_content")

    def test_validate_user_text_accepts_normal_question(self) -> None:
        result = validate_user_text("什么是RAG？", language="zh", max_chars=100)
        self.assertEqual(result, "什么是RAG？")

    def test_sanitize_user_question_for_prompt_collapses_repeated_chars(self) -> None:
        noisy = "A" * 300
        cleaned, strategies = sanitize_user_question_for_prompt(noisy, language="zh")
        self.assertIn("问题去噪", strategies)
        self.assertIn("已省略", cleaned)
        self.assertLess(len(cleaned), len(noisy))

    def test_sanitize_user_question_for_prompt_trims_token_budget(self) -> None:
        noisy = "背景说明 " * 2000
        cleaned, strategies = sanitize_user_question_for_prompt(
            noisy,
            language="zh",
            model_name="",
            token_limit=80,
        )
        self.assertIn("问题压缩", strategies)
        self.assertTrue(cleaned)
        self.assertLess(len(cleaned), len(noisy))


if __name__ == "__main__":
    unittest.main()
