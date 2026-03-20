from __future__ import annotations

import unittest

from src.errors import classify_provider_exception


class ErrorClassificationTests(unittest.TestCase):
    def test_classify_provider_exception_maps_max_seq_len_to_context_length(self) -> None:
        exc = RuntimeError(
            "Error code: 400 - {'code': 20015, 'message': "
            "'length of prompt_tokens (75033) must be less than max_seq_len (32768).'}"
        )
        mapped = classify_provider_exception(exc, "zh")
        self.assertEqual(mapped.code, "context_length")
        self.assertIn("上下文", mapped.user_message)


if __name__ == "__main__":
    unittest.main()
