"""
Tests for the pure helpers in config.py and utils.py.

These avoid importing the heavy RAG pipeline (langchain / google clients) and
need no API key, so they run fast and offline. They cover the environment-
variable parsing that drives all app configuration, plus byte formatting.

Run:  .venv/bin/python -m pytest tests/ -q
  or: .venv/bin/python -m unittest tests.test_config_utils
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config  # noqa: E402
import utils  # noqa: E402

VAR = "PDF_RAG_TEST_VAR_XYZ"  # unique name so we never clobber real config


class _EnvSandbox(unittest.TestCase):
    """Ensure our probe variable is always removed after each test."""

    def tearDown(self):
        os.environ.pop(VAR, None)


class TestGetStr(_EnvSandbox):
    def test_returns_value_when_set(self):
        os.environ[VAR] = "hello"
        self.assertEqual(config._get_str(VAR, "fallback"), "hello")

    def test_returns_default_when_missing(self):
        self.assertEqual(config._get_str(VAR, "fallback"), "fallback")

    def test_empty_string_falls_back_to_default(self):
        os.environ[VAR] = ""
        self.assertEqual(config._get_str(VAR, "fallback"), "fallback")


class TestGetNumbers(_EnvSandbox):
    def test_int_parses(self):
        os.environ[VAR] = "42"
        self.assertEqual(config._get_int(VAR, 0), 42)

    def test_int_bad_value_uses_default(self):
        os.environ[VAR] = "not-a-number"
        self.assertEqual(config._get_int(VAR, 5), 5)

    def test_float_parses(self):
        os.environ[VAR] = "0.25"
        self.assertAlmostEqual(config._get_float(VAR, 1.0), 0.25)

    def test_float_bad_value_uses_default(self):
        os.environ[VAR] = "xyz"
        self.assertAlmostEqual(config._get_float(VAR, 1.5), 1.5)


class TestGetBool(_EnvSandbox):
    def test_truthy_tokens(self):
        for token in ("1", "true", "TRUE", "Yes", "on"):
            os.environ[VAR] = token
            self.assertTrue(config._get_bool(VAR, False), f"{token!r} should be True")

    def test_falsey_tokens(self):
        for token in ("0", "false", "no", "off", "banana"):
            os.environ[VAR] = token
            self.assertFalse(config._get_bool(VAR, True), f"{token!r} should be False")

    def test_missing_uses_default(self):
        self.assertTrue(config._get_bool(VAR, True))
        self.assertFalse(config._get_bool(VAR, False))


class TestHumanSize(unittest.TestCase):
    def test_bytes(self):
        self.assertEqual(utils.human_size(512), "512 B")
        self.assertEqual(utils.human_size(0), "0 B")

    def test_kilobytes(self):
        self.assertEqual(utils.human_size(1536), "1.5 KB")
        self.assertEqual(utils.human_size(1024), "1.0 KB")

    def test_megabytes(self):
        self.assertEqual(utils.human_size(1024 * 1024), "1.0 MB")

    def test_gigabytes_and_above(self):
        self.assertEqual(utils.human_size(1024 ** 3), "1.0 GB")
        # Beyond GB the unit stays GB (no TB tier).
        self.assertTrue(utils.human_size(5 * 1024 ** 4).endswith("GB"))


if __name__ == "__main__":
    unittest.main()
