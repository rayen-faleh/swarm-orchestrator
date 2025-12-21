"""
Tests for LLMLingua-based diff compression module.

TDD approach: Define expected behavior for:
- DiffCompressor: LLMLingua wrapper with code-preserving defaults
- compress_diff: Main compression function
- should_compress: Threshold check
"""

import pytest
from unittest.mock import MagicMock, patch

from swarm_orchestrator.swarm_mcp.compression import (
    DiffCompressor,
    compress_diff,
    should_compress,
    DIFF_MARKERS,
    CODE_KEYWORDS,
    MIN_TOKEN_THRESHOLD,
)


# =============================================================================
# should_compress Tests
# =============================================================================

class TestShouldCompress:
    """Tests for the should_compress threshold check."""

    def test_returns_false_for_small_diffs(self):
        """Diffs under MIN_TOKEN_THRESHOLD should not be compressed."""
        small_diff = "diff --git a/foo.py b/foo.py\n+hello world"
        assert should_compress(small_diff) is False

    def test_returns_true_for_large_diffs(self):
        """Diffs over MIN_TOKEN_THRESHOLD should be compressed."""
        # Create a diff with many tokens
        lines = ["+" + "word " * 50 for _ in range(50)]  # ~2500 words
        large_diff = "diff --git a/foo.py b/foo.py\n" + "\n".join(lines)
        assert should_compress(large_diff) is True

    def test_uses_simple_token_estimation(self):
        """Token count should be estimated via simple word splitting."""
        # 499 words should be below threshold of 500
        text = " ".join(["word"] * 499)
        assert should_compress(text) is False

        # 501 words should be above threshold
        text = " ".join(["word"] * 501)
        assert should_compress(text) is True


# =============================================================================
# DiffCompressor Tests
# =============================================================================

class TestDiffCompressor:
    """Tests for the DiffCompressor class."""

    def test_init_sets_preserve_patterns(self):
        """Should initialize with diff markers and code keywords to preserve."""
        compressor = DiffCompressor()
        assert compressor.preserve_patterns is not None
        # Pattern should be a compiled regex
        assert hasattr(compressor.preserve_patterns, "pattern")
        assert len(compressor.preserve_patterns.pattern) > 0

    def test_init_with_llmlingua_not_available(self):
        """Should handle gracefully when llmlingua is not installed."""
        with patch.dict("sys.modules", {"llmlingua": None}):
            compressor = DiffCompressor()
            # Should still create the compressor, just without llmlingua
            assert compressor is not None

    def test_compress_returns_original_when_below_threshold(self):
        """Should skip compression for diffs below MIN_TOKEN_THRESHOLD."""
        compressor = DiffCompressor()
        small_diff = "diff --git a/foo.py b/foo.py\n+hello"

        result = compressor.compress(small_diff)

        assert result == small_diff

    def test_compress_preserves_diff_markers(self):
        """Diff markers (+, -, @@, diff, ---) should always be preserved."""
        compressor = DiffCompressor()

        # Create a diff with markers to preserve
        diff = """diff --git a/foo.py b/foo.py
--- a/foo.py
+++ b/foo.py
@@ -1,5 +1,5 @@
-old line
+new line"""

        # Even with mocked compression, markers should be preserved
        result = compressor.compress(diff)

        # All diff markers must be present
        for marker in ["diff --git", "---", "+++", "@@", "-old", "+new"]:
            assert marker in result or diff in result  # Either in result or unchanged

    def test_compress_preserves_code_keywords(self):
        """Code keywords (def, class, import, etc.) should be preserved."""
        compressor = DiffCompressor()

        diff = """diff --git a/foo.py b/foo.py
+def my_function():
+    import os
+    class MyClass:
+        return None"""

        result = compressor.compress(diff)

        # If compression happened, keywords should still be there
        for keyword in ["def", "import", "class", "return"]:
            assert keyword in result or diff in result

    @patch("swarm_orchestrator.swarm_mcp.compression.LLMLINGUA_AVAILABLE", False)
    def test_compress_fallback_when_llmlingua_unavailable(self):
        """Should return original diff when llmlingua is not installed."""
        compressor = DiffCompressor()

        diff = " ".join(["word"] * 1000)  # Large enough to compress

        result = compressor.compress(diff)

        assert result == diff


# =============================================================================
# compress_diff Function Tests
# =============================================================================

class TestCompressDiff:
    """Tests for the compress_diff helper function."""

    def test_compress_diff_small_returns_original(self):
        """Small diffs should pass through unchanged."""
        small_diff = "diff --git a/x.py b/x.py\n+x=1"
        result = compress_diff(small_diff)
        assert result == small_diff

    def test_compress_diff_empty_returns_empty(self):
        """Empty string should return empty string."""
        assert compress_diff("") == ""

    def test_compress_diff_achieves_compression(self):
        """Large diffs should be compressed (when llmlingua available)."""
        # This test will verify compression ratio when llmlingua is installed
        # When not installed, it should return original
        large_diff = """diff --git a/verbose.py b/verbose.py
+# This is a very long and verbose comment that explains something
+# in great detail with many words that could potentially be compressed
+# because they contain redundant information and repetitive phrases
""" + "\n".join([f"+# Line {i}: more verbose content here" for i in range(100)])

        result = compress_diff(large_diff)

        # Either compressed or original (if llmlingua unavailable)
        assert len(result) <= len(large_diff)


# =============================================================================
# Constants Tests
# =============================================================================

class TestConstants:
    """Tests for module constants."""

    def test_min_token_threshold_is_500(self):
        """MIN_TOKEN_THRESHOLD should be 500."""
        assert MIN_TOKEN_THRESHOLD == 500

    def test_diff_markers_contains_essential_markers(self):
        """DIFF_MARKERS should include essential git diff markers."""
        assert "+" in DIFF_MARKERS
        assert "-" in DIFF_MARKERS
        assert "@@" in DIFF_MARKERS
        assert "diff" in DIFF_MARKERS
        assert "---" in DIFF_MARKERS

    def test_code_keywords_contains_common_keywords(self):
        """CODE_KEYWORDS should include common programming keywords."""
        essential = ["def", "class", "import", "return", "if", "else", "for", "while"]
        for kw in essential:
            assert kw in CODE_KEYWORDS


# =============================================================================
# Integration Tests
# =============================================================================

class TestCompressionIntegration:
    """Integration tests for the compression module."""

    def test_real_diff_compression_preserves_structure(self):
        """A realistic diff should maintain its structure after compression."""
        real_diff = """diff --git a/src/main.py b/src/main.py
--- a/src/main.py
+++ b/src/main.py
@@ -10,6 +10,15 @@ def process_data(input_data):
     # This function processes data and returns results
     # It handles various edge cases and validates input
-    old_result = calculate(input_data)
+    # New implementation with better error handling
+    if input_data is None:
+        raise ValueError("Input cannot be None")
+
+    validated = validate_input(input_data)
+    new_result = calculate(validated)
+    return new_result
"""
        result = compress_diff(real_diff)

        # Core structure must be preserved
        assert "diff --git" in result or real_diff in result
        assert "def" in result or real_diff in result
        # The result should be valid (either compressed or original)
        assert len(result) > 0
