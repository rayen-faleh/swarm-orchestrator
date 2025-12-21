"""
LLMLingua-based diff compression with code-preserving defaults.

Provides compression for large diffs while preserving:
- Git diff markers (+, -, @@, diff, ---, +++)
- Code keywords (def, class, import, return, etc.)
"""

import re
from typing import Optional

# Try to import LLMLingua - gracefully handle if not installed
try:
    from llmlingua import PromptCompressor
    LLMLINGUA_AVAILABLE = True
except ImportError:
    LLMLINGUA_AVAILABLE = False
    PromptCompressor = None

# Minimum token count to trigger compression
MIN_TOKEN_THRESHOLD = 500

# Diff markers that must always be preserved
DIFF_MARKERS = frozenset([
    "+", "-", "@@", "diff", "---", "+++",
    "index", "new file", "deleted file", "rename",
])

# Code keywords to force-preserve during compression
CODE_KEYWORDS = frozenset([
    # Python
    "def", "class", "import", "from", "return", "yield",
    "if", "else", "elif", "for", "while", "try", "except",
    "finally", "with", "as", "raise", "assert", "pass",
    "break", "continue", "lambda", "async", "await",
    # Common across languages
    "function", "const", "let", "var", "export", "interface",
    "type", "struct", "enum", "impl", "fn", "pub", "mod",
    "package", "func", "go", "chan", "select", "defer",
])


def _estimate_tokens(text: str) -> int:
    """Estimate token count using simple word splitting."""
    return len(text.split())


def should_compress(text: str) -> bool:
    """Check if text exceeds the minimum token threshold for compression."""
    return _estimate_tokens(text) >= MIN_TOKEN_THRESHOLD


def _build_preserve_pattern() -> re.Pattern:
    """Build regex pattern for tokens to force-preserve."""
    # Escape special regex chars in markers
    escaped_markers = [re.escape(m) for m in DIFF_MARKERS]
    # Keywords can be matched as whole words
    keyword_patterns = [rf"\b{kw}\b" for kw in CODE_KEYWORDS]
    all_patterns = escaped_markers + keyword_patterns
    return re.compile("|".join(all_patterns))


class DiffCompressor:
    """
    LLMLingua-2 wrapper for diff compression with code-preserving defaults.

    Compresses large diffs while preserving:
    - Git diff structure markers
    - Code keywords for readability
    """

    def __init__(self, target_ratio: float = 0.4):
        """
        Initialize the compressor.

        Args:
            target_ratio: Target compression ratio (0.4 = 40% of original size)
        """
        self.target_ratio = target_ratio
        self.preserve_patterns = _build_preserve_pattern()
        self._compressor: Optional["PromptCompressor"] = None

    def _get_compressor(self) -> Optional["PromptCompressor"]:
        """Lazily initialize the LLMLingua compressor."""
        if not LLMLINGUA_AVAILABLE:
            return None

        if self._compressor is None:
            try:
                self._compressor = PromptCompressor(
                    model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
                    use_llmlingua2=True,
                )
            except Exception:
                # Failed to initialize - treat as unavailable
                return None

        return self._compressor

    def _extract_preserved_tokens(self, text: str) -> list[str]:
        """Extract tokens that must be preserved from the text."""
        return self.preserve_patterns.findall(text)

    def compress(self, diff: str) -> str:
        """
        Compress a diff using LLMLingua-2.

        Args:
            diff: The git diff text to compress

        Returns:
            Compressed diff, or original if below threshold or compression unavailable
        """
        if not diff:
            return diff

        # Skip compression for small diffs
        if not should_compress(diff):
            return diff

        compressor = self._get_compressor()
        if compressor is None:
            return diff  # Graceful fallback

        try:
            # Get tokens to force-preserve
            force_tokens = list(set(self._extract_preserved_tokens(diff)))

            result = compressor.compress_prompt(
                diff,
                rate=self.target_ratio,
                force_tokens=force_tokens,
                force_reserve_digit=True,  # Preserve line numbers
            )

            compressed = result.get("compressed_prompt", diff)
            return compressed if compressed else diff

        except Exception:
            # Any error during compression - return original
            return diff


# Module-level default compressor for convenience
_default_compressor: Optional[DiffCompressor] = None


def compress_diff(diff: str, target_ratio: float = 0.4) -> str:
    """
    Compress a diff using LLMLingua-2 with code-preserving defaults.

    This is a convenience function that uses a module-level compressor.

    Args:
        diff: The git diff text to compress
        target_ratio: Target compression ratio (0.4 = 40% of original)

    Returns:
        Compressed diff, or original if below threshold or compression unavailable
    """
    global _default_compressor

    if _default_compressor is None or _default_compressor.target_ratio != target_ratio:
        _default_compressor = DiffCompressor(target_ratio=target_ratio)

    return _default_compressor.compress(diff)
