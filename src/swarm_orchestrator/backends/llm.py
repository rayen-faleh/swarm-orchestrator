"""
LLM backend implementations for task decomposition and exploration.

Provides ClaudeCLIBackend (using claude CLI) and AnthropicAPIBackend (using API).
"""

import json
import re
import subprocess

from anthropic import Anthropic, AuthenticationError

from .base import LLMBackend, DecomposeResult


class LLMBackendError(Exception):
    """Raised when an LLM backend operation fails."""


class ClaudeCLIBackend(LLMBackend):
    """
    LLM backend using Claude CLI.

    Uses your existing Claude Code authentication (Max/Pro subscription).
    No API key needed.
    """

    def __init__(self, timeout: int = 120):
        """
        Initialize the CLI backend.

        Args:
            timeout: Timeout in seconds for CLI calls (default: 120)
        """
        self.timeout = timeout

    def decompose(self, query: str, context: str | None = None) -> DecomposeResult:
        prompt = self._build_prompt(query, context)
        text = self._call_cli(prompt)
        return self._parse_decompose_response(text)

    def explore(self, query: str) -> str:
        return self._call_cli(query)

    def _build_prompt(self, query: str, context: str | None) -> str:
        prompt = query
        if context:
            prompt = f"{query}\n\nContext:\n{context}"
        return prompt

    def _call_cli(self, prompt: str) -> str:
        try:
            result = subprocess.run(
                ["claude", "-p", prompt, "--output-format", "text"],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            if result.returncode != 0:
                raise LLMBackendError(f"Claude CLI failed: {result.stderr or 'Unknown error'}")
            return result.stdout
        except FileNotFoundError:
            raise LLMBackendError(
                "Claude CLI not found. Install Claude Code: https://claude.ai/download"
            )
        except subprocess.TimeoutExpired:
            raise LLMBackendError(f"Claude CLI timed out after {self.timeout} seconds")

    def _parse_decompose_response(self, text: str) -> DecomposeResult:
        json_match = re.search(r"\{[\s\S]*\}", text)
        if not json_match:
            raise LLMBackendError(f"Could not parse JSON from response: {text[:200]}")
        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            raise LLMBackendError(f"Invalid JSON in response: {e}")
        return DecomposeResult(
            is_atomic=data.get("is_atomic", True),
            raw_response=data,
            reasoning=data.get("reasoning"),
        )


class AnthropicAPIBackend(LLMBackend):
    """
    LLM backend using Anthropic API directly.

    Requires ANTHROPIC_API_KEY environment variable.
    """

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        """
        Initialize the API backend.

        Args:
            model: Model to use (default: claude-sonnet-4-20250514)
        """
        self.model = model
        self._client = None

    def _get_client(self) -> Anthropic:
        if self._client is None:
            try:
                self._client = Anthropic()
            except AuthenticationError as e:
                raise LLMBackendError(f"Anthropic API key missing or invalid: {e}")
        return self._client

    def decompose(self, query: str, context: str | None = None) -> DecomposeResult:
        prompt = self._build_prompt(query, context)
        text = self._call_api(prompt)
        return self._parse_decompose_response(text)

    def explore(self, query: str) -> str:
        return self._call_api(query)

    def _build_prompt(self, query: str, context: str | None) -> str:
        prompt = query
        if context:
            prompt = f"{query}\n\nContext:\n{context}"
        return prompt

    def _call_api(self, prompt: str) -> str:
        client = self._get_client()
        response = client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def _parse_decompose_response(self, text: str) -> DecomposeResult:
        json_match = re.search(r"\{[\s\S]*\}", text)
        if not json_match:
            raise LLMBackendError(f"Could not parse JSON from response: {text[:200]}")
        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            raise LLMBackendError(f"Invalid JSON in response: {e}")
        return DecomposeResult(
            is_atomic=data.get("is_atomic", True),
            raw_response=data,
            reasoning=data.get("reasoning"),
        )
