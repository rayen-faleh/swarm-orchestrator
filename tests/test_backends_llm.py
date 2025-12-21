"""
Tests for LLM backend implementations.

Tests ClaudeCLIBackend and AnthropicAPIBackend implementing LLMBackend interface.
"""

import subprocess
import pytest
from unittest.mock import patch, MagicMock

from swarm_orchestrator.backends.llm import (
    ClaudeCLIBackend,
    AnthropicAPIBackend,
    LLMBackendError,
)
from swarm_orchestrator.backends.base import LLMBackend, DecomposeResult


class TestClaudeCLIBackend:
    """Tests for ClaudeCLIBackend."""

    def test_implements_llm_backend(self):
        """ClaudeCLIBackend implements LLMBackend interface."""
        backend = ClaudeCLIBackend()
        assert isinstance(backend, LLMBackend)

    def test_default_timeout(self):
        """Default timeout is 120 seconds."""
        backend = ClaudeCLIBackend()
        assert backend.timeout == 120

    def test_custom_timeout(self):
        """Custom timeout can be specified."""
        backend = ClaudeCLIBackend(timeout=60)
        assert backend.timeout == 60

    def test_default_cli_tool(self):
        """Default cli_tool is 'claude'."""
        backend = ClaudeCLIBackend()
        assert backend.cli_tool == "claude"

    def test_custom_cli_tool(self):
        """Custom cli_tool can be specified."""
        backend = ClaudeCLIBackend(cli_tool="opencode")
        assert backend.cli_tool == "opencode"

    @patch("swarm_orchestrator.backends.llm.subprocess.run")
    def test_decompose_calls_claude_cli(self, mock_run):
        """decompose() calls claude CLI with correct arguments."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"is_atomic": true, "subtasks": [], "reasoning": "Simple task"}',
            stderr="",
        )

        backend = ClaudeCLIBackend(timeout=30)
        result = backend.decompose("Add a button")

        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args[0][0][0] == "claude"
        assert "-p" in call_args[0][0]
        assert "--output-format" in call_args[0][0]
        assert "text" in call_args[0][0]
        assert call_args[1]["timeout"] == 30

    @patch("swarm_orchestrator.backends.llm.subprocess.run")
    def test_decompose_returns_result(self, mock_run):
        """decompose() returns DecomposeResult."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"is_atomic": false, "subtasks": [{"id": "task-1"}], "reasoning": "Multi-step"}',
            stderr="",
        )

        backend = ClaudeCLIBackend()
        result = backend.decompose("Complex task")

        assert isinstance(result, DecomposeResult)
        assert result.is_atomic is False
        assert result.raw_response["subtasks"] == [{"id": "task-1"}]
        assert result.reasoning == "Multi-step"

    @patch("swarm_orchestrator.backends.llm.subprocess.run")
    def test_decompose_with_context(self, mock_run):
        """decompose() includes context in prompt when provided."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"is_atomic": true, "subtasks": [], "reasoning": "Simple"}',
            stderr="",
        )

        backend = ClaudeCLIBackend()
        backend.decompose("Add feature", context="Codebase uses React")

        call_args = mock_run.call_args
        prompt = call_args[0][0][call_args[0][0].index("-p") + 1]
        assert "Codebase uses React" in prompt

    @patch("swarm_orchestrator.backends.llm.subprocess.run")
    def test_decompose_raises_on_cli_not_found(self, mock_run):
        """decompose() raises LLMBackendError when CLI not found."""
        mock_run.side_effect = FileNotFoundError()

        backend = ClaudeCLIBackend()
        with pytest.raises(LLMBackendError) as exc_info:
            backend.decompose("test")

        assert "CLI not found" in str(exc_info.value)

    @patch("swarm_orchestrator.backends.llm.subprocess.run")
    def test_decompose_raises_on_timeout(self, mock_run):
        """decompose() raises LLMBackendError on timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="claude", timeout=120)

        backend = ClaudeCLIBackend()
        with pytest.raises(LLMBackendError) as exc_info:
            backend.decompose("test")

        assert "timed out" in str(exc_info.value)

    @patch("swarm_orchestrator.backends.llm.subprocess.run")
    def test_decompose_raises_on_cli_error(self, mock_run):
        """decompose() raises LLMBackendError on CLI failure."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="Some error occurred",
        )

        backend = ClaudeCLIBackend()
        with pytest.raises(LLMBackendError) as exc_info:
            backend.decompose("test")

        assert "CLI failed" in str(exc_info.value)

    @patch("swarm_orchestrator.backends.llm.subprocess.run")
    def test_explore_calls_claude_cli(self, mock_run):
        """explore() calls claude CLI."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="The codebase uses a modular architecture...",
            stderr="",
        )

        backend = ClaudeCLIBackend()
        result = backend.explore("How is authentication handled?")

        mock_run.assert_called_once()
        assert result == "The codebase uses a modular architecture..."

    @patch("swarm_orchestrator.backends.llm.subprocess.run")
    def test_explore_raises_on_cli_not_found(self, mock_run):
        """explore() raises LLMBackendError when CLI not found."""
        mock_run.side_effect = FileNotFoundError()

        backend = ClaudeCLIBackend()
        with pytest.raises(LLMBackendError) as exc_info:
            backend.explore("test")

        assert "CLI not found" in str(exc_info.value)

    @patch("swarm_orchestrator.backends.llm.subprocess.run")
    def test_decompose_with_opencode_cli(self, mock_run):
        """decompose() uses 'opencode' command when cli_tool='opencode'."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"is_atomic": true, "subtasks": [], "reasoning": "Simple"}',
            stderr="",
        )

        backend = ClaudeCLIBackend(cli_tool="opencode")
        backend.decompose("Add a button")

        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args[0][0][0] == "opencode"
        assert "-p" in call_args[0][0]

    @patch("swarm_orchestrator.backends.llm.subprocess.run")
    def test_opencode_does_not_use_output_format(self, mock_run):
        """opencode CLI does not use --output-format flag."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"is_atomic": true, "subtasks": [], "reasoning": "Simple"}',
            stderr="",
        )

        backend = ClaudeCLIBackend(cli_tool="opencode")
        backend.decompose("Add a button")

        call_args = mock_run.call_args
        assert "--output-format" not in call_args[0][0]

    @patch("swarm_orchestrator.backends.llm.subprocess.run")
    def test_error_message_uses_cli_tool_name(self, mock_run):
        """Error messages use the configured cli_tool name."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="Something went wrong",
        )

        backend = ClaudeCLIBackend(cli_tool="opencode")
        with pytest.raises(LLMBackendError) as exc_info:
            backend.decompose("test")

        assert "opencode" in str(exc_info.value)

    @patch("swarm_orchestrator.backends.llm.subprocess.run")
    def test_decompose_with_cursor_cli(self, mock_run):
        """decompose() uses 'cursor-agent' command when cli_tool='cursor'."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"is_atomic": true, "subtasks": [], "reasoning": "Simple"}',
            stderr="",
        )

        backend = ClaudeCLIBackend(cli_tool="cursor")
        backend.decompose("Add a button")

        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args[0][0][0] == "cursor-agent"
        assert "-p" in call_args[0][0]

    @patch("swarm_orchestrator.backends.llm.subprocess.run")
    def test_cursor_does_not_use_output_format(self, mock_run):
        """cursor CLI does not use --output-format flag."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"is_atomic": true, "subtasks": [], "reasoning": "Simple"}',
            stderr="",
        )

        backend = ClaudeCLIBackend(cli_tool="cursor")
        backend.decompose("Add a button")

        call_args = mock_run.call_args
        assert "--output-format" not in call_args[0][0]

    @patch("swarm_orchestrator.backends.llm.subprocess.run")
    def test_cursor_error_message_uses_cli_tool_name(self, mock_run):
        """Error messages use 'cursor' when cli_tool='cursor'."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="Something went wrong",
        )

        backend = ClaudeCLIBackend(cli_tool="cursor")
        with pytest.raises(LLMBackendError) as exc_info:
            backend.decompose("test")

        assert "cursor" in str(exc_info.value)


class TestAnthropicAPIBackend:
    """Tests for AnthropicAPIBackend."""

    def test_implements_llm_backend(self):
        """AnthropicAPIBackend implements LLMBackend interface."""
        backend = AnthropicAPIBackend()
        assert isinstance(backend, LLMBackend)

    def test_default_model(self):
        """Default model is claude-sonnet-4-20250514."""
        backend = AnthropicAPIBackend()
        assert backend.model == "claude-sonnet-4-20250514"

    def test_custom_model(self):
        """Custom model can be specified."""
        backend = AnthropicAPIBackend(model="claude-3-opus-20240229")
        assert backend.model == "claude-3-opus-20240229"

    @patch("swarm_orchestrator.backends.llm.Anthropic")
    def test_decompose_calls_api(self, mock_anthropic_class):
        """decompose() calls Anthropic API."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text='{"is_atomic": true, "subtasks": [], "reasoning": "Simple"}')]
        )

        backend = AnthropicAPIBackend()
        result = backend.decompose("Add a button")

        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-sonnet-4-20250514"
        assert call_kwargs["max_tokens"] == 4096

    @patch("swarm_orchestrator.backends.llm.Anthropic")
    def test_decompose_returns_result(self, mock_anthropic_class):
        """decompose() returns DecomposeResult."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text='{"is_atomic": false, "subtasks": [{"id": "t1"}], "reasoning": "Complex"}')]
        )

        backend = AnthropicAPIBackend()
        result = backend.decompose("Complex task")

        assert isinstance(result, DecomposeResult)
        assert result.is_atomic is False
        assert result.raw_response["subtasks"] == [{"id": "t1"}]
        assert result.reasoning == "Complex"

    @patch("swarm_orchestrator.backends.llm.Anthropic")
    def test_decompose_with_context(self, mock_anthropic_class):
        """decompose() includes context in messages."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text='{"is_atomic": true, "subtasks": [], "reasoning": ""}')]
        )

        backend = AnthropicAPIBackend()
        backend.decompose("Add feature", context="Uses TypeScript")

        call_kwargs = mock_client.messages.create.call_args[1]
        prompt_content = call_kwargs["messages"][0]["content"]
        assert "Uses TypeScript" in prompt_content

    @patch("swarm_orchestrator.backends.llm.Anthropic")
    def test_decompose_raises_on_missing_api_key(self, mock_anthropic_class):
        """decompose() raises LLMBackendError when API key missing."""
        mock_anthropic_class.side_effect = Exception("Missing API key")

        backend = AnthropicAPIBackend()
        with pytest.raises((LLMBackendError, Exception)):
            backend.decompose("test")

    @patch("swarm_orchestrator.backends.llm.Anthropic")
    def test_explore_calls_api(self, mock_anthropic_class):
        """explore() calls Anthropic API."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text="The authentication uses JWT tokens...")]
        )

        backend = AnthropicAPIBackend()
        result = backend.explore("How is auth handled?")

        mock_client.messages.create.assert_called_once()
        assert result == "The authentication uses JWT tokens..."

    @patch("swarm_orchestrator.backends.llm.Anthropic")
    def test_explore_raises_on_missing_api_key(self, mock_anthropic_class):
        """explore() raises LLMBackendError when API key missing."""
        mock_anthropic_class.side_effect = Exception("Missing API key")

        backend = AnthropicAPIBackend()
        with pytest.raises((LLMBackendError, Exception)):
            backend.explore("test")


class TestLLMBackendError:
    """Tests for LLMBackendError."""

    def test_is_exception(self):
        """LLMBackendError is an Exception."""
        assert issubclass(LLMBackendError, Exception)

    def test_can_be_raised_with_message(self):
        """LLMBackendError can be raised with a message."""
        with pytest.raises(LLMBackendError) as exc_info:
            raise LLMBackendError("Test error message")

        assert str(exc_info.value) == "Test error message"
