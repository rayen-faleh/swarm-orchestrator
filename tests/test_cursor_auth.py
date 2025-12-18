"""
Tests for Cursor authentication helper module.

Tests cursor_auth.py functions for checking authentication status
and triggering browser-based login.
"""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from swarm_orchestrator.backends.cursor_auth import (
    is_authenticated,
    login,
    check_and_prompt_login,
)


class TestIsAuthenticated:
    """Tests for is_authenticated function."""

    def test_returns_true_when_api_key_set(self):
        """API key environment variable takes precedence."""
        with patch.dict("os.environ", {"CURSOR_API_KEY": "test-key"}):
            assert is_authenticated() is True

    def test_returns_true_when_cursor_agent_status_succeeds(self):
        """Returns True when cursor-agent status exits with 0."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)
                assert is_authenticated() is True
                mock_run.assert_called_once()
                # Verify correct command
                args = mock_run.call_args[0][0]
                assert "cursor-agent" in args
                assert "status" in args

    def test_returns_false_when_cursor_agent_status_fails(self):
        """Returns False when cursor-agent status exits with non-zero."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=1)
                assert is_authenticated() is False

    def test_returns_false_when_cursor_agent_not_found(self):
        """Returns False when cursor-agent command not found."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("subprocess.run") as mock_run:
                mock_run.side_effect = FileNotFoundError("cursor-agent not found")
                assert is_authenticated() is False


class TestLogin:
    """Tests for login function."""

    def test_invokes_cursor_agent_login(self):
        """login() invokes cursor-agent login command."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = login()
            assert result is True
            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            assert "cursor-agent" in args
            assert "login" in args

    def test_returns_false_on_login_failure(self):
        """login() returns False when login command fails."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            result = login()
            assert result is False

    def test_returns_false_when_cursor_agent_not_found(self):
        """login() returns False when cursor-agent not found."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("cursor-agent not found")
            result = login()
            assert result is False


class TestCheckAndPromptLogin:
    """Tests for check_and_prompt_login function."""

    def test_returns_true_if_already_authenticated(self):
        """Does not prompt if already authenticated."""
        with patch("swarm_orchestrator.backends.cursor_auth.is_authenticated", return_value=True):
            with patch("swarm_orchestrator.backends.cursor_auth.login") as mock_login:
                result = check_and_prompt_login()
                assert result is True
                mock_login.assert_not_called()

    def test_prompts_and_logs_in_when_not_authenticated(self):
        """Prompts user and runs login when not authenticated."""
        with patch("swarm_orchestrator.backends.cursor_auth.is_authenticated", return_value=False):
            with patch("swarm_orchestrator.backends.cursor_auth.login", return_value=True) as mock_login:
                with patch("builtins.input", return_value="y"):
                    result = check_and_prompt_login()
                    assert result is True
                    mock_login.assert_called_once()

    def test_returns_false_when_user_declines(self):
        """Returns False when user declines to login."""
        with patch("swarm_orchestrator.backends.cursor_auth.is_authenticated", return_value=False):
            with patch("swarm_orchestrator.backends.cursor_auth.login") as mock_login:
                with patch("builtins.input", return_value="n"):
                    result = check_and_prompt_login()
                    assert result is False
                    mock_login.assert_not_called()

    def test_returns_false_when_login_fails(self):
        """Returns False when login fails after user confirms."""
        with patch("swarm_orchestrator.backends.cursor_auth.is_authenticated", return_value=False):
            with patch("swarm_orchestrator.backends.cursor_auth.login", return_value=False):
                with patch("builtins.input", return_value="y"):
                    result = check_and_prompt_login()
                    assert result is False

    def test_skips_prompt_when_auto_mode(self):
        """In auto mode, attempts login without prompting."""
        with patch("swarm_orchestrator.backends.cursor_auth.is_authenticated", return_value=False):
            with patch("swarm_orchestrator.backends.cursor_auth.login", return_value=True) as mock_login:
                with patch("builtins.input") as mock_input:
                    result = check_and_prompt_login(auto=True)
                    assert result is True
                    mock_login.assert_called_once()
                    mock_input.assert_not_called()
