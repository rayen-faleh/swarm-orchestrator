"""
Tests for CLI session management commands.

Tests 'swarm sessions', 'swarm diff <session>', and 'swarm merge <session>' commands.
"""

import pytest
from unittest.mock import patch, MagicMock
from click.testing import CliRunner

from swarm_orchestrator.cli import main
from swarm_orchestrator.backends.base import SessionInfo, DiffResult


class TestSessionsCommand:
    """Tests for the 'swarm sessions' command."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_sessions_lists_active_sessions(self, runner):
        """Should list all active sessions with name, status, and branch."""
        mock_backend = MagicMock()
        mock_backend.list_sessions.return_value = [
            SessionInfo(name="task-agent-0", status="running", branch="git-native/task-agent-0"),
            SessionInfo(name="task-agent-1", status="reviewed", branch="git-native/task-agent-1"),
        ]

        with patch("swarm_orchestrator.cli._get_worktree_backend", return_value=mock_backend):
            result = runner.invoke(main, ["sessions"])

        assert result.exit_code == 0
        assert "task-agent-0" in result.output
        assert "task-agent-1" in result.output
        assert "running" in result.output
        assert "reviewed" in result.output

    def test_sessions_shows_empty_message(self, runner):
        """Should show message when no sessions exist."""
        mock_backend = MagicMock()
        mock_backend.list_sessions.return_value = []

        with patch("swarm_orchestrator.cli._get_worktree_backend", return_value=mock_backend):
            result = runner.invoke(main, ["sessions"])

        assert result.exit_code == 0
        assert "no" in result.output.lower() and "session" in result.output.lower()

    def test_sessions_filter_active(self, runner):
        """Should filter sessions when --active flag is used."""
        mock_backend = MagicMock()
        mock_backend.list_sessions.return_value = [
            SessionInfo(name="task-agent-0", status="running", branch="branch"),
        ]

        with patch("swarm_orchestrator.cli._get_worktree_backend", return_value=mock_backend):
            result = runner.invoke(main, ["sessions", "--active"])

        assert result.exit_code == 0
        mock_backend.list_sessions.assert_called_with("active")

    def test_sessions_filter_reviewed(self, runner):
        """Should filter sessions when --reviewed flag is used."""
        mock_backend = MagicMock()
        mock_backend.list_sessions.return_value = []

        with patch("swarm_orchestrator.cli._get_worktree_backend", return_value=mock_backend):
            result = runner.invoke(main, ["sessions", "--reviewed"])

        assert result.exit_code == 0
        mock_backend.list_sessions.assert_called_with("reviewed")


class TestDiffCommand:
    """Tests for the 'swarm diff <session>' command."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_diff_shows_session_diff(self, runner):
        """Should show the diff for a session."""
        mock_backend = MagicMock()
        mock_backend.get_session.return_value = SessionInfo(
            name="task-agent-0", status="running", branch="branch"
        )
        mock_backend.get_diff.return_value = DiffResult(
            files=["src/foo.py", "tests/test_foo.py"],
            content="+def hello(): pass\n-def old(): pass",
        )

        with patch("swarm_orchestrator.cli._get_worktree_backend", return_value=mock_backend):
            result = runner.invoke(main, ["diff", "task-agent-0"])

        assert result.exit_code == 0
        assert "src/foo.py" in result.output
        assert "tests/test_foo.py" in result.output
        assert "+def hello()" in result.output

    def test_diff_invalid_session(self, runner):
        """Should show error for invalid session name."""
        mock_backend = MagicMock()
        mock_backend.get_session.return_value = None

        with patch("swarm_orchestrator.cli._get_worktree_backend", return_value=mock_backend):
            result = runner.invoke(main, ["diff", "nonexistent"])

        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "error" in result.output.lower()

    def test_diff_shows_file_count(self, runner):
        """Should show the number of files changed."""
        mock_backend = MagicMock()
        mock_backend.get_session.return_value = SessionInfo(
            name="task-agent-0", status="running", branch="branch"
        )
        mock_backend.get_diff.return_value = DiffResult(
            files=["a.py", "b.py", "c.py"],
            content="+content",
        )

        with patch("swarm_orchestrator.cli._get_worktree_backend", return_value=mock_backend):
            result = runner.invoke(main, ["diff", "task-agent-0"])

        assert result.exit_code == 0
        assert "3" in result.output  # 3 files


class TestMergeCommand:
    """Tests for the 'swarm merge <session>' command."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_merge_merges_session(self, runner):
        """Should merge a session to parent branch."""
        mock_backend = MagicMock()
        mock_backend.get_session.return_value = SessionInfo(
            name="task-agent-0", status="reviewed", branch="branch", ready_to_merge=True
        )

        with patch("swarm_orchestrator.cli._get_worktree_backend", return_value=mock_backend):
            result = runner.invoke(main, ["merge", "task-agent-0"])

        assert result.exit_code == 0
        mock_backend.merge_session.assert_called_once()
        assert "merged" in result.output.lower() or "success" in result.output.lower()

    def test_merge_invalid_session(self, runner):
        """Should show error for invalid session name."""
        mock_backend = MagicMock()
        mock_backend.get_session.return_value = None

        with patch("swarm_orchestrator.cli._get_worktree_backend", return_value=mock_backend):
            result = runner.invoke(main, ["merge", "nonexistent"])

        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "error" in result.output.lower()

    def test_merge_with_message(self, runner):
        """Should use custom commit message when provided."""
        mock_backend = MagicMock()
        mock_backend.get_session.return_value = SessionInfo(
            name="task-agent-0", status="reviewed", branch="branch", ready_to_merge=True
        )

        with patch("swarm_orchestrator.cli._get_worktree_backend", return_value=mock_backend):
            result = runner.invoke(main, ["merge", "task-agent-0", "-m", "Custom message"])

        assert result.exit_code == 0
        mock_backend.merge_session.assert_called_with("task-agent-0", "Custom message")

    def test_merge_uses_default_message(self, runner):
        """Should use default commit message when none provided."""
        mock_backend = MagicMock()
        mock_backend.get_session.return_value = SessionInfo(
            name="task-agent-0", status="reviewed", branch="branch", ready_to_merge=True
        )

        with patch("swarm_orchestrator.cli._get_worktree_backend", return_value=mock_backend):
            result = runner.invoke(main, ["merge", "task-agent-0"])

        assert result.exit_code == 0
        # Check default message contains session name
        call_args = mock_backend.merge_session.call_args
        assert "task-agent-0" in call_args[0][1]

    def test_merge_handles_backend_error(self, runner):
        """Should handle merge errors gracefully."""
        mock_backend = MagicMock()
        mock_backend.get_session.return_value = SessionInfo(
            name="task-agent-0", status="running", branch="branch"
        )
        mock_backend.merge_session.side_effect = Exception("Merge conflict")

        with patch("swarm_orchestrator.cli._get_worktree_backend", return_value=mock_backend):
            result = runner.invoke(main, ["merge", "task-agent-0"])

        assert result.exit_code != 0
        assert "error" in result.output.lower()


class TestStopCommand:
    """Tests for the 'swarm stop <session>' command."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_stop_stops_running_session(self, runner):
        """Should stop a running session."""
        mock_agent_backend = MagicMock()
        mock_agent_backend.stop_agent.return_value = True

        with patch("swarm_orchestrator.cli._get_agent_backend", return_value=mock_agent_backend):
            result = runner.invoke(main, ["stop", "task-agent-0"])

        assert result.exit_code == 0
        mock_agent_backend.stop_agent.assert_called_once_with("task-agent-0")
        assert "stopped" in result.output.lower()

    def test_stop_shows_message_for_not_running(self, runner):
        """Should show message when session not running."""
        mock_agent_backend = MagicMock()
        mock_agent_backend.stop_agent.return_value = False

        with patch("swarm_orchestrator.cli._get_agent_backend", return_value=mock_agent_backend):
            result = runner.invoke(main, ["stop", "task-agent-0"])

        assert result.exit_code == 0
        assert "not running" in result.output.lower() or "not found" in result.output.lower()

    def test_stop_errors_when_no_agent_backend(self, runner):
        """Should error when agent backend doesn't support stop."""
        with patch("swarm_orchestrator.cli._get_agent_backend", return_value=None):
            result = runner.invoke(main, ["stop", "task-agent-0"])

        assert result.exit_code != 0
        assert "error" in result.output.lower()


class TestDeleteCommand:
    """Tests for the 'swarm delete <session>' command."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_delete_deletes_session(self, runner):
        """Should delete a session."""
        mock_worktree_backend = MagicMock()
        mock_worktree_backend.get_session.return_value = SessionInfo(
            name="task-agent-0", status="running", branch="branch"
        )
        mock_agent_backend = MagicMock()

        with patch("swarm_orchestrator.cli._get_worktree_backend", return_value=mock_worktree_backend):
            with patch("swarm_orchestrator.cli._get_agent_backend", return_value=mock_agent_backend):
                result = runner.invoke(main, ["delete", "task-agent-0"])

        assert result.exit_code == 0
        mock_worktree_backend.delete_session.assert_called_once_with("task-agent-0", force=False)
        assert "deleted" in result.output.lower()

    def test_delete_stops_agent_first(self, runner):
        """Should stop agent before deleting session."""
        mock_worktree_backend = MagicMock()
        mock_worktree_backend.get_session.return_value = SessionInfo(
            name="task-agent-0", status="running", branch="branch"
        )
        mock_agent_backend = MagicMock()

        with patch("swarm_orchestrator.cli._get_worktree_backend", return_value=mock_worktree_backend):
            with patch("swarm_orchestrator.cli._get_agent_backend", return_value=mock_agent_backend):
                result = runner.invoke(main, ["delete", "task-agent-0"])

        assert result.exit_code == 0
        mock_agent_backend.stop_agent.assert_called_once_with("task-agent-0")

    def test_delete_invalid_session(self, runner):
        """Should error for invalid session name."""
        mock_worktree_backend = MagicMock()
        mock_worktree_backend.get_session.return_value = None
        mock_agent_backend = MagicMock()

        with patch("swarm_orchestrator.cli._get_worktree_backend", return_value=mock_worktree_backend):
            with patch("swarm_orchestrator.cli._get_agent_backend", return_value=mock_agent_backend):
                result = runner.invoke(main, ["delete", "nonexistent"])

        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "error" in result.output.lower()

    def test_delete_with_force_flag(self, runner):
        """Should pass force flag to backend."""
        mock_worktree_backend = MagicMock()
        mock_worktree_backend.get_session.return_value = SessionInfo(
            name="task-agent-0", status="running", branch="branch"
        )
        mock_agent_backend = MagicMock()

        with patch("swarm_orchestrator.cli._get_worktree_backend", return_value=mock_worktree_backend):
            with patch("swarm_orchestrator.cli._get_agent_backend", return_value=mock_agent_backend):
                result = runner.invoke(main, ["delete", "--force", "task-agent-0"])

        assert result.exit_code == 0
        mock_worktree_backend.delete_session.assert_called_once_with("task-agent-0", force=True)
