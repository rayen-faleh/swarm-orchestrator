"""
Tests for GitNativeAgentBackend implementation.

Tests GitNativeAgentBackend implementation against the abstract AgentBackend interface.
"""

import subprocess
import pytest
from unittest.mock import MagicMock, patch

from swarm_orchestrator.backends.base import AgentBackend, AgentStatus
from swarm_orchestrator.backends.git_native import GitNativeAgentBackend


class TestGitNativeAgentBackend:
    """Tests for the GitNativeAgentBackend implementation."""

    @pytest.fixture
    def mock_popen(self):
        """Create a mock Popen class."""
        mock_process = MagicMock()
        mock_process.poll.return_value = None  # Still running
        mock_process.pid = 12345
        with patch("subprocess.Popen", return_value=mock_process) as mock:
            yield mock

    @pytest.fixture
    def backend(self, tmp_path):
        """Create a backend instance with test worktree base."""
        return GitNativeAgentBackend(worktree_base=tmp_path)

    def test_implements_agent_backend(self, backend):
        """GitNativeAgentBackend implements AgentBackend interface."""
        assert isinstance(backend, AgentBackend)

    def test_spawn_agent_writes_prompt_file(self, backend, mock_popen, tmp_path):
        """spawn_agent writes prompt to .swarm-prompt.md file."""
        worktree_path = tmp_path / "test-session"
        worktree_path.mkdir(parents=True)
        prompt = "Test prompt content"

        backend.spawn_agent("test-session", prompt)

        # Verify prompt file was created
        prompt_file = worktree_path / ".swarm-prompt.md"
        assert prompt_file.exists()
        assert prompt_file.read_text() == prompt

    def test_spawn_agent_runs_claude_cli(self, backend, mock_popen, tmp_path):
        """spawn_agent runs claude CLI in the worktree directory."""
        worktree_path = tmp_path / "test-session"
        worktree_path.mkdir(parents=True)

        backend.spawn_agent("test-session", "prompt")

        mock_popen.assert_called_once()
        call_args = mock_popen.call_args
        cmd = call_args[0][0]

        assert cmd[0] == "claude"
        assert "-p" in cmd
        assert "--dangerously-skip-permissions" in cmd

    def test_spawn_agent_runs_in_worktree_directory(self, backend, mock_popen, tmp_path):
        """spawn_agent sets cwd to worktree path."""
        worktree_path = tmp_path / "test-session"
        worktree_path.mkdir(parents=True)

        backend.spawn_agent("test-session", "prompt")

        call_args = mock_popen.call_args
        assert call_args[1]["cwd"] == worktree_path

    def test_spawn_agent_returns_agent_id(self, backend, mock_popen, tmp_path):
        """spawn_agent returns the session name as agent_id."""
        worktree_path = tmp_path / "test-session"
        worktree_path.mkdir(parents=True)

        result = backend.spawn_agent("test-session", "prompt")

        assert result == "test-session"

    def test_spawn_agent_stores_process(self, backend, mock_popen, tmp_path):
        """spawn_agent stores process in internal dict for tracking."""
        worktree_path = tmp_path / "test-session"
        worktree_path.mkdir(parents=True)

        backend.spawn_agent("test-session", "prompt")

        assert "test-session" in backend._processes

    def test_get_status_running(self, backend):
        """get_status returns is_finished=False when process is running."""
        mock_process = MagicMock()
        mock_process.poll.return_value = None  # Still running
        backend._processes["test-session"] = mock_process

        result = backend.get_status("test-session")

        assert isinstance(result, AgentStatus)
        assert result.agent_id == "test-session"
        assert result.is_finished is False

    def test_get_status_finished(self, backend):
        """get_status returns is_finished=True when process completed."""
        mock_process = MagicMock()
        mock_process.poll.return_value = 0  # Completed
        backend._processes["test-session"] = mock_process

        result = backend.get_status("test-session")

        assert result.is_finished is True

    def test_get_status_unknown_agent(self, backend):
        """get_status returns is_finished=False for unknown agent."""
        result = backend.get_status("unknown-agent")

        assert result.agent_id == "unknown-agent"
        assert result.is_finished is False

    def test_wait_for_completion_with_timeout(self, backend):
        """wait_for_completion uses communicate() with timeout."""
        mock_process = MagicMock()
        mock_process.communicate.return_value = (b"stdout", b"stderr")
        mock_process.returncode = 0
        backend._processes["test-session"] = mock_process

        result = backend.wait_for_completion(["test-session"], timeout=60)

        mock_process.communicate.assert_called_once_with(timeout=60)
        assert "test-session" in result
        assert result["test-session"].is_finished is True

    def test_wait_for_completion_handles_timeout_exception(self, backend):
        """wait_for_completion handles subprocess.TimeoutExpired."""
        mock_process = MagicMock()
        mock_process.communicate.side_effect = subprocess.TimeoutExpired("cmd", 60)
        mock_process.poll.return_value = None
        backend._processes["test-session"] = mock_process

        result = backend.wait_for_completion(["test-session"], timeout=60)

        assert result["test-session"].is_finished is False

    def test_wait_for_completion_multiple_agents(self, backend):
        """wait_for_completion handles multiple agents."""
        mock_process1 = MagicMock()
        mock_process1.communicate.return_value = (b"out1", b"")
        mock_process1.returncode = 0

        mock_process2 = MagicMock()
        mock_process2.communicate.return_value = (b"out2", b"")
        mock_process2.returncode = 0

        backend._processes["agent-1"] = mock_process1
        backend._processes["agent-2"] = mock_process2

        result = backend.wait_for_completion(["agent-1", "agent-2"], timeout=120)

        assert len(result) == 2
        assert result["agent-1"].is_finished is True
        assert result["agent-2"].is_finished is True

    def test_send_message_writes_to_stdin(self, backend):
        """send_message writes to process stdin."""
        mock_process = MagicMock()
        mock_process.stdin = MagicMock()
        mock_process.poll.return_value = None  # Still running
        backend._processes["test-session"] = mock_process

        backend.send_message("test-session", "test message")

        mock_process.stdin.write.assert_called()
        mock_process.stdin.flush.assert_called()

    def test_send_message_raises_for_unknown_agent(self, backend):
        """send_message raises ValueError for unknown agent."""
        with pytest.raises(ValueError, match="not found"):
            backend.send_message("unknown-agent", "message")

    def test_send_message_raises_for_finished_agent(self, backend):
        """send_message raises ValueError for finished agent."""
        mock_process = MagicMock()
        mock_process.poll.return_value = 0  # Finished
        backend._processes["test-session"] = mock_process

        with pytest.raises(ValueError, match="not running"):
            backend.send_message("test-session", "message")


class TestGitNativeAgentBackendDefaults:
    """Tests for default configuration."""

    def test_processes_dict_initialized_empty(self):
        """Backend initializes with empty processes dict."""
        backend = GitNativeAgentBackend()
        assert backend._processes == {}

    def test_default_worktree_base(self):
        """Backend uses .swarm/worktrees as default worktree base."""
        backend = GitNativeAgentBackend()
        assert ".swarm" in str(backend._worktree_base)
        assert "worktrees" in str(backend._worktree_base)


class TestGitNativeAgentBackendWorktreePath:
    """Tests for worktree path resolution."""

    def test_worktree_path_uses_configured_base(self, tmp_path):
        """Worktree path uses the configured worktree_base."""
        worktree_base = tmp_path / "custom" / "worktrees"
        worktree_base.mkdir(parents=True)
        session_dir = worktree_base / "my-session"
        session_dir.mkdir()

        backend = GitNativeAgentBackend(worktree_base=worktree_base)

        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock(pid=123)
            backend.spawn_agent("my-session", "prompt")

            call_args = mock_popen.call_args
            assert call_args[1]["cwd"] == session_dir
