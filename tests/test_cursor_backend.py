"""
Tests for Cursor CLI backend implementation.

Tests CursorCLIAgentBackend implementation against the abstract AgentBackend interface.
"""

import os
import pytest
from unittest.mock import MagicMock, patch, mock_open

from swarm_orchestrator.backends.base import AgentBackend, AgentStatus
from swarm_orchestrator.backends.cursor import CursorCLIAgentBackend


class TestCursorCLIAgentBackend:
    """Tests for the CursorCLIAgentBackend implementation."""

    @pytest.fixture
    def mock_popen(self):
        """Create a mock Popen class."""
        mock_process = MagicMock()
        mock_process.poll.return_value = None  # Still running
        mock_process.pid = 12345
        with patch("subprocess.Popen", return_value=mock_process) as mock:
            yield mock

    @pytest.fixture
    def backend(self):
        """Create a backend instance."""
        return CursorCLIAgentBackend()

    def test_implements_agent_backend(self, backend):
        """CursorCLIAgentBackend implements AgentBackend interface."""
        assert isinstance(backend, AgentBackend)

    def test_spawn_agent_creates_prompt_file(self, backend, mock_popen, tmp_path):
        """spawn_agent writes prompt to .swarm-prompt.md file."""
        worktree_path = str(tmp_path)
        prompt = "Test prompt content"

        with patch.object(backend, "_get_worktree_path", return_value=worktree_path):
            with patch("swarm_orchestrator.backends.cursor.is_authenticated", return_value=True):
                backend.spawn_agent("test-session", prompt)

        # Verify prompt file was created
        prompt_file = tmp_path / ".swarm-prompt.md"
        assert prompt_file.exists()
        assert prompt_file.read_text() == prompt

    def test_spawn_agent_runs_cursor_with_correct_flags(self, backend, mock_popen, tmp_path):
        """spawn_agent runs cursor-agent with -p, --force, --output-format stream-json."""
        worktree_path = str(tmp_path)

        with patch.object(backend, "_get_worktree_path", return_value=worktree_path):
            with patch("swarm_orchestrator.backends.cursor.is_authenticated", return_value=True):
                backend.spawn_agent("test-session", "prompt")

        mock_popen.assert_called_once()
        call_args = mock_popen.call_args
        cmd = call_args[0][0]

        assert "cursor-agent" in cmd[0]
        assert "-p" in cmd
        assert ".swarm-prompt.md" in cmd
        assert "--force" in cmd
        assert "--output-format" in cmd
        assert "stream-json" in cmd

    def test_spawn_agent_passes_cursor_api_key(self, backend, mock_popen, tmp_path):
        """spawn_agent passes CURSOR_API_KEY environment variable."""
        worktree_path = str(tmp_path)

        with patch.object(backend, "_get_worktree_path", return_value=worktree_path):
            with patch.dict(os.environ, {"CURSOR_API_KEY": "test-key"}):
                with patch("swarm_orchestrator.backends.cursor.is_authenticated", return_value=True):
                    backend.spawn_agent("test-session", "prompt")

        call_args = mock_popen.call_args
        env = call_args[1].get("env", {})
        assert env.get("CURSOR_API_KEY") == "test-key"

    def test_spawn_agent_runs_in_worktree_directory(self, backend, mock_popen, tmp_path):
        """spawn_agent sets cwd to worktree path."""
        worktree_path = str(tmp_path)

        with patch.object(backend, "_get_worktree_path", return_value=worktree_path):
            with patch("swarm_orchestrator.backends.cursor.is_authenticated", return_value=True):
                backend.spawn_agent("test-session", "prompt")

        call_args = mock_popen.call_args
        assert call_args[1]["cwd"] == worktree_path

    def test_spawn_agent_returns_agent_id(self, backend, mock_popen, tmp_path):
        """spawn_agent returns the session name as agent_id."""
        with patch.object(backend, "_get_worktree_path", return_value=str(tmp_path)):
            with patch("swarm_orchestrator.backends.cursor.is_authenticated", return_value=True):
                result = backend.spawn_agent("test-session", "prompt")

        assert result == "test-session"

    def test_spawn_agent_stores_process(self, backend, mock_popen, tmp_path):
        """spawn_agent stores process in internal dict for tracking."""
        with patch.object(backend, "_get_worktree_path", return_value=str(tmp_path)):
            with patch("swarm_orchestrator.backends.cursor.is_authenticated", return_value=True):
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
        assert result.implementation is None

    def test_get_status_finished(self, backend):
        """get_status returns is_finished=True when process completed."""
        mock_process = MagicMock()
        mock_process.poll.return_value = 0  # Completed
        mock_process.stdout = MagicMock()
        mock_process.stdout.read.return_value = "output"
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
        import subprocess

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

    def test_send_message_raises_not_implemented(self, backend):
        """send_message raises NotImplementedError as documented."""
        with pytest.raises(NotImplementedError):
            backend.send_message("test-session", "message")

    def test_stop_agent_running_process(self, backend):
        """stop_agent terminates a running process."""
        mock_process = MagicMock()
        mock_process.poll.return_value = None  # Still running
        backend._processes["test-session"] = mock_process

        result = backend.stop_agent("test-session")

        assert result is True
        mock_process.terminate.assert_called_once()
        mock_process.wait.assert_called_once_with(timeout=5)
        assert "test-session" not in backend._processes

    def test_stop_agent_uses_sigkill_on_timeout(self, backend):
        """stop_agent falls back to SIGKILL when terminate times out."""
        import subprocess
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_process.wait.side_effect = [subprocess.TimeoutExpired("cmd", 5), None]
        backend._processes["test-session"] = mock_process

        result = backend.stop_agent("test-session")

        assert result is True
        mock_process.terminate.assert_called_once()
        mock_process.kill.assert_called_once()
        assert "test-session" not in backend._processes

    def test_stop_agent_returns_false_for_unknown_session(self, backend):
        """stop_agent returns False for unknown session."""
        result = backend.stop_agent("unknown-session")

        assert result is False

    def test_stop_agent_returns_false_for_already_stopped(self, backend):
        """stop_agent returns False when process already finished."""
        mock_process = MagicMock()
        mock_process.poll.return_value = 0  # Already finished
        backend._processes["test-session"] = mock_process

        result = backend.stop_agent("test-session")

        assert result is False
        mock_process.terminate.assert_not_called()
        assert "test-session" not in backend._processes


class TestCursorCLIAgentBackendDefaults:
    """Tests for default configuration."""

    def test_processes_dict_initialized_empty(self):
        """Backend initializes with empty processes dict."""
        backend = CursorCLIAgentBackend()
        assert backend._processes == {}


class TestCursorCLIAgentBackendAuth:
    """Tests for authentication checking in CursorCLIAgentBackend."""

    @pytest.fixture
    def backend(self):
        """Create a backend instance."""
        return CursorCLIAgentBackend()

    def test_spawn_agent_raises_error_when_not_authenticated(self, backend, tmp_path):
        """spawn_agent raises clear error when not authenticated."""
        with patch.object(backend, "_get_worktree_path", return_value=str(tmp_path)):
            with patch("swarm_orchestrator.backends.cursor.is_authenticated", return_value=False):
                with pytest.raises(RuntimeError) as exc_info:
                    backend.spawn_agent("test-session", "prompt")
                assert "not authenticated" in str(exc_info.value).lower()

    def test_spawn_agent_error_mentions_both_auth_options(self, backend, tmp_path):
        """Error message mentions both 'swarm cursor login' and CURSOR_API_KEY."""
        with patch.object(backend, "_get_worktree_path", return_value=str(tmp_path)):
            with patch("swarm_orchestrator.backends.cursor.is_authenticated", return_value=False):
                with pytest.raises(RuntimeError) as exc_info:
                    backend.spawn_agent("test-session", "prompt")
                error_msg = str(exc_info.value)
                assert "CURSOR_API_KEY" in error_msg
                assert "cursor" in error_msg.lower() and "login" in error_msg.lower()

    def test_spawn_agent_proceeds_when_cursor_api_key_set(self, backend, tmp_path):
        """spawn_agent proceeds when CURSOR_API_KEY environment variable is set."""
        with patch.object(backend, "_get_worktree_path", return_value=str(tmp_path)):
            with patch.dict(os.environ, {"CURSOR_API_KEY": "test-key"}):
                with patch("swarm_orchestrator.backends.cursor.is_authenticated", return_value=True):
                    with patch("subprocess.Popen") as mock_popen:
                        mock_popen.return_value = MagicMock(pid=123)
                        result = backend.spawn_agent("test-session", "prompt")
                        assert result == "test-session"

    def test_spawn_agent_proceeds_when_browser_auth_valid(self, backend, tmp_path):
        """spawn_agent proceeds when browser-based auth is valid."""
        with patch.object(backend, "_get_worktree_path", return_value=str(tmp_path)):
            # No CURSOR_API_KEY, but is_authenticated returns True (browser auth)
            with patch.dict(os.environ, {}, clear=True):
                with patch("swarm_orchestrator.backends.cursor.is_authenticated", return_value=True):
                    with patch("subprocess.Popen") as mock_popen:
                        mock_popen.return_value = MagicMock(pid=123)
                        result = backend.spawn_agent("test-session", "prompt")
                        assert result == "test-session"

    def test_check_auth_uses_is_authenticated(self, backend):
        """_check_auth method uses is_authenticated from cursor_auth."""
        with patch("swarm_orchestrator.backends.cursor.is_authenticated") as mock_is_auth:
            mock_is_auth.return_value = True
            backend._check_auth()
            mock_is_auth.assert_called_once()

    def test_check_auth_raises_when_not_authenticated(self, backend):
        """_check_auth raises RuntimeError when not authenticated."""
        with patch("swarm_orchestrator.backends.cursor.is_authenticated", return_value=False):
            with pytest.raises(RuntimeError):
                backend._check_auth()
