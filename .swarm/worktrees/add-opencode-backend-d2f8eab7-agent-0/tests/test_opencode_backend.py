"""
Tests for OpenCode CLI backend implementation.

Tests OpenCodeAgentBackend implementation against the abstract AgentBackend interface.
"""

import subprocess
import pytest
from unittest.mock import MagicMock, patch

from swarm_orchestrator.backends.base import AgentBackend, AgentStatus
from swarm_orchestrator.backends.opencode import OpenCodeAgentBackend
from swarm_orchestrator.config import get_backend_choices, SwarmConfig


class TestOpenCodeAgentBackend:
    """Tests for the OpenCodeAgentBackend implementation."""

    @pytest.fixture
    def mock_popen(self):
        """Create a mock Popen class."""
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_process.pid = 12345
        with patch("subprocess.Popen", return_value=mock_process) as mock:
            yield mock

    @pytest.fixture
    def backend(self):
        """Create a backend instance."""
        return OpenCodeAgentBackend()

    def test_implements_agent_backend(self, backend):
        """OpenCodeAgentBackend implements AgentBackend interface."""
        assert isinstance(backend, AgentBackend)

    def test_spawn_agent_runs_opencode_with_correct_flags(self, backend, mock_popen, tmp_path):
        """spawn_agent runs opencode with -p flag."""
        worktree_path = str(tmp_path)
        prompt = "Test prompt content"

        with patch.object(backend, "_get_worktree_path", return_value=worktree_path):
            backend.spawn_agent("test-session", prompt)

        mock_popen.assert_called_once()
        call_args = mock_popen.call_args
        cmd = call_args[0][0]

        assert cmd[0] == "opencode"
        assert "-p" in cmd
        assert prompt in cmd

    def test_spawn_agent_runs_in_worktree_directory(self, backend, mock_popen, tmp_path):
        """spawn_agent sets cwd to worktree path."""
        worktree_path = str(tmp_path)

        with patch.object(backend, "_get_worktree_path", return_value=worktree_path):
            backend.spawn_agent("test-session", "prompt")

        call_args = mock_popen.call_args
        assert call_args[1]["cwd"] == worktree_path

    def test_spawn_agent_returns_agent_id(self, backend, mock_popen, tmp_path):
        """spawn_agent returns the session name as agent_id."""
        with patch.object(backend, "_get_worktree_path", return_value=str(tmp_path)):
            result = backend.spawn_agent("test-session", "prompt")

        assert result == "test-session"

    def test_spawn_agent_stores_process(self, backend, mock_popen, tmp_path):
        """spawn_agent stores process in internal dict for tracking."""
        with patch.object(backend, "_get_worktree_path", return_value=str(tmp_path)):
            backend.spawn_agent("test-session", "prompt")

        assert "test-session" in backend._processes

    def test_get_status_running(self, backend):
        """get_status returns is_finished=False when process is running."""
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        backend._processes["test-session"] = mock_process

        result = backend.get_status("test-session")

        assert isinstance(result, AgentStatus)
        assert result.agent_id == "test-session"
        assert result.is_finished is False
        assert result.implementation is None

    def test_get_status_finished(self, backend):
        """get_status returns is_finished=True when process completed."""
        mock_process = MagicMock()
        mock_process.poll.return_value = 0
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

    def test_wait_for_completion_unknown_agent(self, backend):
        """wait_for_completion returns is_finished=False for unknown agent."""
        result = backend.wait_for_completion(["unknown-agent"])

        assert result["unknown-agent"].is_finished is False

    def test_send_message_raises_not_implemented(self, backend):
        """send_message raises NotImplementedError as documented."""
        with pytest.raises(NotImplementedError):
            backend.send_message("test-session", "message")

    def test_stop_agent_running_process(self, backend):
        """stop_agent terminates a running process."""
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        backend._processes["test-session"] = mock_process

        result = backend.stop_agent("test-session")

        assert result is True
        mock_process.terminate.assert_called_once()
        mock_process.wait.assert_called_once_with(timeout=5)
        assert "test-session" not in backend._processes

    def test_stop_agent_uses_sigkill_on_timeout(self, backend):
        """stop_agent falls back to SIGKILL when terminate times out."""
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
        mock_process.poll.return_value = 0
        backend._processes["test-session"] = mock_process

        result = backend.stop_agent("test-session")

        assert result is False
        mock_process.terminate.assert_not_called()
        assert "test-session" not in backend._processes


class TestOpenCodeAgentBackendDefaults:
    """Tests for default configuration."""

    def test_processes_dict_initialized_empty(self):
        """Backend initializes with empty processes dict."""
        backend = OpenCodeAgentBackend()
        assert backend._processes == {}


class TestOpenCodeConfigRegistration:
    """Tests for config.py registration of opencode backend."""

    def test_opencode_in_agent_backend_choices(self):
        """'opencode' is registered in BACKENDS['agent'] choices."""
        choices = get_backend_choices("agent")
        assert "opencode" in choices

    def test_config_accepts_opencode_as_agent_backend(self):
        """SwarmConfig accepts 'opencode' as valid agent_backend."""
        config = SwarmConfig(agent_backend="opencode")
        assert config.agent_backend == "opencode"

    def test_config_rejects_invalid_agent_backend(self):
        """SwarmConfig rejects invalid agent_backend values."""
        with pytest.raises(ValueError) as exc_info:
            SwarmConfig(agent_backend="invalid-backend")
        assert "Invalid agent_backend" in str(exc_info.value)
