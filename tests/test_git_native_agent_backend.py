"""
Tests for GitNativeAgentBackend implementation.

Tests GitNativeAgentBackend implementation against the abstract AgentBackend interface.
"""

import subprocess
import pytest
from unittest.mock import MagicMock, patch

from swarm_orchestrator.backends.base import AgentBackend, AgentStatus
from swarm_orchestrator.backends.git_native import GitNativeAgentBackend
from swarm_orchestrator.backends.git_native_store import SessionStore, SessionRecord


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

    def test_spawn_agent_uses_open_command(self, backend, mock_popen, tmp_path):
        """spawn_agent uses 'open -a Terminal' to launch .command file."""
        worktree_path = tmp_path / "test-session"
        worktree_path.mkdir(parents=True)

        backend.spawn_agent("test-session", "prompt")

        mock_popen.assert_called_once()
        call_args = mock_popen.call_args
        cmd = call_args[0][0]

        assert cmd[0] == "open"
        assert "-a" in cmd
        assert "Terminal" in cmd

    def test_spawn_agent_creates_command_file(self, backend, mock_popen, tmp_path):
        """spawn_agent creates a .command file with claude CLI command."""
        worktree_path = tmp_path / "test-session"
        worktree_path.mkdir(parents=True)

        backend.spawn_agent("test-session", "prompt")

        command_file = worktree_path / ".swarm-agent.command"
        assert command_file.exists()

        content = command_file.read_text()
        assert "claude" in content
        # Should NOT use 'claude -p' (headless mode) - need interactive mode for MCP
        assert "claude -p" not in content
        assert "--dangerously-skip-permissions" in content

    def test_spawn_agent_command_file_includes_worktree_cd(self, backend, mock_popen, tmp_path):
        """spawn_agent .command file changes to worktree directory."""
        worktree_path = tmp_path / "test-session"
        worktree_path.mkdir(parents=True)

        backend.spawn_agent("test-session", "prompt")

        command_file = worktree_path / ".swarm-agent.command"
        content = command_file.read_text()

        assert f"cd '{worktree_path}'" in content

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

    def test_wait_for_completion_with_timeout(self, backend, tmp_path):
        """wait_for_completion uses wait() with timeout."""
        # Create worktree and log file
        worktree_path = tmp_path / "test-session"
        worktree_path.mkdir(parents=True)
        log_file = worktree_path / ".claude-output.log"
        log_file.write_text("test output")

        mock_process = MagicMock()
        mock_process.wait.return_value = 0
        backend._processes["test-session"] = mock_process

        result = backend.wait_for_completion(["test-session"], timeout=60)

        mock_process.wait.assert_called_once()
        assert "test-session" in result
        assert result["test-session"].is_finished is True
        assert result["test-session"].implementation == "test output"

    def test_wait_for_completion_handles_timeout_exception(self, backend):
        """wait_for_completion handles subprocess.TimeoutExpired."""
        mock_process = MagicMock()
        mock_process.wait.side_effect = subprocess.TimeoutExpired("cmd", 60)
        backend._processes["test-session"] = mock_process

        result = backend.wait_for_completion(["test-session"], timeout=60)

        assert result["test-session"].is_finished is False

    def test_wait_for_completion_multiple_agents(self, backend, tmp_path):
        """wait_for_completion handles multiple agents."""
        # Create worktrees and log files
        for name in ["agent-1", "agent-2"]:
            worktree_path = tmp_path / name
            worktree_path.mkdir(parents=True)
            log_file = worktree_path / ".claude-output.log"
            log_file.write_text(f"output-{name}")

        mock_process1 = MagicMock()
        mock_process1.wait.return_value = 0

        mock_process2 = MagicMock()
        mock_process2.wait.return_value = 0

        backend._processes["agent-1"] = mock_process1
        backend._processes["agent-2"] = mock_process2

        result = backend.wait_for_completion(["agent-1", "agent-2"], timeout=120)

        assert len(result) == 2
        assert result["agent-1"].is_finished is True
        assert result["agent-2"].is_finished is True

    def test_send_message_uses_osascript(self, backend):
        """send_message uses osascript to send keystrokes to Terminal."""
        mock_process = MagicMock()
        backend._processes["test-session"] = mock_process

        with patch("subprocess.run") as mock_run:
            backend.send_message("test-session", "test message")

            mock_run.assert_called_once()
            call_args = mock_run.call_args
            cmd = call_args[0][0]
            assert cmd[0] == "osascript"
            assert "-e" in cmd

    def test_send_message_raises_for_unknown_agent(self, backend):
        """send_message raises ValueError for unknown agent."""
        with pytest.raises(ValueError, match="not found"):
            backend.send_message("unknown-agent", "message")


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

    def test_default_cli_tool_is_claude(self):
        """Backend defaults to 'claude' CLI tool."""
        backend = GitNativeAgentBackend()
        assert backend._cli_tool == "claude"

    def test_cli_tool_can_be_set_to_opencode(self):
        """Backend accepts 'opencode' as CLI tool."""
        backend = GitNativeAgentBackend(cli_tool="opencode")
        assert backend._cli_tool == "opencode"


class TestGenerateCommand:
    """Tests for _generate_command method."""

    @pytest.fixture
    def backend_claude(self, tmp_path):
        """Backend with claude CLI tool."""
        return GitNativeAgentBackend(worktree_base=tmp_path, cli_tool="claude")

    @pytest.fixture
    def backend_opencode(self, tmp_path):
        """Backend with opencode CLI tool."""
        return GitNativeAgentBackend(worktree_base=tmp_path, cli_tool="opencode")

    def test_generate_command_claude(self, backend_claude):
        """_generate_command returns claude command with correct flags."""
        cmd = backend_claude._generate_command("$PROMPT_FILE")
        assert "claude" in cmd
        assert "--dangerously-skip-permissions" in cmd
        assert '$(cat "$PROMPT_FILE")' in cmd

    def test_generate_command_opencode(self, backend_opencode):
        """_generate_command returns opencode command with -p flag."""
        cmd = backend_opencode._generate_command("$PROMPT_FILE")
        assert "opencode" in cmd
        assert "-p" in cmd
        assert '$(cat "$PROMPT_FILE")' in cmd
        assert "--dangerously-skip-permissions" not in cmd

    def test_generate_command_opencode_non_interactive(self, backend_opencode):
        """_generate_command for opencode uses -p for non-interactive mode."""
        cmd = backend_opencode._generate_command("$PROMPT_FILE")
        # OpenCode -p flag runs non-interactively and exits after completion
        assert "opencode -p" in cmd


class TestSpawnAgentWithCliTool:
    """Tests for spawn_agent with different CLI tools."""

    @pytest.fixture
    def mock_popen(self):
        """Create a mock Popen class."""
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_process.pid = 12345
        with patch("subprocess.Popen", return_value=mock_process) as mock:
            yield mock

    def test_spawn_agent_uses_claude_cli(self, tmp_path, mock_popen):
        """spawn_agent uses claude CLI when cli_tool='claude'."""
        backend = GitNativeAgentBackend(worktree_base=tmp_path, cli_tool="claude")
        worktree_path = tmp_path / "test-session"
        worktree_path.mkdir(parents=True)

        backend.spawn_agent("test-session", "test prompt")

        command_file = worktree_path / ".swarm-agent.command"
        content = command_file.read_text()
        assert "claude" in content
        assert "--dangerously-skip-permissions" in content

    def test_spawn_agent_uses_opencode_cli(self, tmp_path, mock_popen):
        """spawn_agent uses opencode CLI when cli_tool='opencode'."""
        backend = GitNativeAgentBackend(worktree_base=tmp_path, cli_tool="opencode")
        worktree_path = tmp_path / "test-session"
        worktree_path.mkdir(parents=True)

        backend.spawn_agent("test-session", "test prompt")

        command_file = worktree_path / ".swarm-agent.command"
        content = command_file.read_text()
        assert "opencode -p" in content
        assert "--dangerously-skip-permissions" not in content


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

            # .command file embeds the cd command, verify worktree path is in the script
            command_file = session_dir / ".swarm-agent.command"
            assert command_file.exists()
            content = command_file.read_text()
            assert f"cd '{session_dir}'" in content


class TestGitNativeAgentBackendStopAgent:
    """Tests for stop_agent method."""

    @pytest.fixture
    def store(self, tmp_path):
        """Create a session store with temp path."""
        return SessionStore(store_path=tmp_path / ".swarm" / "sessions.json")

    @pytest.fixture
    def backend(self, tmp_path, store):
        """Create a backend with temp worktree base and store."""
        return GitNativeAgentBackend(worktree_base=tmp_path, store=store)

    def test_stop_agent_terminates_running_process(self, backend):
        """stop_agent terminates a running process."""
        mock_process = MagicMock()
        mock_process.poll.return_value = None  # Running
        mock_process.wait.return_value = None
        backend._processes["test-session"] = mock_process

        result = backend.stop_agent("test-session")

        assert result is True
        mock_process.terminate.assert_called_once()
        assert "test-session" not in backend._processes

    def test_stop_agent_kills_on_timeout(self, backend):
        """stop_agent uses kill() if terminate() times out."""
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_process.wait.side_effect = subprocess.TimeoutExpired("cmd", 5)
        backend._processes["test-session"] = mock_process

        result = backend.stop_agent("test-session")

        assert result is True
        mock_process.terminate.assert_called_once()
        mock_process.kill.assert_called_once()

    def test_stop_agent_returns_false_for_unknown(self, backend):
        """stop_agent returns False for unknown session."""
        result = backend.stop_agent("unknown-session")
        assert result is False

    def test_stop_agent_clears_pid_from_store(self, backend, store):
        """stop_agent clears PID from session store."""
        store.save(SessionRecord(
            name="test-session",
            status="running",
            branch="git-native/test-session",
            pid=12345,
        ))
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        backend._processes["test-session"] = mock_process

        with patch("os.kill"):
            backend.stop_agent("test-session")

        record = store.get("test-session")
        assert record.pid is None

    def test_stop_agent_uses_stored_pid_when_no_process(self, backend, store):
        """stop_agent uses stored PID when process not in memory."""
        store.save(SessionRecord(
            name="test-session",
            status="running",
            branch="git-native/test-session",
            pid=99999,  # Non-existent PID
        ))

        with patch("os.kill") as mock_kill:
            mock_kill.side_effect = ProcessLookupError()
            result = backend.stop_agent("test-session")

        # Returns False because process doesn't exist (only PID failed)
        assert result is False
        # But PID should be cleared
        record = store.get("test-session")
        assert record.pid is None

    def test_stop_agent_returns_true_on_successful_kill(self, backend, store):
        """stop_agent returns True when os.kill succeeds."""
        store.save(SessionRecord(
            name="test-session",
            status="running",
            branch="git-native/test-session",
            pid=12345,
        ))

        with patch("os.kill") as mock_kill:
            with patch("subprocess.run"):  # Mock AppleScript call
                result = backend.stop_agent("test-session")

        assert result is True
        mock_kill.assert_called_once()

    def test_stop_agent_closes_terminal_window(self, backend, store, tmp_path):
        """stop_agent runs AppleScript to close Terminal window."""
        worktree_path = tmp_path / "test-session"
        store.save(SessionRecord(
            name="test-session",
            status="running",
            branch="git-native/test-session",
            worktree_path=str(worktree_path),
            pid=12345,
        ))

        with patch("os.kill"):
            with patch.object(subprocess, "run") as mock_run:
                backend.stop_agent("test-session")

        # Verify osascript was called with Terminal close command
        osascript_calls = [
            call for call in mock_run.call_args_list
            if call[0][0][0] == "osascript"
        ]
        assert len(osascript_calls) >= 1

        # Check the AppleScript contains Terminal and close logic
        applescript = osascript_calls[0][0][0][2]  # -e argument
        assert "Terminal" in applescript
        assert "close" in applescript.lower()

    def test_stop_agent_handles_applescript_error_gracefully(self, backend, store, tmp_path):
        """stop_agent handles AppleScript errors without crashing."""
        worktree_path = tmp_path / "test-session"
        store.save(SessionRecord(
            name="test-session",
            status="running",
            branch="git-native/test-session",
            worktree_path=str(worktree_path),
            pid=12345,
        ))

        def raise_on_osascript(cmd, **kwargs):
            if cmd[0] == "osascript":
                raise Exception("AppleScript error")
            return MagicMock()

        with patch("os.kill"):
            with patch.object(subprocess, "run", side_effect=raise_on_osascript):
                # Should not raise, should handle gracefully
                result = backend.stop_agent("test-session")

        # Should still return True because process termination succeeded
        assert result is True

    def test_stop_agent_closes_window_even_without_pid(self, backend, store, tmp_path):
        """stop_agent closes Terminal window even when no PID stored."""
        worktree_path = tmp_path / "test-session"
        store.save(SessionRecord(
            name="test-session",
            status="running",
            branch="git-native/test-session",
            worktree_path=str(worktree_path),
            pid=None,  # No PID stored
        ))

        with patch.object(subprocess, "run") as mock_run:
            result = backend.stop_agent("test-session")

        # AppleScript should still be called to close window
        osascript_calls = [
            call for call in mock_run.call_args_list
            if call[0][0][0] == "osascript"
        ]
        assert len(osascript_calls) >= 1

        # Returns True because Terminal window close was attempted
        assert result is True


class TestGitNativeAgentBackendPIDTracking:
    """Tests for PID tracking in spawn_agent."""

    @pytest.fixture
    def store(self, tmp_path):
        """Create a session store with temp path."""
        return SessionStore(store_path=tmp_path / ".swarm" / "sessions.json")

    def test_spawn_agent_saves_pid_to_store(self, tmp_path, store):
        """spawn_agent saves process PID to session store."""
        worktree_path = tmp_path / "test-session"
        worktree_path.mkdir(parents=True)

        # Pre-create session record (normally done by worktree backend)
        store.save(SessionRecord(
            name="test-session",
            status="running",
            branch="git-native/test-session",
        ))

        backend = GitNativeAgentBackend(worktree_base=tmp_path, store=store)

        with patch("subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.pid = 54321
            mock_popen.return_value = mock_process

            backend.spawn_agent("test-session", "prompt")

        record = store.get("test-session")
        assert record.pid == 54321
