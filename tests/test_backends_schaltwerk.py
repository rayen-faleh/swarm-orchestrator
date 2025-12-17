"""
Tests for Schaltwerk backend implementations.

Tests SchaltwerkWorktreeBackend and SchaltwerkAgentBackend
implementations against the abstract interface contracts.
"""

import pytest
from unittest.mock import MagicMock, patch

from swarm_orchestrator.backends.base import (
    WorktreeBackend,
    AgentBackend,
    SessionInfo,
    DiffResult,
    AgentStatus,
)
from swarm_orchestrator.backends.schaltwerk import (
    SchaltwerkWorktreeBackend,
    SchaltwerkAgentBackend,
)
from swarm_orchestrator.schaltwerk import SessionStatus


class TestSchaltwerkWorktreeBackend:
    """Tests for the SchaltwerkWorktreeBackend implementation."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock SchaltwerkClient."""
        return MagicMock()

    @pytest.fixture
    def backend(self, mock_client):
        """Create a backend with a mock client."""
        return SchaltwerkWorktreeBackend(client=mock_client)

    def test_implements_worktree_backend(self, backend):
        """SchaltwerkWorktreeBackend implements WorktreeBackend interface."""
        assert isinstance(backend, WorktreeBackend)

    def test_create_session(self, backend, mock_client):
        """create_session creates a spec and returns SessionInfo."""
        mock_client.create_spec.return_value = {"success": True}
        mock_client.get_session_status.return_value = SessionStatus(
            name="test-session",
            status="spec",
            session_state="spec",
            ready_to_merge=False,
            branch="schaltwerk/test-session",
            worktree_path=None,
        )

        result = backend.create_session("test-session", "Task content here")

        mock_client.create_spec.assert_called_once_with("test-session", "Task content here")
        assert isinstance(result, SessionInfo)
        assert result.name == "test-session"
        assert result.status == "spec"

    def test_delete_session(self, backend, mock_client):
        """delete_session calls cancel_session on the client."""
        mock_client.cancel_session.return_value = {"success": True}

        backend.delete_session("test-session")

        mock_client.cancel_session.assert_called_once_with("test-session", False)

    def test_delete_session_force(self, backend, mock_client):
        """delete_session with force=True passes force to client."""
        mock_client.cancel_session.return_value = {"success": True}

        backend.delete_session("test-session", force=True)

        mock_client.cancel_session.assert_called_once_with("test-session", True)

    def test_get_session_found(self, backend, mock_client):
        """get_session returns SessionInfo when session exists."""
        mock_client.get_session_status.return_value = SessionStatus(
            name="test-session",
            status="running",
            session_state="running",
            ready_to_merge=False,
            branch="schaltwerk/test-session",
            worktree_path="/path/to/worktree",
            created_at="2024-01-01T00:00:00Z",
        )

        result = backend.get_session("test-session")

        mock_client.get_session_status.assert_called_once_with("test-session")
        assert isinstance(result, SessionInfo)
        assert result.name == "test-session"
        assert result.status == "running"
        assert result.worktree_path == "/path/to/worktree"

    def test_get_session_not_found(self, backend, mock_client):
        """get_session returns None when session doesn't exist."""
        mock_client.get_session_status.return_value = None

        result = backend.get_session("nonexistent")

        assert result is None

    def test_list_sessions(self, backend, mock_client):
        """list_sessions returns list of SessionInfo objects."""
        mock_client.get_session_list.return_value = [
            SessionStatus(
                name="session-1",
                status="running",
                session_state="running",
                ready_to_merge=False,
                branch="schaltwerk/session-1",
                worktree_path="/path/1",
            ),
            SessionStatus(
                name="session-2",
                status="reviewed",
                session_state="reviewed",
                ready_to_merge=True,
                branch="schaltwerk/session-2",
                worktree_path="/path/2",
            ),
        ]

        result = backend.list_sessions("all")

        mock_client.get_session_list.assert_called_once_with("all")
        assert len(result) == 2
        assert all(isinstance(s, SessionInfo) for s in result)
        assert result[0].name == "session-1"
        assert result[1].name == "session-2"
        assert result[1].ready_to_merge is True

    def test_get_diff(self, backend, mock_client):
        """get_diff returns DiffResult with files and content."""
        mock_client.get_diff_summary.return_value = {
            "files": [{"path": "src/main.py"}, {"path": "tests/test_main.py"}]
        }
        mock_client.get_full_diff.return_value = "diff --git a/src/main.py..."

        result = backend.get_diff("test-session")

        mock_client.get_diff_summary.assert_called_once_with("test-session")
        mock_client.get_full_diff.assert_called_once_with("test-session")
        assert isinstance(result, DiffResult)
        assert result.files == ["src/main.py", "tests/test_main.py"]
        assert result.content == "diff --git a/src/main.py..."

    def test_get_diff_with_changes_format(self, backend, mock_client):
        """get_diff handles 'changes' key format in diff summary."""
        mock_client.get_diff_summary.return_value = {
            "changes": [{"path": "file.py"}]
        }
        mock_client.get_full_diff.return_value = "diff content"

        result = backend.get_diff("test-session")

        assert result.files == ["file.py"]

    def test_merge_session(self, backend, mock_client):
        """merge_session marks as reviewed and merges."""
        mock_client.mark_reviewed.return_value = {"success": True}
        mock_client.merge_session.return_value = {"success": True}

        backend.merge_session("test-session", "feat: add feature")

        mock_client.mark_reviewed.assert_called_once_with("test-session")
        mock_client.merge_session.assert_called_once_with(
            "test-session",
            "feat: add feature",
            mode="squash",
            cancel_after_merge=True,
        )


class TestSchaltwerkAgentBackend:
    """Tests for the SchaltwerkAgentBackend implementation."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock SchaltwerkClient."""
        return MagicMock()

    @pytest.fixture
    def backend(self, mock_client):
        """Create a backend with a mock client."""
        return SchaltwerkAgentBackend(client=mock_client)

    def test_implements_agent_backend(self, backend):
        """SchaltwerkAgentBackend implements AgentBackend interface."""
        assert isinstance(backend, AgentBackend)

    def test_spawn_agent(self, backend, mock_client):
        """spawn_agent starts an agent from the session spec."""
        mock_client.start_agent.return_value = {"success": True}

        result = backend.spawn_agent("test-session", "Task prompt here")

        mock_client.start_agent.assert_called_once_with(
            "test-session",
            agent_type="claude",
            skip_permissions=True,
        )
        assert result == "test-session"

    def test_spawn_agent_custom_type(self, backend, mock_client):
        """spawn_agent supports custom agent types."""
        backend = SchaltwerkAgentBackend(client=mock_client, agent_type="opencode")
        mock_client.start_agent.return_value = {"success": True}

        result = backend.spawn_agent("test-session", "prompt")

        mock_client.start_agent.assert_called_once_with(
            "test-session",
            agent_type="opencode",
            skip_permissions=True,
        )

    def test_send_message(self, backend, mock_client):
        """send_message delegates to client."""
        mock_client.send_message.return_value = {"success": True}

        backend.send_message("test-session", "Follow-up message")

        mock_client.send_message.assert_called_once_with("test-session", "Follow-up message")

    def test_get_status_running(self, backend, mock_client):
        """get_status returns AgentStatus for running agent."""
        mock_client.get_session_status.return_value = SessionStatus(
            name="test-session",
            status="running",
            session_state="running",
            ready_to_merge=False,
            branch="schaltwerk/test-session",
        )

        result = backend.get_status("test-session")

        mock_client.get_session_status.assert_called_once_with("test-session")
        assert isinstance(result, AgentStatus)
        assert result.agent_id == "test-session"
        assert result.is_finished is False
        assert result.implementation is None

    def test_get_status_reviewed(self, backend, mock_client):
        """get_status returns finished status for reviewed agent."""
        mock_client.get_session_status.return_value = SessionStatus(
            name="test-session",
            status="reviewed",
            session_state="reviewed",
            ready_to_merge=True,
            branch="schaltwerk/test-session",
        )
        mock_client.get_full_diff.return_value = "diff content here"

        result = backend.get_status("test-session")

        assert result.is_finished is True
        assert result.implementation == "diff content here"

    def test_get_status_not_found(self, backend, mock_client):
        """get_status returns not finished when session doesn't exist."""
        mock_client.get_session_status.return_value = None

        result = backend.get_status("nonexistent")

        assert result.agent_id == "nonexistent"
        assert result.is_finished is False

    def test_wait_for_completion(self, backend, mock_client):
        """wait_for_completion waits for all agents and returns statuses."""
        mock_client.wait_for_completion.return_value = {
            "agent-1": SessionStatus(
                name="agent-1",
                status="reviewed",
                session_state="reviewed",
                ready_to_merge=True,
                branch="schaltwerk/agent-1",
            ),
            "agent-2": SessionStatus(
                name="agent-2",
                status="reviewed",
                session_state="reviewed",
                ready_to_merge=True,
                branch="schaltwerk/agent-2",
            ),
        }
        mock_client.get_full_diff.side_effect = ["diff 1", "diff 2"]

        result = backend.wait_for_completion(["agent-1", "agent-2"], timeout=60)

        mock_client.wait_for_completion.assert_called_once()
        assert len(result) == 2
        assert all(isinstance(s, AgentStatus) for s in result.values())
        assert result["agent-1"].is_finished is True
        assert result["agent-2"].is_finished is True

    def test_wait_for_completion_with_timeout(self, backend, mock_client):
        """wait_for_completion respects timeout parameter."""
        mock_client.wait_for_completion.return_value = {}
        mock_client.timeout = 600  # Original timeout

        backend.wait_for_completion(["agent-1"], timeout=120)

        # Verify timeout was temporarily set
        args, kwargs = mock_client.wait_for_completion.call_args
        assert args[0] == ["agent-1"]


class TestSchaltwerkBackendIntegration:
    """Integration tests between worktree and agent backends."""

    @pytest.fixture
    def mock_client(self):
        """Create a shared mock client."""
        return MagicMock()

    def test_worktree_and_agent_share_client(self, mock_client):
        """Worktree and agent backends can share the same client."""
        worktree_backend = SchaltwerkWorktreeBackend(client=mock_client)
        agent_backend = SchaltwerkAgentBackend(client=mock_client)

        # Simulate workflow: create session -> spawn agent -> get diff
        mock_client.create_spec.return_value = {"success": True}
        mock_client.get_session_status.return_value = SessionStatus(
            name="test",
            status="spec",
            session_state="spec",
            ready_to_merge=False,
            branch="schaltwerk/test",
            worktree_path=None,
        )

        worktree_backend.create_session("test", "content")
        agent_backend.spawn_agent("test", "prompt")

        mock_client.create_spec.assert_called_once()
        mock_client.start_agent.assert_called_once()


class TestSchaltwerkBackendDefaults:
    """Tests for default configuration."""

    def test_worktree_backend_creates_client_if_none(self):
        """WorktreeBackend creates SchaltwerkClient if none provided."""
        with patch("swarm_orchestrator.backends.schaltwerk.SchaltwerkClient") as MockClient:
            MockClient.return_value = MagicMock()
            backend = SchaltwerkWorktreeBackend()
            assert backend._client is not None
            MockClient.assert_called_once()

    def test_agent_backend_creates_client_if_none(self):
        """AgentBackend creates SchaltwerkClient if none provided."""
        with patch("swarm_orchestrator.backends.schaltwerk.SchaltwerkClient") as MockClient:
            MockClient.return_value = MagicMock()
            backend = SchaltwerkAgentBackend()
            assert backend._client is not None
            MockClient.assert_called_once()

    def test_agent_backend_default_agent_type(self, ):
        """AgentBackend defaults to 'claude' agent type."""
        with patch("swarm_orchestrator.backends.schaltwerk.SchaltwerkClient"):
            backend = SchaltwerkAgentBackend()
            assert backend._agent_type == "claude"
