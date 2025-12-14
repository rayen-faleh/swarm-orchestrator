"""
Tests for the Schaltwerk client module.

TDD approach: These tests define the expected behavior of the Schaltwerk integration.
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from swarm_orchestrator.schaltwerk import (
    SchaltwerkClient,
    SessionStatus,
    get_client,
    reset_client,
)


class TestSessionStatus:
    """Tests for SessionStatus dataclass."""

    def test_create_from_dict(self):
        """Should create SessionStatus from API response dict."""
        data = {
            "name": "test-session",
            "status": "running",
            "session_state": "active",
            "ready_to_merge": False,
            "branch": "schaltwerk/test-session",
            "worktree_path": "/path/to/worktree",
            "created_at": "2024-01-01T00:00:00Z",
        }

        status = SessionStatus.from_dict(data)

        assert status.name == "test-session"
        assert status.status == "running"
        assert status.session_state == "active"
        assert status.ready_to_merge is False
        assert status.branch == "schaltwerk/test-session"
        assert status.worktree_path == "/path/to/worktree"

    def test_handles_missing_optional_fields(self):
        """Should handle missing optional fields."""
        data = {
            "name": "minimal",
            "status": "spec",
            "session_state": "spec",
            "ready_to_merge": False,
            "branch": "test",
        }

        status = SessionStatus.from_dict(data)

        assert status.name == "minimal"
        assert status.worktree_path is None
        assert status.created_at is None


class TestSchaltwerkClient:
    """Tests for SchaltwerkClient class."""

    @pytest.fixture
    def mock_mcp(self):
        """Mock the MCP client."""
        with patch("swarm_orchestrator.schaltwerk.MCPClient") as mock_class:
            mock_instance = MagicMock()
            mock_class.from_config_file.return_value = mock_instance
            mock_instance.call_tool.return_value = {}
            yield mock_instance

    @pytest.fixture
    def client_with_mock(self, mock_mcp, tmp_path):
        """Create a client with mocked MCP."""
        # Create a fake config file
        config_file = tmp_path / ".mcp.json"
        config_file.write_text('{"mcpServers": {"schaltwerk": {"command": "node"}}}')

        client = SchaltwerkClient(config_path=str(config_file))
        return client, mock_mcp

    def test_create_spec(self, client_with_mock):
        """Should call schaltwerk_spec_create tool."""
        client, mock_mcp = client_with_mock
        mock_mcp.call_tool.return_value = {"created": True}

        result = client.create_spec("test-spec", "# Test content")

        # Check the call was made with correct tool and params
        call_args = mock_mcp.call_tool.call_args
        assert call_args[0][0] == "schaltwerk_spec_create"
        assert call_args[0][1] == {"name": "test-spec", "content": "# Test content"}
        assert result == {"created": True}

    def test_start_agent(self, client_with_mock):
        """Should call schaltwerk_draft_start tool."""
        client, mock_mcp = client_with_mock
        mock_mcp.call_tool.return_value = {"started": True}

        result = client.start_agent("test-session", skip_permissions=True)

        call_args = mock_mcp.call_tool.call_args
        assert call_args[0][0] == "schaltwerk_draft_start"
        assert call_args[0][1] == {
            "session_name": "test-session",
            "agent_type": "claude",
            "skip_permissions": True,
        }

    def test_get_session_list(self, client_with_mock):
        """Should parse session list response."""
        client, mock_mcp = client_with_mock
        mock_mcp.call_tool.return_value = {
            "sessions": [
                {
                    "name": "session-1",
                    "status": "running",
                    "session_state": "active",
                    "ready_to_merge": False,
                    "branch": "test",
                },
                {
                    "name": "session-2",
                    "status": "completed",
                    "session_state": "reviewed",
                    "ready_to_merge": True,
                    "branch": "test2",
                },
            ]
        }

        sessions = client.get_session_list()

        assert len(sessions) == 2
        assert sessions[0].name == "session-1"
        assert sessions[1].ready_to_merge is True

    def test_get_session_status(self, client_with_mock):
        """Should find specific session by name."""
        client, mock_mcp = client_with_mock
        mock_mcp.call_tool.return_value = {
            "sessions": [
                {
                    "name": "target",
                    "status": "running",
                    "session_state": "active",
                    "ready_to_merge": False,
                    "branch": "test",
                },
            ]
        }

        status = client.get_session_status("target")

        assert status is not None
        assert status.name == "target"

    def test_get_session_status_not_found(self, client_with_mock):
        """Should return None for unknown session."""
        client, mock_mcp = client_with_mock
        mock_mcp.call_tool.return_value = {"sessions": []}

        status = client.get_session_status("nonexistent")

        assert status is None

    def test_cancel_session(self, client_with_mock):
        """Should call schaltwerk_cancel tool."""
        client, mock_mcp = client_with_mock

        client.cancel_session("test-session", force=True)

        call_args = mock_mcp.call_tool.call_args
        assert call_args[0][0] == "schaltwerk_cancel"
        assert call_args[0][1] == {"session_name": "test-session", "force": True}

    def test_merge_session(self, client_with_mock):
        """Should call schaltwerk_merge_session tool."""
        client, mock_mcp = client_with_mock

        client.merge_session(
            "test-session",
            commit_message="Test commit",
            mode="squash",
        )

        call_args = mock_mcp.call_tool.call_args
        assert call_args[0][0] == "schaltwerk_merge_session"
        assert call_args[0][1] == {
            "session_name": "test-session",
            "commit_message": "Test commit",
            "mode": "squash",
            "cancel_after_merge": True,
        }

    def test_get_diff_summary(self, client_with_mock):
        """Should call schaltwerk_diff_summary tool."""
        client, mock_mcp = client_with_mock
        mock_mcp.call_tool.return_value = {"files": ["file1.py", "file2.py"]}

        result = client.get_diff_summary("test-session")

        call_args = mock_mcp.call_tool.call_args
        assert call_args[0][0] == "schaltwerk_diff_summary"
        assert call_args[0][1] == {"session": "test-session"}

    def test_mark_reviewed(self, client_with_mock):
        """Should call schaltwerk_mark_session_reviewed tool."""
        client, mock_mcp = client_with_mock

        client.mark_reviewed("test-session")

        call_args = mock_mcp.call_tool.call_args
        assert call_args[0][0] == "schaltwerk_mark_session_reviewed"
        assert call_args[0][1] == {"session_name": "test-session"}


class TestSchaltwerkClientSingleton:
    """Tests for the module-level singleton functions."""

    def teardown_method(self):
        """Reset the singleton after each test."""
        reset_client()

    def test_get_client_creates_singleton(self):
        """Should create and return the same client instance."""
        with patch("swarm_orchestrator.schaltwerk.SchaltwerkClient") as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance

            client1 = get_client()
            client2 = get_client()

            assert client1 is client2
            mock_class.assert_called_once()

    def test_reset_client_clears_singleton(self):
        """Should clear the singleton and close the client."""
        with patch("swarm_orchestrator.schaltwerk.SchaltwerkClient") as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance

            get_client()
            reset_client()

            mock_instance.close.assert_called_once()

            # Getting client again should create new instance
            get_client()
            assert mock_class.call_count == 2

    def test_get_client_updates_timeout_on_existing(self):
        """Should update timeout on existing client when called with different value."""
        with patch("swarm_orchestrator.schaltwerk.SchaltwerkClient") as mock_class:
            mock_instance = MagicMock()
            mock_instance.timeout = 600  # Default timeout
            mock_class.return_value = mock_instance

            # Create client with default timeout
            client1 = get_client(timeout=600)
            assert client1.timeout == 600

            # Get client again with different timeout
            client2 = get_client(timeout=120)

            # Should be the same instance but with updated timeout
            assert client1 is client2
            assert client2.timeout == 120
            mock_class.assert_called_once()  # Only created once
