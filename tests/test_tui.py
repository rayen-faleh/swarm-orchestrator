"""
Tests for TUI dashboard.

Tests SessionsDashboard rendering, keyboard navigation, and actions.
"""

import pytest
from unittest.mock import MagicMock, patch
from io import StringIO

from swarm_orchestrator.tui import SessionsDashboard
from swarm_orchestrator.backends.base import SessionInfo, DiffResult


class TestSessionsDashboard:
    """Tests for the SessionsDashboard class."""

    @pytest.fixture
    def mock_backend(self):
        """Create a mock worktree backend."""
        backend = MagicMock()
        backend.list_sessions.return_value = [
            SessionInfo(name="task-agent-0", status="running", branch="git-native/task-agent-0"),
            SessionInfo(name="task-agent-1", status="reviewed", branch="git-native/task-agent-1"),
            SessionInfo(name="task-agent-2", status="spec", branch="git-native/task-agent-2"),
        ]
        backend.get_diff.return_value = DiffResult(
            files=["src/foo.py"],
            content="+def hello(): pass",
        )
        return backend

    @pytest.fixture
    def dashboard(self, mock_backend):
        """Create a dashboard instance with mocked backend."""
        return SessionsDashboard(backend=mock_backend)

    def test_dashboard_init(self, dashboard, mock_backend):
        """Dashboard should initialize with backend and load sessions."""
        assert dashboard.backend == mock_backend
        assert dashboard.selected_index == 0
        assert dashboard.running is False

    def test_dashboard_refresh_sessions(self, dashboard, mock_backend):
        """refresh_sessions should fetch sessions from backend."""
        dashboard.refresh_sessions()
        mock_backend.list_sessions.assert_called()
        assert len(dashboard.sessions) == 3

    def test_render_returns_renderable(self, dashboard):
        """render should return a Rich renderable object."""
        dashboard.refresh_sessions()
        output = dashboard.render()
        # Should return a renderable (Panel, Layout, Table, etc.)
        assert output is not None

    def test_render_empty_sessions(self, mock_backend):
        """render should handle empty session list."""
        mock_backend.list_sessions.return_value = []
        dashboard = SessionsDashboard(backend=mock_backend)
        dashboard.refresh_sessions()
        output = dashboard.render()
        assert output is not None

    def test_move_selection_down(self, dashboard):
        """'j' key should move selection down."""
        dashboard.refresh_sessions()
        assert dashboard.selected_index == 0
        dashboard.handle_input("j")
        assert dashboard.selected_index == 1
        dashboard.handle_input("j")
        assert dashboard.selected_index == 2

    def test_move_selection_up(self, dashboard):
        """'k' key should move selection up."""
        dashboard.refresh_sessions()
        dashboard.selected_index = 2
        dashboard.handle_input("k")
        assert dashboard.selected_index == 1
        dashboard.handle_input("k")
        assert dashboard.selected_index == 0

    def test_selection_wraps_at_bounds(self, dashboard):
        """Selection should wrap around at list boundaries."""
        dashboard.refresh_sessions()
        # At top, going up wraps to bottom
        dashboard.selected_index = 0
        dashboard.handle_input("k")
        assert dashboard.selected_index == 2
        # At bottom, going down wraps to top
        dashboard.selected_index = 2
        dashboard.handle_input("j")
        assert dashboard.selected_index == 0

    def test_selection_with_empty_list(self, mock_backend):
        """Selection should handle empty list gracefully."""
        mock_backend.list_sessions.return_value = []
        dashboard = SessionsDashboard(backend=mock_backend)
        dashboard.refresh_sessions()
        # Should not crash
        dashboard.handle_input("j")
        dashboard.handle_input("k")
        assert dashboard.selected_index == 0

    def test_show_diff_action(self, dashboard, mock_backend):
        """'d' key should toggle diff preview for selected session."""
        dashboard.refresh_sessions()
        assert dashboard.show_diff is False
        dashboard.handle_input("d")
        assert dashboard.show_diff is True
        mock_backend.get_diff.assert_called_with("task-agent-0")
        # Toggle off
        dashboard.handle_input("d")
        assert dashboard.show_diff is False

    def test_diff_updates_on_selection_change(self, dashboard, mock_backend):
        """Changing selection while diff shown should update diff."""
        dashboard.refresh_sessions()
        dashboard.handle_input("d")  # Show diff
        mock_backend.get_diff.assert_called_with("task-agent-0")
        dashboard.handle_input("j")  # Move to next session
        mock_backend.get_diff.assert_called_with("task-agent-1")

    def test_merge_action(self, dashboard, mock_backend):
        """'m' key should trigger merge for selected session."""
        dashboard.refresh_sessions()
        dashboard.handle_input("m")
        mock_backend.merge_session.assert_called_once()
        # Verify session name is passed
        call_args = mock_backend.merge_session.call_args
        assert call_args[0][0] == "task-agent-0"

    def test_merge_refreshes_sessions(self, dashboard, mock_backend):
        """Merge should refresh session list after completion."""
        dashboard.refresh_sessions()
        initial_call_count = mock_backend.list_sessions.call_count
        dashboard.handle_input("m")
        assert mock_backend.list_sessions.call_count > initial_call_count

    def test_quit_action(self, dashboard):
        """'q' key should set running to False."""
        dashboard.running = True
        dashboard.handle_input("q")
        assert dashboard.running is False

    def test_refresh_action(self, dashboard, mock_backend):
        """'r' key should refresh session list."""
        dashboard.refresh_sessions()
        initial_count = mock_backend.list_sessions.call_count
        dashboard.handle_input("r")
        assert mock_backend.list_sessions.call_count > initial_count

    def test_unknown_key_ignored(self, dashboard):
        """Unknown keys should be ignored."""
        dashboard.refresh_sessions()
        initial_index = dashboard.selected_index
        dashboard.handle_input("x")  # Unknown key
        assert dashboard.selected_index == initial_index

    def test_get_selected_session(self, dashboard):
        """get_selected_session should return currently selected session."""
        dashboard.refresh_sessions()
        session = dashboard.get_selected_session()
        assert session is not None
        assert session.name == "task-agent-0"

    def test_get_selected_session_empty_list(self, mock_backend):
        """get_selected_session should return None for empty list."""
        mock_backend.list_sessions.return_value = []
        dashboard = SessionsDashboard(backend=mock_backend)
        dashboard.refresh_sessions()
        assert dashboard.get_selected_session() is None


class TestWatchCommand:
    """Tests for the 'swarm watch' CLI command."""

    @pytest.fixture
    def runner(self):
        from click.testing import CliRunner
        return CliRunner()

    def test_watch_command_exists(self, runner):
        """'swarm watch' command should exist."""
        from swarm_orchestrator.cli import main
        result = runner.invoke(main, ["watch", "--help"])
        assert result.exit_code == 0
        assert "watch" in result.output.lower() or "dashboard" in result.output.lower()

    def test_watch_creates_dashboard(self, runner):
        """'swarm watch' should create and run SessionsDashboard."""
        from swarm_orchestrator.cli import main

        with patch("swarm_orchestrator.cli._get_worktree_backend") as mock_get_backend, \
             patch("swarm_orchestrator.tui.SessionsDashboard") as MockDashboard:
            mock_backend = MagicMock()
            mock_get_backend.return_value = mock_backend
            mock_dashboard = MagicMock()
            MockDashboard.return_value = mock_dashboard
            mock_dashboard.run.side_effect = KeyboardInterrupt  # Exit immediately

            result = runner.invoke(main, ["watch"])

            MockDashboard.assert_called_once_with(backend=mock_backend)
            mock_dashboard.run.assert_called_once()
