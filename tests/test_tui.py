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

    def test_scroll_state_initialization(self, dashboard):
        """Scroll offsets should initialize to zero."""
        assert dashboard.diff_scroll_offset == 0
        assert dashboard.sessions_scroll_offset == 0

    def test_scroll_diff_down(self, dashboard, mock_backend):
        """'l' key should scroll diff down."""
        mock_backend.get_diff.return_value = DiffResult(
            files=["src/foo.py"],
            content="\n".join([f"line {i}" for i in range(50)]),
        )
        dashboard.refresh_sessions()
        dashboard.handle_input("d")  # Enable diff view
        assert dashboard.diff_scroll_offset == 0
        dashboard.handle_input("l")
        assert dashboard.diff_scroll_offset == 10

    def test_scroll_diff_up(self, dashboard, mock_backend):
        """'h' key should scroll diff up."""
        mock_backend.get_diff.return_value = DiffResult(
            files=["src/foo.py"],
            content="\n".join([f"line {i}" for i in range(50)]),
        )
        dashboard.refresh_sessions()
        dashboard.handle_input("d")  # Enable diff view
        dashboard.diff_scroll_offset = 20
        dashboard.handle_input("h")
        assert dashboard.diff_scroll_offset == 10

    def test_scroll_diff_clamps_to_zero(self, dashboard, mock_backend):
        """Scroll offset should not go below zero."""
        mock_backend.get_diff.return_value = DiffResult(
            files=["src/foo.py"],
            content="line 1\nline 2\nline 3",
        )
        dashboard.refresh_sessions()
        dashboard.handle_input("d")
        dashboard.handle_input("h")  # Try to scroll up from 0
        assert dashboard.diff_scroll_offset == 0

    def test_scroll_diff_clamps_to_max(self, dashboard, mock_backend):
        """Scroll offset should not exceed content length."""
        mock_backend.get_diff.return_value = DiffResult(
            files=["src/foo.py"],
            content="line 1\nline 2\nline 3",
        )
        dashboard.refresh_sessions()
        dashboard.handle_input("d")
        # Scroll down many times
        for _ in range(10):
            dashboard.handle_input("l")
        # Should be clamped to max (3 lines - 1 = 2)
        assert dashboard.diff_scroll_offset == 2

    def test_scroll_sessions_down(self, dashboard):
        """'L' key should scroll sessions list down."""
        dashboard.refresh_sessions()
        assert dashboard.sessions_scroll_offset == 0
        dashboard.handle_input("L")
        assert dashboard.sessions_scroll_offset == 2  # Clamped to max (3 sessions - 1)

    def test_scroll_sessions_up(self, dashboard):
        """'H' key should scroll sessions list up."""
        dashboard.refresh_sessions()
        dashboard.sessions_scroll_offset = 2
        dashboard.handle_input("H")
        assert dashboard.sessions_scroll_offset == 0  # 2 - 5 clamped to 0

    def test_scroll_sessions_clamps_to_zero(self, dashboard):
        """Sessions scroll should not go below zero."""
        dashboard.refresh_sessions()
        dashboard.handle_input("H")
        assert dashboard.sessions_scroll_offset == 0

    def test_scroll_sessions_clamps_to_max(self, dashboard):
        """Sessions scroll should not exceed list length."""
        dashboard.refresh_sessions()
        for _ in range(10):
            dashboard.handle_input("L")
        # Should be clamped to max (3 sessions - 1 = 2)
        assert dashboard.sessions_scroll_offset == 2

    def test_toggle_diff_resets_scroll(self, dashboard, mock_backend):
        """Toggling diff should reset scroll offset."""
        mock_backend.get_diff.return_value = DiffResult(
            files=["src/foo.py"],
            content="\n".join([f"line {i}" for i in range(50)]),
        )
        dashboard.refresh_sessions()
        dashboard.handle_input("d")  # Enable diff
        dashboard.diff_scroll_offset = 20
        dashboard.handle_input("d")  # Disable diff
        dashboard.handle_input("d")  # Enable again
        assert dashboard.diff_scroll_offset == 0

    def test_scroll_diff_no_content(self, dashboard, mock_backend):
        """Scrolling with no diff content should not crash."""
        mock_backend.get_diff.return_value = None
        dashboard.refresh_sessions()
        dashboard.current_diff = None
        dashboard.handle_input("l")  # Should not crash
        assert dashboard.diff_scroll_offset == 0

    def test_scroll_sessions_empty_list(self, mock_backend):
        """Scrolling with no sessions should not crash."""
        mock_backend.list_sessions.return_value = []
        dashboard = SessionsDashboard(backend=mock_backend)
        dashboard.refresh_sessions()
        dashboard.handle_input("L")  # Should not crash
        assert dashboard.sessions_scroll_offset == 0

    def test_help_panel_renders_all_keys(self, dashboard):
        """Help panel should show all keyboard shortcuts without truncation."""
        from rich.console import Console
        from io import StringIO

        dashboard.refresh_sessions()
        help_panel = dashboard._render_help()

        # Render to a string to verify content
        console = Console(file=StringIO(), force_terminal=True, width=80)
        console.print(help_panel)
        output = console.file.getvalue()

        # All keyboard shortcuts should be visible
        assert "j/k" in output
        assert "nav" in output
        assert "d" in output
        assert "diff" in output
        assert "h/l" in output
        assert "scroll" in output
        assert "m" in output
        assert "merge" in output
        assert "r" in output
        assert "refresh" in output
        assert "q" in output
        assert "quit" in output

    def test_render_layout_has_minimum_size_for_help(self, dashboard):
        """Layout should have minimum_size set to prevent help panel truncation."""
        from rich.console import Console
        from rich.layout import Layout
        from io import StringIO

        dashboard.refresh_sessions()
        panel = dashboard.render()

        # The panel contains a Layout - we verify it renders without crashing
        # even in a small terminal by rendering to a narrow console
        console = Console(file=StringIO(), force_terminal=True, width=60, height=10)
        console.print(panel)
        output = console.file.getvalue()

        # Should contain help keys even in small terminal
        assert "j/k" in output or "nav" in output

    def test_render_with_diff_has_minimum_size_for_help(self, dashboard, mock_backend):
        """Layout with diff view should also have minimum_size for help panel."""
        from rich.console import Console
        from io import StringIO

        dashboard.refresh_sessions()
        dashboard.show_diff = True
        dashboard.current_diff = mock_backend.get_diff.return_value

        panel = dashboard.render()

        # Render to verify it doesn't crash with diff enabled
        console = Console(file=StringIO(), force_terminal=True, width=80, height=20)
        console.print(panel)
        output = console.file.getvalue()

        # Help should still be visible (check for keys that appear early in the bar)
        assert "j/k" in output or "nav" in output


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
             patch("swarm_orchestrator.cli._get_agent_backend") as mock_get_agent_backend, \
             patch("swarm_orchestrator.tui.SessionsDashboard") as MockDashboard:
            mock_backend = MagicMock()
            mock_agent_backend = MagicMock()
            mock_get_backend.return_value = mock_backend
            mock_get_agent_backend.return_value = mock_agent_backend
            mock_dashboard = MagicMock()
            MockDashboard.return_value = mock_dashboard
            mock_dashboard.run.side_effect = KeyboardInterrupt  # Exit immediately

            result = runner.invoke(main, ["watch"])

            MockDashboard.assert_called_once_with(
                backend=mock_backend,
                agent_backend=mock_agent_backend,
            )
            mock_dashboard.run.assert_called_once()
