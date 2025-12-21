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
        """Create a mock worktree backend.

        Returns only active sessions (running/reviewed), not specs.
        This simulates the 'active' filter used by refresh_sessions().
        """
        backend = MagicMock()
        backend.list_sessions.return_value = [
            SessionInfo(name="task-agent-0", status="running", branch="git-native/task-agent-0"),
            SessionInfo(name="task-agent-1", status="reviewed", branch="git-native/task-agent-1"),
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
        assert len(dashboard.sessions) == 2

    def test_refresh_sessions_excludes_specs(self, mock_backend):
        """refresh_sessions should use 'active' filter to exclude spec-only sessions."""
        dashboard = SessionsDashboard(backend=mock_backend)
        dashboard.refresh_sessions()
        # Should call list_sessions with 'active' to exclude specs
        mock_backend.list_sessions.assert_called_with("active")

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

    def test_move_selection_up(self, dashboard):
        """'k' key should move selection up."""
        dashboard.refresh_sessions()
        dashboard.selected_index = 1
        dashboard.handle_input("k")
        assert dashboard.selected_index == 0

    def test_selection_wraps_at_bounds(self, dashboard):
        """Selection should wrap around at list boundaries."""
        dashboard.refresh_sessions()
        # At top, going up wraps to bottom
        dashboard.selected_index = 0
        dashboard.handle_input("k")
        assert dashboard.selected_index == 1
        # At bottom, going down wraps to top
        dashboard.selected_index = 1
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
        assert dashboard.sessions_scroll_offset == 1  # Clamped to max (2 sessions - 1)

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
        # Should be clamped to max (2 sessions - 1 = 1)
        assert dashboard.sessions_scroll_offset == 1

    def test_sessions_table_shows_row_count_indicator(self, mock_backend):
        """Sessions table should show row count indicator for long lists."""
        # Create a list of 20 sessions
        sessions = [
            SessionInfo(name=f"task-agent-{i}", status="running", branch=f"git-native/task-agent-{i}")
            for i in range(20)
        ]
        mock_backend.list_sessions.return_value = sessions
        dashboard = SessionsDashboard(backend=mock_backend)
        dashboard.refresh_sessions()

        # Set viewport size and scroll offset
        dashboard.sessions_viewport_size = 5
        dashboard.sessions_scroll_offset = 10

        table = dashboard._render_sessions_table()

        # Verify caption is set on the table
        assert table.caption == "11-15 of 20 sessions"

    def test_selection_auto_scrolls_viewport_down(self, mock_backend):
        """Moving selection should auto-scroll viewport to keep selection visible."""
        sessions = [
            SessionInfo(name=f"task-agent-{i}", status="running", branch=f"git-native/task-agent-{i}")
            for i in range(20)
        ]
        mock_backend.list_sessions.return_value = sessions
        dashboard = SessionsDashboard(backend=mock_backend)
        dashboard.refresh_sessions()
        dashboard.sessions_viewport_size = 5

        # Start at index 0, scroll offset 0
        assert dashboard.selected_index == 0
        assert dashboard.sessions_scroll_offset == 0

        # Move to index 6 (beyond viewport of 5)
        for _ in range(6):
            dashboard.handle_input("j")

        # Viewport should have scrolled to keep selection visible
        # Selection is at 6, viewport size is 5, so offset should be at least 2
        assert dashboard.sessions_scroll_offset >= 2

    def test_selection_auto_scrolls_viewport_up(self, mock_backend):
        """Moving selection up should auto-scroll viewport to keep selection visible."""
        sessions = [
            SessionInfo(name=f"task-agent-{i}", status="running", branch=f"git-native/task-agent-{i}")
            for i in range(20)
        ]
        mock_backend.list_sessions.return_value = sessions
        dashboard = SessionsDashboard(backend=mock_backend)
        dashboard.refresh_sessions()
        dashboard.sessions_viewport_size = 5

        # Start at index 15, scroll offset 15
        dashboard.selected_index = 15
        dashboard.sessions_scroll_offset = 15

        # Move up to index 10 (beyond top of current viewport)
        for _ in range(5):
            dashboard.handle_input("k")

        # Viewport should have scrolled to keep selection visible
        # Selection is at 10, so offset should be <= 10
        assert dashboard.sessions_scroll_offset <= dashboard.selected_index

    def test_viewport_shows_correct_window_of_sessions(self, mock_backend):
        """Sessions table should show correct window of sessions."""
        sessions = [
            SessionInfo(name=f"session-{i:02d}", status="running", branch=f"branch-{i}")
            for i in range(30)
        ]
        mock_backend.list_sessions.return_value = sessions
        dashboard = SessionsDashboard(backend=mock_backend)
        dashboard.refresh_sessions()
        dashboard.sessions_viewport_size = 10
        dashboard.sessions_scroll_offset = 10

        table = dashboard._render_sessions_table()

        from rich.console import Console
        from io import StringIO
        console = Console(file=StringIO(), force_terminal=True, width=120)
        console.print(table)
        output = console.file.getvalue()

        # Sessions 10-19 should be visible
        assert "session-10" in output
        assert "session-19" in output
        # Sessions 0-9 and 20+ should not be visible
        assert "session-09" not in output
        assert "session-00" not in output
        assert "session-20" not in output

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
        console = Console(file=StringIO(), force_terminal=True, width=100)
        console.print(help_panel)
        output = console.file.getvalue()

        # All keyboard shortcuts should be visible
        assert "j/k" in output
        assert "nav" in output
        assert "d" in output
        assert "diff" in output
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

    def test_diff_renders_correct_line_window(self, dashboard, mock_backend):
        """Diff should render the correct window of lines based on scroll offset."""
        # Create diff with 200 lines
        lines = [f"line {i}" for i in range(200)]
        mock_backend.get_diff.return_value = DiffResult(
            files=["src/foo.py"],
            content="\n".join(lines),
        )
        dashboard.refresh_sessions()
        dashboard.handle_input("d")  # Enable diff view

        # Scroll down to line 50
        dashboard.diff_scroll_offset = 50
        panel = dashboard._render_diff()

        # Render to string
        from rich.console import Console
        from io import StringIO
        console = Console(file=StringIO(), force_terminal=True, width=120, height=40)
        console.print(panel)
        output = console.file.getvalue()

        # Line 50 should be visible, line 0 should not be
        assert "line 50" in output
        assert "line 0" not in output

    def test_diff_scroll_indicator_shows_position(self, dashboard, mock_backend):
        """Scroll indicator should show current position and total in large diffs."""
        lines = [f"line {i}" for i in range(200)]
        mock_backend.get_diff.return_value = DiffResult(
            files=["src/foo.py"],
            content="\n".join(lines),
        )
        dashboard.refresh_sessions()
        dashboard.handle_input("d")  # Enable diff view
        dashboard.diff_scroll_offset = 50

        panel = dashboard._render_diff()

        # Title should indicate scroll position with format [start-end/total]
        from rich.console import Console
        from io import StringIO
        console = Console(file=StringIO(), force_terminal=True, width=120, height=40)
        console.print(panel)
        output = console.file.getvalue()

        # Should show line range and total (51-150/200)
        assert "51" in output
        assert "200" in output  # Total lines

    def test_full_diff_accessible_via_scrolling(self, dashboard, mock_backend):
        """Full diff content should be accessible by scrolling."""
        lines = [f"unique-line-{i}" for i in range(300)]
        mock_backend.get_diff.return_value = DiffResult(
            files=["src/foo.py"],
            content="\n".join(lines),
        )
        dashboard.refresh_sessions()
        dashboard.handle_input("d")  # Enable diff view

        # Check first line is visible initially
        panel = dashboard._render_diff()
        from rich.console import Console
        from io import StringIO
        console = Console(file=StringIO(), force_terminal=True, width=120, height=40)
        console.print(panel)
        output = console.file.getvalue()
        assert "unique-line-0" in output

        # Scroll to end and check last line is accessible
        dashboard.diff_scroll_offset = 250
        panel = dashboard._render_diff()
        console = Console(file=StringIO(), force_terminal=True, width=120, height=40)
        console.print(panel)
        output = console.file.getvalue()
        assert "unique-line-299" in output

    def test_diff_scroll_beyond_2000_chars(self, dashboard, mock_backend):
        """Diff content beyond 2000 chars should be viewable via scrolling."""
        # Create content with more than 2000 chars
        # Each line is ~15 chars, so 200 lines = ~3000 chars
        lines = [f"longline-{i:05d}" for i in range(200)]
        content = "\n".join(lines)
        assert len(content) > 2000, "Test content should exceed 2000 chars"

        mock_backend.get_diff.return_value = DiffResult(
            files=["src/foo.py"],
            content=content,
        )
        dashboard.refresh_sessions()
        dashboard.handle_input("d")

        # Scroll to later content (beyond what 2000 chars would show)
        dashboard.diff_scroll_offset = 150
        panel = dashboard._render_diff()

        from rich.console import Console
        from io import StringIO
        console = Console(file=StringIO(), force_terminal=True, width=120, height=40)
        console.print(panel)
        output = console.file.getvalue()

        # Line 199 should be visible when scrolled
        assert "longline-00199" in output

    def test_footer_visible_with_many_sessions(self, mock_backend):
        """Footer should remain visible with many sessions in a small terminal."""
        from rich.console import Console
        from io import StringIO

        # Create 50 sessions to ensure content would exceed screen
        sessions = [
            SessionInfo(name=f"session-{i:02d}", status="running", branch=f"branch-{i}")
            for i in range(50)
        ]
        mock_backend.list_sessions.return_value = sessions
        dashboard = SessionsDashboard(backend=mock_backend)
        dashboard.refresh_sessions()

        panel = dashboard.render()

        # Render with small terminal height (only 15 lines)
        console = Console(file=StringIO(), force_terminal=True, width=80, height=15)
        console.print(panel)
        output = console.file.getvalue()

        # Footer help keys should be visible
        assert "j/k" in output or "nav" in output
        assert "quit" in output or "q" in output

    def test_footer_visible_with_large_diff(self, mock_backend):
        """Footer should remain visible with large diff content."""
        from rich.console import Console
        from io import StringIO

        mock_backend.list_sessions.return_value = [
            SessionInfo(name="test-session", status="running", branch="test-branch"),
        ]
        # Create large diff content (500 lines)
        lines = [f"diff-line-{i}" for i in range(500)]
        mock_backend.get_diff.return_value = DiffResult(
            files=["src/foo.py", "src/bar.py", "src/baz.py"],
            content="\n".join(lines),
        )

        dashboard = SessionsDashboard(backend=mock_backend)
        dashboard.refresh_sessions()
        dashboard.show_diff = True
        dashboard.current_diff = mock_backend.get_diff.return_value

        panel = dashboard.render()

        # Render with small terminal (height 20)
        console = Console(file=StringIO(), force_terminal=True, width=80, height=20)
        console.print(panel)
        output = console.file.getvalue()

        # Footer should still be visible
        assert "j/k" in output or "nav" in output

    def test_footer_shows_scroll_hints(self, dashboard):
        """Footer should include mouse scroll hints."""
        from rich.console import Console
        from io import StringIO

        dashboard.refresh_sessions()
        help_panel = dashboard._render_help()

        console = Console(file=StringIO(), force_terminal=True, width=100)
        console.print(help_panel)
        output = console.file.getvalue()

        # Should have scroll hints (indicating mouse wheel navigation)
        assert "scroll" in output


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


class TestMouseScrollSupport:
    """Tests for mouse wheel scroll support."""

    @pytest.fixture
    def mock_backend(self):
        """Create a mock worktree backend."""
        backend = MagicMock()
        backend.list_sessions.return_value = [
            SessionInfo(name="task-agent-0", status="running", branch="git-native/task-agent-0"),
            SessionInfo(name="task-agent-1", status="reviewed", branch="git-native/task-agent-1"),
        ]
        backend.get_diff.return_value = DiffResult(
            files=["src/foo.py"],
            content="\n".join([f"line {i}" for i in range(50)]),
        )
        return backend

    @pytest.fixture
    def dashboard(self, mock_backend):
        """Create a dashboard instance with mocked backend."""
        return SessionsDashboard(backend=mock_backend)

    def test_parse_mouse_event_scroll_up(self, dashboard):
        """Should parse SGR scroll wheel up event (button 64)."""
        # SGR format: CSI < button ; x ; y M
        # Button 64 = scroll up
        event = dashboard._parse_mouse_event("<64;10;5M")
        assert event is not None
        assert event["button"] == 64
        assert event["action"] == "scroll_up"

    def test_parse_mouse_event_scroll_down(self, dashboard):
        """Should parse SGR scroll wheel down event (button 65)."""
        # Button 65 = scroll down
        event = dashboard._parse_mouse_event("<65;10;5M")
        assert event is not None
        assert event["button"] == 65
        assert event["action"] == "scroll_down"

    def test_parse_mouse_event_invalid(self, dashboard):
        """Should return None for invalid mouse event."""
        assert dashboard._parse_mouse_event("invalid") is None
        assert dashboard._parse_mouse_event("<1;10;5M") is None  # Regular click, not scroll
        assert dashboard._parse_mouse_event("") is None

    def test_handle_mouse_scroll_up_no_diff(self, dashboard):
        """Scroll up without diff visible should scroll sessions list."""
        dashboard.refresh_sessions()
        dashboard.sessions_scroll_offset = 5
        dashboard.show_diff = False
        dashboard._handle_mouse_scroll("up")
        assert dashboard.sessions_scroll_offset < 5

    def test_handle_mouse_scroll_down_no_diff(self, dashboard):
        """Scroll down without diff visible should scroll sessions list."""
        dashboard.refresh_sessions()
        dashboard.sessions_scroll_offset = 0
        dashboard.show_diff = False
        dashboard._handle_mouse_scroll("down")
        # Will be clamped since we only have 2 sessions
        assert dashboard.sessions_scroll_offset >= 0

    def test_handle_mouse_scroll_up_with_diff(self, dashboard, mock_backend):
        """Scroll up with diff visible should scroll diff panel."""
        dashboard.refresh_sessions()
        dashboard.show_diff = True
        dashboard.current_diff = mock_backend.get_diff.return_value
        dashboard.diff_scroll_offset = 20
        dashboard._handle_mouse_scroll("up")
        assert dashboard.diff_scroll_offset < 20

    def test_handle_mouse_scroll_down_with_diff(self, dashboard, mock_backend):
        """Scroll down with diff visible should scroll diff panel."""
        dashboard.refresh_sessions()
        dashboard.show_diff = True
        dashboard.current_diff = mock_backend.get_diff.return_value
        dashboard.diff_scroll_offset = 0
        dashboard._handle_mouse_scroll("down")
        assert dashboard.diff_scroll_offset > 0

    def test_mouse_tracking_escape_sequences(self, dashboard):
        """Should have correct escape sequences for mouse tracking."""
        assert dashboard._enable_mouse_tracking_seq() == "\x1b[?1000h\x1b[?1006h"
        assert dashboard._disable_mouse_tracking_seq() == "\x1b[?1000l\x1b[?1006l"
