"""
Interactive TUI dashboard for session management.

Provides a Rich Live-based TUI showing sessions table with status,
keyboard navigation, and actions (diff, merge, quit).
"""

import sys
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich.syntax import Syntax

from .backends.base import WorktreeBackend, AgentBackend, SessionInfo, DiffResult


class SessionsDashboard:
    """
    Interactive TUI dashboard for viewing and managing sessions.

    Keyboard controls:
    - j/↓: Move selection down
    - k/↑: Move selection up
    - d: Toggle diff preview
    - m: Merge selected session
    - s: Stop selected session (agent only)
    - x: Delete selected session
    - r: Refresh session list
    - q: Quit
    - h/PageUp: Scroll diff up
    - l/PageDown: Scroll diff down
    - H: Scroll sessions list up
    - L: Scroll sessions list down
    """

    def __init__(
        self,
        backend: WorktreeBackend,
        agent_backend: AgentBackend | None = None,
    ):
        self.backend = backend
        self.agent_backend = agent_backend
        self.sessions: list[SessionInfo] = []
        self.selected_index = 0
        self.running = False
        self.show_diff = False
        self.current_diff: DiffResult | None = None
        self.status_message = ""
        self.console = Console()
        self.diff_scroll_offset = 0
        self.sessions_scroll_offset = 0

    def refresh_sessions(self) -> None:
        """Refresh the session list from the backend."""
        self.sessions = self.backend.list_sessions("all")
        if self.selected_index >= len(self.sessions):
            self.selected_index = max(0, len(self.sessions) - 1)
        if self.show_diff and self.sessions:
            self._load_diff()

    def _load_diff(self) -> None:
        """Load diff for the currently selected session."""
        session = self.get_selected_session()
        if session:
            self.current_diff = self.backend.get_diff(session.name)
        else:
            self.current_diff = None

    def get_selected_session(self) -> SessionInfo | None:
        """Get the currently selected session."""
        if not self.sessions or self.selected_index >= len(self.sessions):
            return None
        return self.sessions[self.selected_index]

    def handle_input(self, key: str) -> None:
        """Handle keyboard input."""
        if key in ("j", "down"):
            self._move_selection(1)
        elif key in ("k", "up"):
            self._move_selection(-1)
        elif key == "d":
            self._toggle_diff()
        elif key == "m":
            self._merge_selected()
        elif key == "s":
            self._stop_selected()
        elif key == "x":
            self._delete_selected()
        elif key == "r":
            self.refresh_sessions()
            self.status_message = "Refreshed"
        elif key == "q":
            self.running = False
        elif key in ("h", "pageup"):
            self._scroll_diff(-10)
        elif key in ("l", "pagedown"):
            self._scroll_diff(10)
        elif key == "H":
            self._scroll_sessions(-5)
        elif key == "L":
            self._scroll_sessions(5)

    def _move_selection(self, delta: int) -> None:
        """Move selection by delta (wrapping around)."""
        if not self.sessions:
            return
        self.selected_index = (self.selected_index + delta) % len(self.sessions)
        if self.show_diff:
            self._load_diff()

    def _toggle_diff(self) -> None:
        """Toggle diff preview."""
        self.show_diff = not self.show_diff
        if self.show_diff:
            self._load_diff()
            self.diff_scroll_offset = 0

    def _scroll_diff(self, delta: int) -> None:
        """Scroll the diff panel by delta lines with bounds clamping."""
        if not self.current_diff:
            return
        lines = self.current_diff.content.split("\n")
        max_offset = max(0, len(lines) - 1)
        self.diff_scroll_offset = max(0, min(max_offset, self.diff_scroll_offset + delta))

    def _scroll_sessions(self, delta: int) -> None:
        """Scroll the sessions list by delta lines with bounds clamping."""
        if not self.sessions:
            return
        max_offset = max(0, len(self.sessions) - 1)
        self.sessions_scroll_offset = max(0, min(max_offset, self.sessions_scroll_offset + delta))

    def _merge_selected(self) -> None:
        """Merge the selected session."""
        session = self.get_selected_session()
        if not session:
            self.status_message = "No session selected"
            return
        try:
            self.backend.merge_session(session.name, f"Merge session: {session.name}")
            self.status_message = f"Merged: {session.name}"
            self.refresh_sessions()
        except Exception as e:
            self.status_message = f"Merge failed: {e}"

    def _stop_selected(self) -> None:
        """Stop the selected session's agent."""
        session = self.get_selected_session()
        if not session:
            self.status_message = "No session selected"
            return

        if not self.agent_backend:
            self.status_message = "No agent backend configured"
            return

        if not hasattr(self.agent_backend, "stop_agent"):
            self.status_message = "Agent backend does not support stopping"
            return

        try:
            stopped = self.agent_backend.stop_agent(session.name)
            if stopped:
                self.status_message = f"Stopped: {session.name}"
            else:
                self.status_message = f"Not running: {session.name}"
            self.refresh_sessions()
        except Exception as e:
            self.status_message = f"Stop failed: {e}"

    def _delete_selected(self) -> None:
        """Delete the selected session and clean up resources."""
        session = self.get_selected_session()
        if not session:
            self.status_message = "No session selected"
            return

        try:
            # Stop any running agent first
            if self.agent_backend and hasattr(self.agent_backend, "stop_agent"):
                self.agent_backend.stop_agent(session.name)

            # Delete the session via worktree backend
            self.backend.delete_session(session.name, force=True)
            self.status_message = f"Deleted: {session.name}"
            self.refresh_sessions()
        except Exception as e:
            self.status_message = f"Delete failed: {e}"

    def render(self) -> Panel:
        """Render the dashboard as a Rich renderable."""
        layout = Layout()

        # Help panel needs minimum 3 rows: 1 top border + 1 content + 1 bottom border
        # Use minimum_size to allow growth but never truncate
        if self.show_diff and self.current_diff:
            layout.split_column(
                Layout(name="sessions", ratio=1),
                Layout(name="diff", ratio=2),
                Layout(name="help", size=3, minimum_size=3),
            )
            layout["diff"].update(self._render_diff())
        else:
            layout.split_column(
                Layout(name="sessions", ratio=1),
                Layout(name="help", size=3, minimum_size=3),
            )

        layout["sessions"].update(self._render_sessions_table())
        layout["help"].update(self._render_help())

        title = "🐝 Swarm Sessions"
        if self.status_message:
            title += f" │ {self.status_message}"

        return Panel(layout, title=title, border_style="blue")

    def _render_sessions_table(self) -> Table:
        """Render the sessions table with scroll offset support."""
        table = Table(show_header=True, header_style="bold cyan", expand=True)
        table.add_column("", width=2)  # Selection indicator
        table.add_column("Session", style="white")
        table.add_column("Status", style="yellow")
        table.add_column("Branch", style="dim")

        if not self.sessions:
            table.add_row("", Text("No sessions found", style="dim"), "", "")
            return table

        # Apply scroll offset to visible sessions
        visible_sessions = self.sessions[self.sessions_scroll_offset:]

        for i, session in enumerate(visible_sessions):
            actual_index = i + self.sessions_scroll_offset
            indicator = "→" if actual_index == self.selected_index else ""
            row_style = "reverse" if actual_index == self.selected_index else ""

            status_style = {
                "running": "yellow",
                "reviewed": "green",
                "spec": "dim",
            }.get(session.status, "white")

            table.add_row(
                Text(indicator, style="bold cyan"),
                Text(session.name, style=row_style),
                Text(session.status, style=status_style),
                Text(session.branch, style="dim"),
            )

        return table

    def _render_diff(self) -> Panel:
        """Render the diff preview panel with scroll offset support."""
        if not self.current_diff:
            return Panel(Text("No diff available", style="dim"), title="Diff")

        lines = self.current_diff.content.split("\n")
        # Apply scroll offset and take visible lines
        visible_lines = lines[self.diff_scroll_offset:]
        content = "\n".join(visible_lines[:100])  # Limit visible lines

        if len(visible_lines) > 100:
            content += "\n... (more below)"

        title = f"Diff ({len(self.current_diff.files)} files)"
        if self.diff_scroll_offset > 0:
            title += f" [line {self.diff_scroll_offset + 1}]"

        try:
            syntax = Syntax(content, "diff", theme="monokai", line_numbers=True, start_line=self.diff_scroll_offset + 1)
            return Panel(syntax, title=title)
        except Exception:
            return Panel(Text(content), title=title)

    def _render_help(self) -> Panel:
        """Render the help panel."""
        help_text = Text()
        keys = [
            ("j/k", "nav"),
            ("d", "diff"),
            ("h/l", "scroll"),
            ("m", "merge"),
            ("x", "del"),
            ("r", "refresh"),
            ("q", "quit"),
        ]
        for i, (key, desc) in enumerate(keys):
            if i > 0:
                help_text.append(" │ ", style="dim")
            help_text.append(key, style="bold cyan")
            help_text.append(f" {desc}", style="white")

        return Panel(help_text, border_style="dim")

    def run(self) -> None:
        """Run the interactive dashboard."""
        self.running = True
        self.refresh_sessions()

        try:
            with Live(self.render(), console=self.console, refresh_per_second=4) as live:
                import select
                import termios
                import tty

                old_settings = termios.tcgetattr(sys.stdin)
                try:
                    tty.setcbreak(sys.stdin.fileno())
                    while self.running:
                        # Check for input with timeout
                        if select.select([sys.stdin], [], [], 0.25)[0]:
                            key = sys.stdin.read(1)
                            if key == "\x1b":  # Escape sequence
                                if select.select([sys.stdin], [], [], 0.1)[0]:
                                    seq = sys.stdin.read(2)
                                    if seq == "[A":
                                        key = "up"
                                    elif seq == "[B":
                                        key = "down"
                            self.handle_input(key)
                            self.status_message = ""  # Clear after any action
                        live.update(self.render())
                finally:
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        except KeyboardInterrupt:
            pass
