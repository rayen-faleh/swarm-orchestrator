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

from .backends.base import WorktreeBackend, SessionInfo, DiffResult


class SessionsDashboard:
    """
    Interactive TUI dashboard for viewing and managing sessions.

    Keyboard controls:
    - j/↓: Move selection down
    - k/↑: Move selection up
    - d: Toggle diff preview
    - m: Merge selected session
    - r: Refresh session list
    - q: Quit
    """

    def __init__(self, backend: WorktreeBackend):
        self.backend = backend
        self.sessions: list[SessionInfo] = []
        self.selected_index = 0
        self.running = False
        self.show_diff = False
        self.current_diff: DiffResult | None = None
        self.status_message = ""
        self.console = Console()

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
        elif key == "r":
            self.refresh_sessions()
            self.status_message = "Refreshed"
        elif key == "q":
            self.running = False

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

    def render(self) -> Panel:
        """Render the dashboard as a Rich renderable."""
        layout = Layout()

        if self.show_diff and self.current_diff:
            layout.split_column(
                Layout(name="sessions", ratio=1),
                Layout(name="diff", ratio=2),
                Layout(name="help", size=3),
            )
            layout["diff"].update(self._render_diff())
        else:
            layout.split_column(
                Layout(name="sessions", ratio=1),
                Layout(name="help", size=3),
            )

        layout["sessions"].update(self._render_sessions_table())
        layout["help"].update(self._render_help())

        title = "🐝 Swarm Sessions"
        if self.status_message:
            title += f" │ {self.status_message}"

        return Panel(layout, title=title, border_style="blue")

    def _render_sessions_table(self) -> Table:
        """Render the sessions table."""
        table = Table(show_header=True, header_style="bold cyan", expand=True)
        table.add_column("", width=2)  # Selection indicator
        table.add_column("Session", style="white")
        table.add_column("Status", style="yellow")
        table.add_column("Branch", style="dim")

        if not self.sessions:
            table.add_row("", Text("No sessions found", style="dim"), "", "")
            return table

        for i, session in enumerate(self.sessions):
            indicator = "→" if i == self.selected_index else ""
            row_style = "reverse" if i == self.selected_index else ""

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
        """Render the diff preview panel."""
        if not self.current_diff:
            return Panel(Text("No diff available", style="dim"), title="Diff")

        content = self.current_diff.content[:2000]  # Limit for display
        if len(self.current_diff.content) > 2000:
            content += "\n... (truncated)"

        try:
            syntax = Syntax(content, "diff", theme="monokai", line_numbers=True)
            return Panel(syntax, title=f"Diff ({len(self.current_diff.files)} files)")
        except Exception:
            return Panel(Text(content), title=f"Diff ({len(self.current_diff.files)} files)")

    def _render_help(self) -> Panel:
        """Render the help panel."""
        help_text = Text()
        keys = [
            ("j/k", "navigate"),
            ("d", "diff"),
            ("m", "merge"),
            ("r", "refresh"),
            ("q", "quit"),
        ]
        for i, (key, desc) in enumerate(keys):
            if i > 0:
                help_text.append("  │  ", style="dim")
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
