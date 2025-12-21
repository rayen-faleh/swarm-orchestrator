"""
Interactive TUI dashboard for session management.

Provides a Rich Live-based TUI showing sessions table with status,
keyboard navigation, and actions (diff, merge, quit).
"""

import re
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
        self.sessions_viewport_size = 10  # Default viewport size

    def refresh_sessions(self) -> None:
        """Refresh the session list from the backend."""
        self.sessions = self.backend.list_sessions("active")
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
        """Move selection by delta (wrapping around), auto-scrolling viewport."""
        if not self.sessions:
            return
        self.selected_index = (self.selected_index + delta) % len(self.sessions)
        self._ensure_selection_visible()
        if self.show_diff:
            self._load_diff()

    def _ensure_selection_visible(self) -> None:
        """Auto-scroll viewport to keep selected session visible."""
        if self.selected_index < self.sessions_scroll_offset:
            self.sessions_scroll_offset = self.selected_index
        elif self.selected_index >= self.sessions_scroll_offset + self.sessions_viewport_size:
            self.sessions_scroll_offset = self.selected_index - self.sessions_viewport_size + 1

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

    def _enable_mouse_tracking_seq(self) -> str:
        """Return escape sequence to enable mouse tracking (SGR mode)."""
        return "\x1b[?1000h\x1b[?1006h"

    def _disable_mouse_tracking_seq(self) -> str:
        """Return escape sequence to disable mouse tracking."""
        return "\x1b[?1000l\x1b[?1006l"

    def _parse_mouse_event(self, seq: str) -> dict | None:
        """Parse SGR mouse event sequence and return event dict or None.

        SGR format: < button ; x ; y M (press) or m (release)
        Button 64 = scroll wheel up, Button 65 = scroll wheel down
        """
        match = re.match(r"<(\d+);(\d+);(\d+)[Mm]", seq)
        if not match:
            return None
        button = int(match.group(1))
        if button == 64:
            return {"button": button, "action": "scroll_up"}
        elif button == 65:
            return {"button": button, "action": "scroll_down"}
        return None

    def _handle_mouse_scroll(self, direction: str) -> None:
        """Handle mouse scroll event by routing to appropriate scroll method."""
        delta = -3 if direction == "up" else 3
        if self.show_diff and self.current_diff:
            self._scroll_diff(delta)
        else:
            self._scroll_sessions(delta)

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

        # Footer uses size=3 (fixed) so it's always visible.
        # Sessions/diff use ratio-based sizing to fill remaining space.
        # Rich's Layout shrinks ratio-based sections first, preserving fixed-size footer.
        if self.show_diff and self.current_diff:
            layout.split_column(
                Layout(name="sessions", ratio=1, minimum_size=3),
                Layout(name="diff", ratio=2, minimum_size=3),
                Layout(name="help", size=3),
            )
            layout["diff"].update(self._render_diff())
        else:
            layout.split_column(
                Layout(name="sessions", ratio=1, minimum_size=3),
                Layout(name="help", size=3),
            )

        layout["sessions"].update(self._render_sessions_table())
        layout["help"].update(self._render_help())

        title = "🐝 Swarm Sessions"
        if self.status_message:
            title += f" │ {self.status_message}"

        return Panel(layout, title=title, border_style="blue")

    def _render_sessions_table(self) -> Table:
        """Render the sessions table with viewport windowing and row count indicator."""
        total = len(self.sessions)

        # Build row count indicator for caption
        if total > self.sessions_viewport_size:
            start = self.sessions_scroll_offset + 1
            end = min(self.sessions_scroll_offset + self.sessions_viewport_size, total)
            caption = f"{start}-{end} of {total} sessions"
        else:
            caption = f"{total} sessions" if total else None

        table = Table(show_header=True, header_style="bold cyan", expand=True, caption=caption)
        table.add_column("", width=2)  # Selection indicator
        table.add_column("Session", style="white")
        table.add_column("Status", style="yellow")
        table.add_column("Branch", style="dim")

        if not self.sessions:
            table.add_row("", Text("No sessions found", style="dim"), "", "")
            return table

        # Apply viewport windowing
        end_offset = self.sessions_scroll_offset + self.sessions_viewport_size
        visible_sessions = self.sessions[self.sessions_scroll_offset:end_offset]

        for i, session in enumerate(visible_sessions):
            actual_index = i + self.sessions_scroll_offset
            indicator = "→" if actual_index == self.selected_index else ""
            row_style = "reverse" if actual_index == self.selected_index else ""

            status_style = {
                "running": "yellow",
                "reviewed": "green",
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
        total_lines = len(lines)

        # Apply scroll offset and take visible lines
        visible_lines = lines[self.diff_scroll_offset:]
        visible_count = min(100, len(visible_lines))
        content = "\n".join(visible_lines[:visible_count])

        # Build title with file count and scroll position
        title = f"Diff ({len(self.current_diff.files)} files)"

        # Show scroll indicator when content exceeds visible area or scrolled
        if total_lines > visible_count or self.diff_scroll_offset > 0:
            end_line = min(self.diff_scroll_offset + visible_count, total_lines)
            title += f" [{self.diff_scroll_offset + 1}-{end_line}/{total_lines}]"

        try:
            syntax = Syntax(content, "diff", theme="monokai", line_numbers=True, start_line=self.diff_scroll_offset + 1)
            return Panel(syntax, title=title)
        except Exception:
            return Panel(Text(content), title=title)

    def _render_help(self) -> Panel:
        """Render the help panel with scroll hints."""
        help_text = Text()
        keys = [
            ("j/k", "nav"),
            ("d", "diff"),
            ("scroll", "↕"),
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
                    # Enable mouse tracking
                    sys.stdout.write(self._enable_mouse_tracking_seq())
                    sys.stdout.flush()

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
                                    elif seq == "[<":
                                        # SGR mouse event: read until M or m
                                        mouse_seq = "<"
                                        while True:
                                            if select.select([sys.stdin], [], [], 0.1)[0]:
                                                c = sys.stdin.read(1)
                                                mouse_seq += c
                                                if c in "Mm":
                                                    break
                                            else:
                                                break
                                        event = self._parse_mouse_event(mouse_seq)
                                        if event:
                                            self._handle_mouse_scroll(event["action"].replace("scroll_", ""))
                                        key = None  # Don't process as regular key
                            if key:
                                self.handle_input(key)
                            self.status_message = ""  # Clear after any action
                        live.update(self.render())
                finally:
                    # Disable mouse tracking
                    sys.stdout.write(self._disable_mouse_tracking_seq())
                    sys.stdout.flush()
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        except KeyboardInterrupt:
            pass
