"""
Schaltwerk MCP wrapper for spawning and managing Claude Code agents.

Integrates with the Schaltwerk MCP server to create specs, spawn agents,
monitor progress, and merge results.
"""

import json
import time
from dataclasses import dataclass
from typing import Any, Optional
from pathlib import Path

from .mcp_client import MCPClient


@dataclass
class SessionStatus:
    """Status of a Schaltwerk session."""
    name: str
    status: str
    session_state: str
    ready_to_merge: bool
    branch: str
    worktree_path: Optional[str] = None
    created_at: Optional[str] = None
    last_activity: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> "SessionStatus":
        """Create SessionStatus from API response dict."""
        return cls(
            name=data.get("name", ""),
            status=data.get("status", ""),
            session_state=data.get("session_state", ""),
            ready_to_merge=data.get("ready_to_merge", False),
            branch=data.get("branch", ""),
            worktree_path=data.get("worktree_path"),
            created_at=data.get("created_at"),
            last_activity=data.get("last_activity"),
        )


class SchaltwerkClient:
    """
    Client for interacting with Schaltwerk MCP server.

    Provides high-level methods for:
    - Creating and managing specs
    - Spawning Claude Code agents
    - Monitoring session progress
    - Comparing and merging outputs
    """

    def __init__(
        self,
        config_path: str = ".mcp.json",
        poll_interval: int = 10,
        timeout: int = 600,
    ):
        self.config_path = config_path
        self.poll_interval = poll_interval
        self.timeout = timeout
        self._mcp_client: Optional[MCPClient] = None

    def _get_mcp_client(self) -> MCPClient:
        """Get or create the MCP client."""
        if self._mcp_client is None:
            # Find config file - check current dir and parent dirs
            config_path = self._find_config_file()
            self._mcp_client = MCPClient.from_config_file(config_path, "schaltwerk")
            self._mcp_client.start()
        return self._mcp_client

    def _find_config_file(self) -> str:
        """Find the MCP config file.

        Priority order:
        1. Explicit config_path if it exists
        2. .mcp.json in current directory or parent directories
        3. ~/.claude/settings.json (global Claude Code settings)
        """
        # Check explicit path first
        if Path(self.config_path).exists():
            return self.config_path

        # Search up the directory tree for .mcp.json
        current = Path.cwd()
        while current != current.parent:
            config_file = current / ".mcp.json"
            if config_file.exists():
                return str(config_file)
            current = current.parent

        # Fallback to global ~/.claude/settings.json
        global_settings = Path.home() / ".claude" / "settings.json"
        if global_settings.exists():
            return str(global_settings)

        raise FileNotFoundError(
            "Could not find MCP config file. Checked: .mcp.json (project), "
            "~/.claude/settings.json (global)"
        )

    def _call_tool(self, tool_name: str, params: dict, timeout: float = 60.0) -> Any:
        """Call a Schaltwerk MCP tool."""
        client = self._get_mcp_client()
        return client.call_tool(tool_name, params, timeout=timeout)

    def close(self) -> None:
        """Close the MCP client connection."""
        if self._mcp_client:
            self._mcp_client.stop()
            self._mcp_client = None

    # =========================================================================
    # Spec Management
    # =========================================================================

    def create_spec(self, name: str, content: str) -> dict:
        """
        Create a Schaltwerk spec for an agent.

        Args:
            name: Unique name for the spec (e.g., "task-1-agent-0")
            content: Markdown content describing the task

        Returns:
            Dict with creation status
        """
        return self._call_tool("schaltwerk_spec_create", {
            "name": name,
            "content": content,
        })

    def update_spec(self, session_name: str, content: str, append: bool = False) -> dict:
        """Update an existing spec's content."""
        return self._call_tool("schaltwerk_draft_update", {
            "session_name": session_name,
            "content": content,
            "append": append,
        })

    def list_specs(self) -> list[dict]:
        """List all spec sessions."""
        result = self._call_tool("schaltwerk_spec_list", {})
        return result.get("specs", []) if isinstance(result, dict) else []

    def read_spec(self, session_name: str) -> str:
        """Read the content of a spec."""
        result = self._call_tool("schaltwerk_spec_read", {
            "session": session_name,
        })
        return result.get("content", "") if isinstance(result, dict) else ""

    def delete_spec(self, session_name: str) -> dict:
        """Delete a spec session."""
        return self._call_tool("schaltwerk_draft_delete", {
            "session_name": session_name,
        })

    # =========================================================================
    # Agent Management
    # =========================================================================

    def start_agent(
        self,
        session_name: str,
        agent_type: str = "claude",
        skip_permissions: bool = True,
        base_branch: Optional[str] = None,
    ) -> dict:
        """
        Start a Claude Code agent from a spec.

        Args:
            session_name: Name of the spec to start
            agent_type: Type of agent (default: "claude")
            skip_permissions: Skip permission prompts (default: True)
            base_branch: Optional base branch override

        Returns:
            Dict with session info
        """
        params = {
            "session_name": session_name,
            "agent_type": agent_type,
            "skip_permissions": skip_permissions,
        }
        if base_branch:
            params["base_branch"] = base_branch

        return self._call_tool("schaltwerk_draft_start", params)

    def send_message(self, session_name: str, message: str) -> dict:
        """Send a follow-up message to a running session."""
        return self._call_tool("schaltwerk_send_message", {
            "session_name": session_name,
            "message": message,
        })

    def cancel_session(self, session_name: str, force: bool = False) -> dict:
        """Cancel and clean up a session."""
        return self._call_tool("schaltwerk_cancel", {
            "session_name": session_name,
            "force": force,
        })

    # =========================================================================
    # Session Status
    # =========================================================================

    def get_session_list(self, filter_type: str = "all") -> list[SessionStatus]:
        """
        Get list of all Schaltwerk sessions.

        Args:
            filter_type: "all", "active", "spec", or "reviewed"

        Returns:
            List of SessionStatus objects
        """
        result = self._call_tool("schaltwerk_list", {
            "filter": filter_type,
            "json": True,
        })

        sessions = result.get("sessions", []) if isinstance(result, dict) else []
        return [SessionStatus.from_dict(s) for s in sessions]

    def get_session_status(self, session_name: str) -> Optional[SessionStatus]:
        """Get status of a specific session."""
        sessions = self.get_session_list()
        for session in sessions:
            if session.name == session_name:
                return session
        return None

    def get_current_tasks(
        self,
        status_filter: str = "all",
        fields: Optional[list[str]] = None,
    ) -> list[dict]:
        """Get detailed info about active sessions."""
        params = {"status_filter": status_filter}
        if fields:
            params["fields"] = fields

        result = self._call_tool("schaltwerk_get_current_tasks", params)
        return result if isinstance(result, list) else []

    def wait_for_completion(
        self,
        session_names: list[str],
        callback=None,
        check_states: Optional[list[str]] = None,
        use_diff_detection: bool = True,
        min_idle_time: int = 30,
    ) -> dict[str, SessionStatus]:
        """
        Wait for all sessions to complete or timeout.

        Detection methods:
        1. State-based: Wait for "reviewed" or "ready_to_merge" states
        2. Diff-based (default): Wait for sessions to have diffs and be idle

        Args:
            session_names: List of session names to wait for
            callback: Optional callback(name, status) called when each completes
            check_states: States considered "complete" (default: reviewed)
            use_diff_detection: If True, also detect completion via diffs + idle time
            min_idle_time: Seconds a session must be idle to be considered complete

        Returns:
            Dict mapping session name to final status
        """
        if check_states is None:
            check_states = ["reviewed", "ready_to_merge"]

        start_time = time.time()
        completed = {}
        last_diff_check: dict[str, tuple[float, int]] = {}  # name -> (timestamp, file_count)

        while len(completed) < len(session_names):
            if time.time() - start_time > self.timeout:
                pending = set(session_names) - set(completed.keys())
                raise TimeoutError(f"Timeout waiting for sessions: {pending}")

            for name in session_names:
                if name in completed:
                    continue

                status = self.get_session_status(name)
                if not status:
                    continue

                # Method 1: Check for explicit completion states
                if status.session_state in check_states or status.ready_to_merge:
                    completed[name] = status
                    if callback:
                        callback(name, status)
                    continue

                # Method 2: Diff-based detection (agent finished if it has diffs and is idle)
                if use_diff_detection and status.session_state == "running":
                    is_done = self._check_session_idle(
                        name, last_diff_check, min_idle_time
                    )
                    if is_done:
                        completed[name] = status
                        if callback:
                            callback(name, status)

            if len(completed) < len(session_names):
                time.sleep(self.poll_interval)

        return completed

    def _check_session_idle(
        self,
        session_name: str,
        last_diff_check: dict[str, tuple[float, int]],
        min_idle_time: int,
    ) -> bool:
        """
        Check if a session has finished working by detecting idle state.

        A session is considered idle/done if:
        1. It has produced diffs (made changes)
        2. The diff count hasn't changed for min_idle_time seconds

        Returns True if session appears to be done.
        """
        try:
            # Get current diff state
            diff_summary = self.get_diff_summary(session_name)
            files = []
            if isinstance(diff_summary, dict):
                files = diff_summary.get("files", []) or diff_summary.get("changes", [])

            current_file_count = len(files)
            current_time = time.time()

            # If no diffs yet, not done
            if current_file_count == 0:
                last_diff_check[session_name] = (current_time, 0)
                return False

            # Check if diff count has stabilized
            if session_name in last_diff_check:
                last_time, last_count = last_diff_check[session_name]

                if current_file_count == last_count:
                    # Same number of files - check if idle long enough
                    idle_time = current_time - last_time
                    if idle_time >= min_idle_time:
                        return True
                else:
                    # File count changed - reset timer
                    last_diff_check[session_name] = (current_time, current_file_count)
            else:
                # First check - record baseline
                last_diff_check[session_name] = (current_time, current_file_count)

            return False

        except Exception:
            # On error, don't consider it done
            return False

    # =========================================================================
    # Diff and Merge
    # =========================================================================

    def get_diff_summary(self, session_name: str) -> dict:
        """
        Get diff summary for a session.

        Returns dict with list of changed files.
        """
        return self._call_tool("schaltwerk_diff_summary", {
            "session": session_name,
        })

    def get_diff_chunk(
        self,
        path: str,
        session_name: Optional[str] = None,
        line_limit: int = 400,
    ) -> dict:
        """
        Get diff content for a specific file.

        Args:
            path: File path relative to repo root
            session_name: Optional session to target
            line_limit: Max lines to return (default: 400)
        """
        params = {"path": path, "line_limit": line_limit}
        if session_name:
            params["session"] = session_name

        return self._call_tool("schaltwerk_diff_chunk", params)

    def get_full_diff(self, session_name: str) -> str:
        """
        Get the complete diff for a session (all files combined).

        Returns concatenated diff content for all changed files.
        """
        summary = self.get_diff_summary(session_name)

        # Handle different response formats
        files = []
        if isinstance(summary, dict):
            files = summary.get("files", [])
            # Also check for 'changes' key
            if not files:
                files = summary.get("changes", [])

        diffs = []
        for file_info in files:
            path = file_info.get("path", "") if isinstance(file_info, dict) else str(file_info)
            if path:
                chunk = self.get_diff_chunk(path, session_name)
                if isinstance(chunk, dict):
                    lines = chunk.get("lines", [])
                    if lines:
                        diffs.append("\n".join(lines))
                elif isinstance(chunk, str):
                    diffs.append(chunk)

        return "\n".join(diffs)

    def mark_reviewed(self, session_name: str) -> dict:
        """Mark a session as reviewed and ready to merge."""
        return self._call_tool("schaltwerk_mark_session_reviewed", {
            "session_name": session_name,
        })

    def merge_session(
        self,
        session_name: str,
        commit_message: str,
        mode: str = "squash",
        cancel_after_merge: bool = True,
    ) -> dict:
        """
        Merge a session back to its parent branch.

        Args:
            session_name: Name of the reviewed session
            commit_message: Commit message for the merge
            mode: Merge mode ("squash" or "reapply")
            cancel_after_merge: Clean up session after merge

        Returns:
            Dict with merge result
        """
        return self._call_tool("schaltwerk_merge_session", {
            "session_name": session_name,
            "commit_message": commit_message,
            "mode": mode,
            "cancel_after_merge": cancel_after_merge,
        })

    def create_pr(
        self,
        session_name: str,
        cancel_after_pr: bool = False,
    ) -> dict:
        """Create a GitHub PR from a session."""
        return self._call_tool("schaltwerk_create_pr", {
            "session_name": session_name,
            "options": {"cancel_after_pr": cancel_after_pr},
        })

    def convert_to_spec(self, session_name: str) -> dict:
        """Convert a running session back to a spec for rework."""
        return self._call_tool("schaltwerk_convert_to_spec", {
            "session_name": session_name,
        })


# =========================================================================
# Module-level convenience functions
# =========================================================================

_client: Optional[SchaltwerkClient] = None


def get_client(
    config_path: str = ".mcp.json",
    poll_interval: int = 10,
    timeout: int = 600,
) -> SchaltwerkClient:
    """Get or create the Schaltwerk client singleton.

    If a client already exists, updates its timeout value to match
    the provided parameter.
    """
    global _client
    if _client is None:
        _client = SchaltwerkClient(
            config_path=config_path,
            poll_interval=poll_interval,
            timeout=timeout,
        )
    else:
        # Update timeout on existing client if it differs
        _client.timeout = timeout
    return _client


def reset_client() -> None:
    """Reset the singleton client (useful for testing)."""
    global _client
    if _client:
        _client.close()
        _client = None
