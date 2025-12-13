"""
Schaltwerk MCP wrapper for spawning and managing Claude Code agents.
"""

import json
import subprocess
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class SessionStatus:
    name: str
    status: str
    session_state: str
    ready_to_merge: bool
    branch: str


class SchaltwerkClient:
    """Client for interacting with Schaltwerk MCP server via claude CLI."""

    def __init__(self, poll_interval: int = 10, timeout: int = 600):
        self.poll_interval = poll_interval
        self.timeout = timeout

    def _run_claude_command(self, prompt: str) -> str:
        """Run a claude command and return the output."""
        # Use claude CLI in non-interactive mode
        cmd = ["claude", "-p", prompt, "--output-format", "text"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            raise RuntimeError(f"Claude command failed: {result.stderr}")
        return result.stdout

    def create_spec(self, name: str, content: str) -> dict:
        """Create a Schaltwerk spec for an agent."""
        # For PoC, we'll use direct MCP calls through the orchestrator
        # In production, this would use the MCP client directly
        return {"name": name, "content": content, "created": True}

    def start_agent(self, session_name: str, skip_permissions: bool = True) -> dict:
        """Start a Claude Code agent from a spec."""
        return {
            "session": session_name,
            "started": True,
            "skip_permissions": skip_permissions,
        }

    def get_session_list(self) -> list[SessionStatus]:
        """Get list of all Schaltwerk sessions."""
        # This would call schaltwerk_list in production
        return []

    def get_session_status(self, session_name: str) -> Optional[SessionStatus]:
        """Get status of a specific session."""
        sessions = self.get_session_list()
        for session in sessions:
            if session.name == session_name:
                return session
        return None

    def wait_for_completion(
        self, session_names: list[str], callback=None
    ) -> dict[str, SessionStatus]:
        """Wait for all sessions to complete or timeout."""
        start_time = time.time()
        completed = {}

        while len(completed) < len(session_names):
            if time.time() - start_time > self.timeout:
                raise TimeoutError(
                    f"Timeout waiting for sessions: {set(session_names) - set(completed.keys())}"
                )

            for name in session_names:
                if name in completed:
                    continue

                status = self.get_session_status(name)
                if status and status.session_state in ["reviewed", "completed"]:
                    completed[name] = status
                    if callback:
                        callback(name, status)

            if len(completed) < len(session_names):
                time.sleep(self.poll_interval)

        return completed

    def get_diff_summary(self, session_name: str) -> dict:
        """Get diff summary for a session."""
        return {"session": session_name, "files": []}

    def get_diff_content(self, session_name: str, path: str) -> str:
        """Get full diff content for a file in a session."""
        return ""

    def get_full_diff(self, session_name: str) -> str:
        """Get the complete diff for a session (all files combined)."""
        summary = self.get_diff_summary(session_name)
        diffs = []
        for file_info in summary.get("files", []):
            path = file_info.get("path", "")
            if path:
                diff = self.get_diff_content(session_name, path)
                diffs.append(diff)
        return "\n".join(diffs)

    def merge_session(self, session_name: str, commit_message: str) -> dict:
        """Merge a session back to main branch."""
        return {
            "session": session_name,
            "merged": True,
            "commit_message": commit_message,
        }

    def cancel_session(self, session_name: str, force: bool = False) -> dict:
        """Cancel and clean up a session."""
        return {"session": session_name, "cancelled": True}


# Singleton instance for easy access
_client: Optional[SchaltwerkClient] = None


def get_client() -> SchaltwerkClient:
    """Get or create the Schaltwerk client singleton."""
    global _client
    if _client is None:
        _client = SchaltwerkClient()
    return _client
