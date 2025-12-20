"""
Schaltwerk backend implementations.

Provides concrete implementations of WorktreeBackend and AgentBackend
using Schaltwerk MCP for git worktree management and agent spawning.
"""

from ..schaltwerk import SchaltwerkClient, SessionStatus
from .base import (
    WorktreeBackend,
    AgentBackend,
    SessionInfo,
    DiffResult,
    AgentStatus,
)


def _session_status_to_info(status: SessionStatus) -> SessionInfo:
    """Convert SchaltwerkClient's SessionStatus to backend SessionInfo."""
    return SessionInfo(
        name=status.name,
        status=status.session_state,
        branch=status.branch,
        worktree_path=status.worktree_path,
        ready_to_merge=status.ready_to_merge,
        created_at=status.created_at,
    )


class SchaltwerkWorktreeBackend(WorktreeBackend):
    """
    Schaltwerk implementation of WorktreeBackend.

    Uses Schaltwerk MCP server to manage git worktrees for agent isolation.
    """

    def __init__(self, client: SchaltwerkClient | None = None):
        """
        Initialize the Schaltwerk worktree backend.

        Args:
            client: Optional SchaltwerkClient instance. Creates one if not provided.
        """
        self._client = client if client is not None else SchaltwerkClient()

    def create_session(self, name: str, content: str) -> SessionInfo:
        """Create a spec session with the given content."""
        self._client.create_spec(name, content)
        status = self._client.get_session_status(name)
        if status is None:
            # Return a minimal SessionInfo if status not available yet
            return SessionInfo(name=name, status="spec", branch=f"schaltwerk/{name}")
        return _session_status_to_info(status)

    def delete_session(self, name: str, force: bool = False) -> None:
        """Delete a session and clean up resources."""
        self._client.cancel_session(name, force)

    def get_session(self, name: str) -> SessionInfo | None:
        """Get session information by name."""
        status = self._client.get_session_status(name)
        if status is None:
            return None
        return _session_status_to_info(status)

    def list_sessions(self, filter_type: str = "all") -> list[SessionInfo]:
        """List all sessions with optional filtering."""
        statuses = self._client.get_session_list(filter_type)
        return [_session_status_to_info(s) for s in statuses]

    def get_diff(self, session_name: str) -> DiffResult:
        """Get the diff for a session."""
        summary = self._client.get_diff_summary(session_name)
        content = self._client.get_full_diff(session_name)

        # Extract file paths from summary (handles both 'files' and 'changes' keys)
        files: list[str] = []
        if isinstance(summary, dict):
            file_list = summary.get("files") or summary.get("changes") or []
            for f in file_list:
                if isinstance(f, dict):
                    files.append(f.get("path", ""))
                else:
                    files.append(str(f))

        return DiffResult(files=files, content=content)

    def merge_session(self, name: str, commit_message: str) -> None:
        """Mark as reviewed and merge the session."""
        self._client.mark_reviewed(name)
        self._client.merge_session(
            name,
            commit_message,
            mode="squash",
            cancel_after_merge=True,
        )


class SchaltwerkAgentBackend(AgentBackend):
    """
    Schaltwerk implementation of AgentBackend.

    Uses Schaltwerk MCP server to spawn and manage Claude Code agents.
    """

    def __init__(
        self,
        client: SchaltwerkClient | None = None,
        agent_type: str = "claude",
    ):
        """
        Initialize the Schaltwerk agent backend.

        Args:
            client: Optional SchaltwerkClient instance. Creates one if not provided.
            agent_type: Agent type to spawn (default: "claude").
        """
        self._client = client if client is not None else SchaltwerkClient()
        self._agent_type = agent_type

    def spawn_agent(self, session_name: str, prompt: str) -> str:
        """Start an agent from the session spec."""
        self._client.start_agent(
            session_name,
            agent_type=self._agent_type,
            skip_permissions=True,
        )
        return session_name

    def wait_for_completion(
        self, agent_ids: list[str], timeout: int | None = None
    ) -> dict[str, AgentStatus]:
        """Wait for agents to complete and return their statuses."""
        # Temporarily adjust timeout if provided
        original_timeout = self._client.timeout
        if timeout is not None:
            self._client.timeout = timeout

        try:
            completed = self._client.wait_for_completion(agent_ids)
        finally:
            self._client.timeout = original_timeout

        result: dict[str, AgentStatus] = {}
        for name, status in completed.items():
            is_finished = status.session_state in ("reviewed", "ready_to_merge") or status.ready_to_merge
            impl = self._client.get_full_diff(name) if is_finished else None
            result[name] = AgentStatus(
                agent_id=name,
                is_finished=is_finished,
                implementation=impl,
            )
        return result

    def send_message(self, agent_id: str, message: str) -> None:
        """Send a message to a running agent."""
        self._client.send_message(agent_id, message)

    def get_status(self, agent_id: str) -> AgentStatus:
        """Get the current status of an agent."""
        status = self._client.get_session_status(agent_id)
        if status is None:
            return AgentStatus(agent_id=agent_id, is_finished=False)

        is_finished = status.session_state in ("reviewed", "ready_to_merge") or status.ready_to_merge
        impl = self._client.get_full_diff(agent_id) if is_finished else None
        return AgentStatus(
            agent_id=agent_id,
            is_finished=is_finished,
            implementation=impl,
        )

    def stop_agent(self, session_name: str) -> bool:
        """
        Stop a running agent.

        Note: Schaltwerk agents are managed by the MCP server and cannot be
        directly stopped. Use cancel_session on the worktree backend to
        terminate and clean up the session instead.

        Returns:
            False - Schaltwerk does not support direct agent termination.
        """
        # Schaltwerk manages agents via MCP - we don't have direct process control
        # To stop an agent, use SchaltwerkWorktreeBackend.delete_session() instead
        return False
