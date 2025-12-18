"""
Abstract base classes for pluggable backends.

Defines contracts for:
- WorktreeBackend: Git worktree/isolation management (create, delete, diff, merge)
- AgentBackend: Agent execution (spawn, wait, message)
- LLMBackend: LLM calls (decompose, explore)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SessionInfo:
    """
    Backend-agnostic session/worktree information.

    Represents the state of an isolated development session,
    regardless of the underlying implementation (Schaltwerk, Docker, etc.).
    """

    name: str
    status: str
    branch: str
    worktree_path: str | None = None
    ready_to_merge: bool = False
    created_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DiffResult:
    """
    Backend-agnostic diff result.

    Contains the changes made in a session, suitable for review and voting.
    """

    files: list[str]
    content: str
    stats: dict[str, int] | None = None


@dataclass
class AgentStatus:
    """
    Backend-agnostic agent execution status.

    Tracks whether an agent has finished and its output.
    """

    agent_id: str
    is_finished: bool
    implementation: str | None = None


@dataclass
class DecomposeResult:
    """
    Backend-agnostic decomposition result.

    Contains the raw LLM response for task decomposition.
    Parsing into domain objects is handled by the orchestrator.
    """

    is_atomic: bool
    raw_response: dict[str, Any]
    reasoning: str | None = None


class WorktreeBackend(ABC):
    """
    Abstract interface for worktree/isolation management.

    Implementations handle creating isolated development environments,
    managing their lifecycle, and merging changes back.

    Example implementations:
    - SchaltwerkBackend: Uses Schaltwerk MCP for git worktrees
    - DockerBackend: Uses Docker containers for isolation
    - LocalBackend: Uses local directories (for testing)
    """

    @abstractmethod
    def create_session(self, name: str, content: str) -> SessionInfo:
        """
        Create a new isolated session for agent work.

        Args:
            name: Unique session identifier
            content: Initial spec/prompt content for the session

        Returns:
            SessionInfo with the created session details
        """
        ...

    @abstractmethod
    def delete_session(self, name: str, force: bool = False) -> None:
        """
        Delete a session and clean up its resources.

        Args:
            name: Session identifier to delete
            force: If True, delete even if uncommitted changes exist
        """
        ...

    @abstractmethod
    def get_session(self, name: str) -> SessionInfo | None:
        """
        Get information about a specific session.

        Args:
            name: Session identifier

        Returns:
            SessionInfo if found, None otherwise
        """
        ...

    @abstractmethod
    def list_sessions(self, filter_type: str = "all") -> list[SessionInfo]:
        """
        List all sessions, optionally filtered by status.

        Args:
            filter_type: Filter ("all", "active", "reviewed", etc.)

        Returns:
            List of SessionInfo for matching sessions
        """
        ...

    @abstractmethod
    def get_diff(self, session_name: str) -> DiffResult:
        """
        Get the diff (changes) for a session.

        Args:
            session_name: Session to get diff for

        Returns:
            DiffResult containing files changed and diff content
        """
        ...

    @abstractmethod
    def merge_session(self, name: str, commit_message: str) -> None:
        """
        Merge a session's changes back to the parent branch.

        Args:
            name: Session to merge
            commit_message: Commit message for the merge
        """
        ...


class AgentBackend(ABC):
    """
    Abstract interface for agent execution.

    Implementations handle spawning agents, tracking their progress,
    and communicating with them.

    Example implementations:
    - ClaudeCodeBackend: Uses Claude Code CLI agents
    - OpenCodeBackend: Uses opencode agents
    - MockBackend: For testing
    """

    @abstractmethod
    def spawn_agent(self, session_name: str, prompt: str) -> str:
        """
        Spawn an agent to work on a task.

        Args:
            session_name: Session/worktree for the agent to work in
            prompt: The task prompt for the agent

        Returns:
            Agent identifier for tracking
        """
        ...

    @abstractmethod
    def wait_for_completion(
        self, agent_ids: list[str], timeout: int | None = None
    ) -> dict[str, AgentStatus]:
        """
        Wait for agents to complete their work.

        Args:
            agent_ids: List of agent identifiers to wait for
            timeout: Optional timeout in seconds (None = no timeout)

        Returns:
            Dict mapping agent_id to their final status
        """
        ...

    @abstractmethod
    def send_message(self, agent_id: str, message: str) -> None:
        """
        Send a follow-up message to a running agent.

        Args:
            agent_id: Agent to send message to
            message: Message content
        """
        ...

    @abstractmethod
    def get_status(self, agent_id: str) -> AgentStatus:
        """
        Get the current status of an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            AgentStatus with current state
        """
        ...

    @abstractmethod
    def stop_agent(self, session_name: str) -> bool:
        """
        Terminate a running agent.

        Stops the agent process associated with the given session.
        Implementations should handle cleanup of any associated resources
        (terminal windows, background processes, etc.).

        Args:
            session_name: Session identifier whose agent should be stopped

        Returns:
            True if the agent was successfully stopped, False otherwise
        """
        ...


class LLMBackend(ABC):
    """
    Abstract interface for LLM calls.

    Implementations handle task decomposition and codebase exploration
    using various LLM providers.

    Example implementations:
    - ClaudeCLIBackend: Uses Claude CLI (Max/Pro subscription)
    - AnthropicAPIBackend: Uses Anthropic API directly
    - OpenAIBackend: Uses OpenAI API
    """

    @abstractmethod
    def decompose(self, query: str, context: str | None = None) -> DecomposeResult:
        """
        Decompose a task into subtasks.

        Args:
            query: The task to decompose
            context: Optional context from exploration

        Returns:
            DecomposeResult with atomic flag and raw response
        """
        ...

    @abstractmethod
    def explore(self, query: str) -> str:
        """
        Explore the codebase for relevant context.

        Args:
            query: What to explore/understand

        Returns:
            Exploration findings as text
        """
        ...
