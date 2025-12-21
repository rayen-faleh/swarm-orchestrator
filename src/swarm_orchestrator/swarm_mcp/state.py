"""
State management for Swarm MCP server.

Tracks:
- Task state (which agents are assigned, their status)
- Agent completion status
- Implementations submitted by each agent (with condensed diffs)
- Votes cast by each agent

Uses file locking for concurrent access from multiple agent processes.
"""

import fcntl
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Generator
from collections import Counter

from .diff_utils import (
    ImplementationSummary,
    DiffStats,
    create_implementation_summary,
)
from .compression import DiffCompressor


class AgentStatus(Enum):
    """Status of an agent in a task."""
    WORKING = "working"
    FINISHED = "finished"
    VOTED = "voted"


@dataclass
class VoteRecord:
    """Record of a vote cast by an agent."""
    voter: str
    voted_for: str
    reason: str


@dataclass
class TaskState:
    """State of a single task being worked on by multiple agents."""
    task_id: str
    agent_ids: list[str]
    session_names: dict[str, str]  # agent_id -> session_name
    agent_statuses: dict[str, AgentStatus] = field(default_factory=dict)
    implementations: dict[str, ImplementationSummary] = field(default_factory=dict)  # agent_id -> summary
    votes: dict[str, VoteRecord] = field(default_factory=dict)  # voter_id -> VoteRecord

    def all_agents_finished(self) -> bool:
        """Check if all agents have finished their work."""
        for agent_id in self.agent_ids:
            status = self.agent_statuses.get(agent_id)
            if status not in (AgentStatus.FINISHED, AgentStatus.VOTED):
                return False
        return True

    def all_agents_voted(self) -> bool:
        """Check if all agents have cast their votes."""
        for agent_id in self.agent_ids:
            if self.agent_statuses.get(agent_id) != AgentStatus.VOTED:
                return False
        return True

    def get_vote_counts(self) -> dict[str, int]:
        """Get vote counts for each agent."""
        votes_for = [v.voted_for for v in self.votes.values()]
        return dict(Counter(votes_for))

    def get_winner(self) -> Optional[str]:
        """Get the agent with the most votes, or None if no votes."""
        counts = self.get_vote_counts()
        if not counts:
            return None
        return max(counts.keys(), key=lambda k: counts[k])

    def to_dict(self) -> dict:
        """Serialize to dictionary for persistence."""
        # Serialize implementations with full data for persistence
        impl_data = {}
        for agent_id, impl in self.implementations.items():
            impl_data[agent_id] = {
                "agent_id": impl.agent_id,
                "session_name": impl.session_name,
                "stats": impl.stats.to_dict(),
                "condensed_diff": impl.condensed_diff,
                "full_diff": impl.full_diff,
            }

        return {
            "task_id": self.task_id,
            "agent_ids": self.agent_ids,
            "session_names": self.session_names,
            "agent_statuses": {k: v.value for k, v in self.agent_statuses.items()},
            "implementations": impl_data,
            "votes": {
                k: {"voter": v.voter, "voted_for": v.voted_for, "reason": v.reason}
                for k, v in self.votes.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TaskState":
        """Deserialize from dictionary."""
        task = cls(
            task_id=data["task_id"],
            agent_ids=data["agent_ids"],
            session_names=data["session_names"],
        )
        task.agent_statuses = {
            k: AgentStatus(v) for k, v in data.get("agent_statuses", {}).items()
        }

        # Deserialize implementations - handle both old (string) and new (dict) formats
        raw_impls = data.get("implementations", {})
        for agent_id, impl_data in raw_impls.items():
            if isinstance(impl_data, str):
                # Old format: raw diff string - convert to new format
                task.implementations[agent_id] = create_implementation_summary(
                    agent_id=agent_id,
                    session_name=data["session_names"].get(agent_id, ""),
                    full_diff=impl_data,
                )
            else:
                # New format: structured dict
                stats_data = impl_data.get("stats", {})
                stats = DiffStats(
                    files_changed=stats_data.get("files", []),
                    lines_added=stats_data.get("added", 0),
                    lines_deleted=stats_data.get("deleted", 0),
                )
                task.implementations[agent_id] = ImplementationSummary(
                    agent_id=impl_data.get("agent_id", agent_id),
                    session_name=impl_data.get("session_name", ""),
                    stats=stats,
                    condensed_diff=impl_data.get("condensed_diff", ""),
                    full_diff=impl_data.get("full_diff", ""),
                )

        task.votes = {
            k: VoteRecord(voter=v["voter"], voted_for=v["voted_for"], reason=v["reason"])
            for k, v in data.get("votes", {}).items()
        }
        return task


class SwarmState:
    """
    Central state manager for the swarm orchestrator.

    Tracks all tasks and their states, with optional persistence.
    Uses file locking for safe concurrent access from multiple agents.
    """

    def __init__(self, persistence_path: Optional[str] = None, compression_config=None):
        """
        Initialize state manager.

        Args:
            persistence_path: Optional path to JSON file for state persistence.
            compression_config: Optional SwarmConfig with compression settings.
        """
        self.persistence_path = persistence_path
        self._tasks: dict[str, TaskState] = {}
        self._lock_path = f"{persistence_path}.lock" if persistence_path else None
        self._compression_config = compression_config
        self._compressor: Optional[DiffCompressor] = None

        # Ensure parent directory exists
        if persistence_path:
            Path(persistence_path).parent.mkdir(parents=True, exist_ok=True)

        # Load existing state if persistence file exists
        if persistence_path and Path(persistence_path).exists():
            self.load()

    @contextmanager
    def _file_lock(self, exclusive: bool = False) -> Generator[None, None, None]:
        """
        Context manager for file locking.

        Args:
            exclusive: If True, acquire exclusive lock (for writing).
                      If False, acquire shared lock (for reading).
        """
        if not self._lock_path:
            yield
            return

        lock_file = open(self._lock_path, "w")
        try:
            if exclusive:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            else:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_SH)
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()

    def create_task(
        self,
        task_id: str,
        agent_ids: list[str],
        session_names: dict[str, str],
    ) -> TaskState:
        """
        Create a new task with the given agents.

        Args:
            task_id: Unique identifier for the task
            agent_ids: List of agent IDs working on this task
            session_names: Mapping of agent_id to Schaltwerk session name

        Returns:
            The created TaskState
        """
        task = TaskState(
            task_id=task_id,
            agent_ids=agent_ids,
            session_names=session_names,
        )
        self._tasks[task_id] = task
        self._auto_save()
        return task

    def get_task(self, task_id: str) -> Optional[TaskState]:
        """Get a task by ID, or None if not found.

        Refreshes from disk to get latest state from other agents.
        """
        # Refresh from disk to get latest updates from other agents
        if self.persistence_path:
            self.load()
        return self._tasks.get(task_id)

    def mark_agent_finished(
        self,
        task_id: str,
        agent_id: str,
        implementation: str,
    ) -> dict:
        """
        Mark an agent as finished and store their implementation.

        Creates a structured ImplementationSummary with:
        - Condensed diff (no context lines, normalized whitespace)
        - Stats (files changed, lines added/deleted)
        - Full diff (kept for reference)

        Args:
            task_id: The task ID
            agent_id: The agent that finished
            implementation: The agent's implementation (git diff)

        Returns:
            Result dict with success status and remaining agent count
        """
        task = self.get_task(task_id)
        if not task:
            return {"success": False, "error": f"Task '{task_id}' not found"}

        if agent_id not in task.agent_ids:
            return {"success": False, "error": f"Agent '{agent_id}' not found in task"}

        task.agent_statuses[agent_id] = AgentStatus.FINISHED

        # Create structured implementation summary
        session_name = task.session_names.get(agent_id, "")
        task.implementations[agent_id] = create_implementation_summary(
            agent_id=agent_id,
            session_name=session_name,
            full_diff=implementation,
        )
        self._auto_save()

        # Count remaining agents
        finished_count = sum(
            1 for aid in task.agent_ids
            if task.agent_statuses.get(aid) in (AgentStatus.FINISHED, AgentStatus.VOTED)
        )
        remaining = len(task.agent_ids) - finished_count

        return {
            "success": True,
            "agents_remaining": remaining,
            "all_finished": remaining == 0,
        }

    def _get_compressor(self) -> Optional[DiffCompressor]:
        """Get or create the diff compressor based on config."""
        if self._compression_config is None:
            return None
        if not getattr(self._compression_config, 'enable_diff_compression', True):
            return None
        if self._compressor is None:
            ratio = getattr(self._compression_config, 'compression_target_ratio', 0.3)
            self._compressor = DiffCompressor(target_ratio=ratio)
        return self._compressor

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count using simple word splitting."""
        return len(text.split())

    def _compress_diff(self, diff: str) -> tuple[str, Optional[dict]]:
        """
        Compress a diff if enabled and above threshold.

        Returns:
            Tuple of (compressed_diff, compression_stats or None)
        """
        if not diff:
            return diff, None

        compressor = self._get_compressor()
        if compressor is None:
            return diff, None

        min_tokens = getattr(self._compression_config, 'compression_min_tokens', 500)
        original_tokens = self._estimate_tokens(diff)

        if original_tokens < min_tokens:
            return diff, None

        compressed = compressor.compress(diff)
        compressed_tokens = self._estimate_tokens(compressed)

        # Always report stats when above threshold (even if compression didn't reduce size)
        return compressed, {
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
        }

    def get_all_implementations(self, task_id: str) -> dict:
        """
        Get all implementations for a task.

        Returns structured summaries with:
        - Metadata (files changed, stats)
        - Condensed diff (no context, normalized whitespace)
        - Compression stats (if compression was applied)

        Only available after all agents have finished.

        Args:
            task_id: The task ID

        Returns:
            Result dict with implementations list or error
        """
        task = self.get_task(task_id)
        if not task:
            return {"success": False, "error": f"Task '{task_id}' not found"}

        if not task.all_agents_finished():
            finished = sum(
                1 for aid in task.agent_ids
                if task.agent_statuses.get(aid) in (AgentStatus.FINISHED, AgentStatus.VOTED)
            )
            return {
                "success": False,
                "error": f"Not all agents have finished yet ({finished}/{len(task.agent_ids)})",
            }

        implementations = []
        for agent_id in task.agent_ids:
            impl = task.implementations.get(agent_id)
            if impl:
                condensed, compression_stats = self._compress_diff(impl.condensed_diff)
                impl_data = {
                    "agent_id": agent_id,
                    "session_name": impl.session_name,
                    "summary": {
                        "files_changed": impl.stats.files_changed,
                        "stats": impl.stats.to_dict(),
                    },
                    "condensed_diff": condensed,
                }
                if compression_stats:
                    impl_data["compression"] = compression_stats
                implementations.append(impl_data)
            else:
                # Fallback for missing implementation
                implementations.append({
                    "agent_id": agent_id,
                    "session_name": task.session_names.get(agent_id, ""),
                    "summary": {
                        "files_changed": [],
                        "stats": {"files": [], "files_count": 0, "added": 0, "deleted": 0, "total": 0},
                    },
                    "condensed_diff": "",
                })

        return {
            "success": True,
            "implementations": implementations,
        }

    def cast_vote(
        self,
        task_id: str,
        agent_id: str,
        voted_for: str,
        reason: str,
    ) -> dict:
        """
        Cast a vote for the best implementation.

        Args:
            task_id: The task ID
            agent_id: The agent casting the vote
            voted_for: The agent ID being voted for
            reason: Reason for the vote

        Returns:
            Result dict with success status and remaining votes
        """
        task = self.get_task(task_id)
        if not task:
            return {"success": False, "error": f"Task '{task_id}' not found"}

        if agent_id not in task.agent_ids:
            return {"success": False, "error": f"Agent '{agent_id}' not found in task"}

        if not task.all_agents_finished():
            return {"success": False, "error": "Cannot vote until all agents have finished"}

        if voted_for not in task.agent_ids:
            return {"success": False, "error": f"Invalid vote target: '{voted_for}' is not an agent in this task"}

        if voted_for == agent_id:
            return {"success": False, "error": f"Cannot vote for yourself. Vote for another agent's implementation."}

        if agent_id in task.votes:
            return {"success": False, "error": f"Agent '{agent_id}' has already voted"}

        task.votes[agent_id] = VoteRecord(
            voter=agent_id,
            voted_for=voted_for,
            reason=reason,
        )
        task.agent_statuses[agent_id] = AgentStatus.VOTED
        self._auto_save()

        votes_remaining = len(task.agent_ids) - len(task.votes)

        return {
            "success": True,
            "votes_remaining": votes_remaining,
            "all_voted": votes_remaining == 0,
        }

    def get_vote_results(self, task_id: str) -> dict:
        """
        Get the current vote results for a task.

        Args:
            task_id: The task ID

        Returns:
            Result dict with vote counts and winner
        """
        task = self.get_task(task_id)
        if not task:
            return {"success": False, "error": f"Task '{task_id}' not found"}

        vote_counts = task.get_vote_counts()
        all_voted = task.all_agents_voted()
        winner = task.get_winner() if all_voted else None

        return {
            "success": True,
            "all_voted": all_voted,
            "votes_cast": len(task.votes),
            "votes_remaining": len(task.agent_ids) - len(task.votes),
            "vote_counts": vote_counts,
            "winner": winner,
            "votes": [
                {"voter": v.voter, "voted_for": v.voted_for, "reason": v.reason}
                for v in task.votes.values()
            ],
        }

    def save(self) -> None:
        """Save state to persistence file with exclusive lock."""
        if not self.persistence_path:
            return

        with self._file_lock(exclusive=True):
            data = {
                "tasks": {tid: task.to_dict() for tid, task in self._tasks.items()}
            }
            # Write atomically using temp file
            temp_path = f"{self.persistence_path}.tmp"
            Path(temp_path).write_text(json.dumps(data, indent=2))
            os.replace(temp_path, self.persistence_path)

    def load(self) -> None:
        """Load state from persistence file with shared lock."""
        if not self.persistence_path or not Path(self.persistence_path).exists():
            return

        with self._file_lock(exclusive=False):
            if Path(self.persistence_path).exists():
                data = json.loads(Path(self.persistence_path).read_text())
                self._tasks = {
                    tid: TaskState.from_dict(tdata)
                    for tid, tdata in data.get("tasks", {}).items()
                }

    def _refresh_and_save(self) -> None:
        """
        Refresh state from disk, merge changes, and save.

        This ensures we don't lose updates from other agents.
        """
        if not self.persistence_path:
            return

        with self._file_lock(exclusive=True):
            # Load current state from disk
            current_tasks = {}
            if Path(self.persistence_path).exists():
                data = json.loads(Path(self.persistence_path).read_text())
                current_tasks = {
                    tid: TaskState.from_dict(tdata)
                    for tid, tdata in data.get("tasks", {}).items()
                }

            # Merge our in-memory state with disk state
            # Our changes take precedence for existing tasks
            for tid, task in self._tasks.items():
                if tid in current_tasks:
                    # Merge agent statuses and implementations
                    disk_task = current_tasks[tid]
                    for aid, status in disk_task.agent_statuses.items():
                        if aid not in task.agent_statuses:
                            task.agent_statuses[aid] = status
                    for aid, impl in disk_task.implementations.items():
                        if aid not in task.implementations:
                            task.implementations[aid] = impl
                    for aid, vote in disk_task.votes.items():
                        if aid not in task.votes:
                            task.votes[aid] = vote
                current_tasks[tid] = task

            self._tasks = current_tasks

            # Save merged state
            data = {
                "tasks": {tid: task.to_dict() for tid, task in self._tasks.items()}
            }
            temp_path = f"{self.persistence_path}.tmp"
            Path(temp_path).write_text(json.dumps(data, indent=2))
            os.replace(temp_path, self.persistence_path)

    def _auto_save(self) -> None:
        """Auto-save if persistence is enabled, with refresh."""
        if self.persistence_path:
            self._refresh_and_save()
