"""
Git-native worktree backend implementation.

Provides WorktreeBackend using native git worktree commands instead of Schaltwerk MCP.
"""

import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .base import WorktreeBackend, SessionInfo, DiffResult
from .git_native_store import SessionStore, SessionRecord


class GitNativeWorktreeBackend(WorktreeBackend):
    """
    Native git implementation of WorktreeBackend.

    Uses subprocess calls to git worktree commands for isolation management.
    """

    def __init__(
        self,
        repo_path: Path | str | None = None,
        store: SessionStore | None = None,
        worktree_base: Path | str | None = None,
    ):
        """
        Initialize the git-native worktree backend.

        Args:
            repo_path: Path to the git repository. Defaults to current directory.
            store: SessionStore for metadata. Creates one if not provided.
            worktree_base: Base directory for worktrees. Defaults to .swarm/worktrees.
        """
        self._repo_path = Path(repo_path) if repo_path else Path.cwd()
        self._store = store if store else SessionStore()
        self._worktree_base = Path(worktree_base) if worktree_base else self._repo_path / ".swarm" / "worktrees"

    def _run_git(self, *args: str, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
        """Run a git command and return the result."""
        return subprocess.run(
            ["git", *args],
            cwd=cwd or self._repo_path,
            capture_output=True,
            text=True,
            check=check,
        )

    def _get_parent_branch(self) -> str:
        """Get the current branch name (parent for new worktrees)."""
        result = self._run_git("rev-parse", "--abbrev-ref", "HEAD")
        return result.stdout.strip()

    def create_session(self, name: str, content: str) -> SessionInfo:
        """Create a new git worktree session."""
        branch = f"git-native/{name}"
        worktree_path = self._worktree_base / name

        # Create the worktree with a new branch
        worktree_path.parent.mkdir(parents=True, exist_ok=True)
        self._run_git("worktree", "add", "-b", branch, str(worktree_path))

        created_at = datetime.now(timezone.utc).isoformat()

        # Store metadata
        record = SessionRecord(
            name=name,
            status="running",
            branch=branch,
            worktree_path=str(worktree_path),
            created_at=created_at,
            spec_content=content,
        )
        self._store.save(record)

        return SessionInfo(
            name=name,
            status="running",
            branch=branch,
            worktree_path=str(worktree_path),
            created_at=created_at,
        )

    def delete_session(self, name: str, force: bool = False) -> None:
        """Delete a session's worktree and clean up resources."""
        record = self._store.get(name)
        if record is None:
            return

        worktree_path = record.worktree_path
        branch = record.branch

        # Remove worktree
        if worktree_path and Path(worktree_path).exists():
            force_flag = ["--force"] if force else []
            self._run_git("worktree", "remove", *force_flag, worktree_path, check=False)

        # Prune worktrees
        self._run_git("worktree", "prune", check=False)

        # Delete branch
        if branch:
            self._run_git("branch", "-D", branch, check=False)

        # Remove metadata
        self._store.delete(name)

    def get_session(self, name: str) -> SessionInfo | None:
        """Get session information by name."""
        record = self._store.get(name)
        if record is None:
            return None

        return SessionInfo(
            name=record.name,
            status=record.status,
            branch=record.branch,
            worktree_path=record.worktree_path,
            ready_to_merge=record.status == "reviewed",
            created_at=record.created_at,
        )

    def list_sessions(self, filter_type: str = "all") -> list[SessionInfo]:
        """List all sessions with optional filtering."""
        records = self._store.list()
        sessions = []

        for record in records:
            if filter_type == "active" and record.status != "running":
                continue
            if filter_type == "reviewed" and record.status != "reviewed":
                continue

            sessions.append(SessionInfo(
                name=record.name,
                status=record.status,
                branch=record.branch,
                worktree_path=record.worktree_path,
                ready_to_merge=record.status == "reviewed",
                created_at=record.created_at,
            ))

        return sessions

    def get_diff(self, session_name: str) -> DiffResult:
        """Get the diff for a session against its parent branch."""
        record = self._store.get(session_name)
        if record is None or record.worktree_path is None:
            return DiffResult(files=[], content="")

        worktree_path = Path(record.worktree_path)
        if not worktree_path.exists():
            return DiffResult(files=[], content="")

        # Find the merge-base with the parent branch
        parent_branch = self._get_parent_branch()
        merge_base_result = self._run_git(
            "merge-base", parent_branch, "HEAD",
            cwd=worktree_path,
            check=False,
        )

        if merge_base_result.returncode != 0:
            return DiffResult(files=[], content="")

        merge_base = merge_base_result.stdout.strip()

        # Get list of changed files
        diff_names_result = self._run_git(
            "diff", "--name-only", merge_base, "HEAD",
            cwd=worktree_path,
            check=False,
        )
        files = [f for f in diff_names_result.stdout.strip().split("\n") if f]

        # Get full diff content
        diff_result = self._run_git(
            "diff", merge_base, "HEAD",
            cwd=worktree_path,
            check=False,
        )
        content = diff_result.stdout

        return DiffResult(files=files, content=content)

    def merge_session(self, name: str, commit_message: str) -> None:
        """Squash-merge session changes to the parent branch."""
        record = self._store.get(name)
        if record is None:
            raise ValueError(f"Session not found: {name}")

        branch = record.branch
        worktree_path = record.worktree_path

        # Squash merge the branch
        self._run_git("merge", "--squash", branch)
        self._run_git("commit", "-m", commit_message)

        # Clean up worktree
        if worktree_path and Path(worktree_path).exists():
            self._run_git("worktree", "remove", "--force", worktree_path, check=False)

        self._run_git("worktree", "prune", check=False)

        # Delete branch
        self._run_git("branch", "-D", branch, check=False)

        # Remove metadata
        self._store.delete(name)
