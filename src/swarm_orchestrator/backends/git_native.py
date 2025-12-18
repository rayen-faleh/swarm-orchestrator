"""
Git-native worktree backend implementation.

Provides WorktreeBackend using native git worktree commands instead of Schaltwerk MCP.
Also provides GitNativeAgentBackend for spawning Claude CLI agents directly.
"""

import os
import signal
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .base import WorktreeBackend, AgentBackend, SessionInfo, DiffResult, AgentStatus
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


class GitNativeAgentBackend(AgentBackend):
    """
    Git-native implementation of AgentBackend.

    Spawns Claude Code CLI agents directly in git worktree directories.
    """

    def __init__(
        self,
        worktree_base: Path | str | None = None,
        store: SessionStore | None = None,
    ):
        """
        Initialize the git-native agent backend.

        Args:
            worktree_base: Base directory for worktrees. Defaults to .swarm/worktrees.
            store: SessionStore for PID persistence. Creates one if not provided.
        """
        self._worktree_base = Path(worktree_base) if worktree_base else Path.cwd() / ".swarm" / "worktrees"
        self._store = store if store else SessionStore()
        self._processes: dict[str, subprocess.Popen] = {}

    def spawn_agent(self, session_name: str, prompt: str) -> str:
        """
        Spawn a Claude CLI agent in a new Terminal.app window.

        Opens a new macOS Terminal window using osascript and runs claude CLI with:
        - -p flag for non-interactive/print mode with prompt as argument
        - --dangerously-skip-permissions to bypass permission prompts

        Args:
            session_name: Session/worktree name
            prompt: The task prompt for the agent

        Returns:
            session_name as the agent identifier
        """
        worktree_path = self._worktree_base / session_name

        # Write prompt to file for reference/debugging
        prompt_file = worktree_path / ".swarm-prompt.md"
        prompt_file.write_text(prompt)

        # Escape prompt for shell (single quotes with proper escaping)
        escaped_prompt = prompt.replace("'", "'\"'\"'")

        # Build the claude command to run in terminal
        claude_cmd = f"cd '{worktree_path}' && claude -p '{escaped_prompt}' --dangerously-skip-permissions"

        # AppleScript to open a new Terminal window and run the command
        applescript = f'''
        tell application "Terminal"
            activate
            set newTab to do script "{claude_cmd.replace('"', '\\"')}"
            return id of window 1
        end tell
        '''

        # Run osascript to open Terminal and execute the command
        process = subprocess.Popen(
            ["osascript", "-e", applescript],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self._processes[session_name] = process

        # Persist PID for process management across restarts
        record = self._store.get(session_name)
        if record:
            record.pid = process.pid
            self._store.save(record)

        return session_name

    def wait_for_completion(
        self, agent_ids: list[str], timeout: int | None = None
    ) -> dict[str, AgentStatus]:
        """
        Wait for agents to complete by polling process status.

        Args:
            agent_ids: List of agent identifiers to wait for
            timeout: Optional timeout in seconds

        Returns:
            Dict mapping agent_id to their final status
        """
        import time

        results: dict[str, AgentStatus] = {}
        start_time = time.time()

        for agent_id in agent_ids:
            process = self._processes.get(agent_id)
            if not process:
                results[agent_id] = AgentStatus(agent_id=agent_id, is_finished=False)
                continue

            # Wait for process with timeout
            try:
                remaining_timeout = None
                if timeout:
                    elapsed = time.time() - start_time
                    remaining_timeout = max(0, timeout - elapsed)

                process.wait(timeout=remaining_timeout)
                is_finished = True

                # Read output from log file
                worktree_path = self._worktree_base / agent_id
                log_file = worktree_path / ".claude-output.log"
                implementation = None
                if log_file.exists():
                    implementation = log_file.read_text()

                results[agent_id] = AgentStatus(
                    agent_id=agent_id,
                    is_finished=is_finished,
                    implementation=implementation,
                )
            except subprocess.TimeoutExpired:
                results[agent_id] = AgentStatus(agent_id=agent_id, is_finished=False)

        return results

    def send_message(self, agent_id: str, message: str) -> None:
        """
        Send a message to a running agent's terminal window.

        Uses AppleScript to send keystrokes to the Terminal window.

        Args:
            agent_id: Agent to send message to
            message: Message content

        Raises:
            ValueError: If agent not found
        """
        if agent_id not in self._processes:
            raise ValueError(f"Agent not found: {agent_id}")

        # Escape message for AppleScript
        escaped_message = message.replace("\\", "\\\\").replace('"', '\\"')

        # Send keystrokes to Terminal
        applescript = f'''
        tell application "Terminal"
            activate
            tell application "System Events"
                keystroke "{escaped_message}"
                keystroke return
            end tell
        end tell
        '''
        subprocess.run(["osascript", "-e", applescript], check=False)

    def get_status(self, agent_id: str) -> AgentStatus:
        """
        Get the current status of an agent by checking poll().

        Args:
            agent_id: Agent identifier

        Returns:
            AgentStatus with is_finished based on process poll()
        """
        process = self._processes.get(agent_id)
        if not process:
            return AgentStatus(agent_id=agent_id, is_finished=False)

        return_code = process.poll()
        return AgentStatus(agent_id=agent_id, is_finished=return_code is not None)

    def stop_agent(self, session_name: str) -> bool:
        """
        Stop a running agent by closing its terminal window.

        Uses SIGTERM/SIGKILL for the stored PID. Also attempts to close
        any Terminal window running in the session's worktree.

        Args:
            session_name: Session/agent identifier to stop

        Returns:
            True if agent was stopped, False if not found or already stopped
        """
        stopped = False

        # Try in-memory process first (the osascript process)
        process = self._processes.get(session_name)
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            stopped = True

        if session_name in self._processes:
            del self._processes[session_name]

        # Kill the stored PID (the actual terminal process)
        record = self._store.get(session_name)
        if record and record.pid:
            try:
                os.kill(record.pid, signal.SIGTERM)
                stopped = True
            except ProcessLookupError:
                pass  # Process already dead
            except PermissionError:
                pass

            # Clear PID from store
            record.pid = None
            self._store.save(record)

        return stopped
