"""
Cursor CLI agent backend implementation.

Provides CursorCLIAgentBackend for running cursor-agent in headless mode.
"""

import os
import subprocess
from pathlib import Path

from .base import AgentBackend, AgentStatus
from .cursor_auth import is_authenticated


class CursorCLIAgentBackend(AgentBackend):
    """
    Cursor CLI implementation of AgentBackend.

    Spawns cursor-agent processes in headless mode with JSON streaming output.
    """

    def __init__(self):
        """Initialize the Cursor CLI backend."""
        self._processes: dict[str, subprocess.Popen] = {}

    def _get_worktree_path(self, session_name: str) -> str:
        """Get worktree path for a session. Override in tests."""
        # Default: look for schaltwerk worktree
        base = Path(".schaltwerk/worktrees")
        # Find matching worktree directory
        if base.exists():
            for path in base.iterdir():
                if path.is_dir() and session_name in path.name:
                    return str(path)
        # Fallback to current directory
        return os.getcwd()

    def _check_auth(self) -> None:
        """
        Check authentication status before spawning agents.

        Raises:
            RuntimeError: If not authenticated, with guidance on how to authenticate.
        """
        if not is_authenticated():
            raise RuntimeError(
                "Cursor is not authenticated. Please authenticate using one of:\n"
                "  1. Run 'swarm cursor login' for browser-based authentication\n"
                "  2. Set CURSOR_API_KEY environment variable for API key authentication"
            )

    def spawn_agent(self, session_name: str, prompt: str) -> str:
        """
        Spawn a cursor-agent process.

        Writes prompt to .swarm-prompt.md and runs cursor-agent with:
        - -p flag for prompt file
        - --force to skip confirmations
        - --output-format stream-json for structured output

        Raises:
            RuntimeError: If not authenticated via CURSOR_API_KEY or browser login.
        """
        self._check_auth()
        worktree_path = self._get_worktree_path(session_name)

        # Write prompt to file
        prompt_file = Path(worktree_path) / ".swarm-prompt.md"
        prompt_file.write_text(prompt)

        # Build command
        cmd = [
            "cursor-agent",
            "-p", ".swarm-prompt.md",
            "--force",
            "--output-format", "stream-json",
        ]

        # Prepare environment with CURSOR_API_KEY
        env = os.environ.copy()

        # Start process
        process = subprocess.Popen(
            cmd,
            cwd=worktree_path,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self._processes[session_name] = process
        return session_name

    def wait_for_completion(
        self, agent_ids: list[str], timeout: int | None = None
    ) -> dict[str, AgentStatus]:
        """
        Wait for agents to complete using communicate() with timeout.

        Returns dict mapping agent_id to their final status.
        """
        results: dict[str, AgentStatus] = {}

        for agent_id in agent_ids:
            process = self._processes.get(agent_id)
            if not process:
                results[agent_id] = AgentStatus(
                    agent_id=agent_id,
                    is_finished=False,
                )
                continue

            try:
                stdout, stderr = process.communicate(timeout=timeout)
                is_finished = process.returncode is not None
                results[agent_id] = AgentStatus(
                    agent_id=agent_id,
                    is_finished=is_finished,
                    implementation=stdout.decode() if stdout else None,
                )
            except subprocess.TimeoutExpired:
                results[agent_id] = AgentStatus(
                    agent_id=agent_id,
                    is_finished=False,
                )

        return results

    def send_message(self, agent_id: str, message: str) -> None:
        """
        Send a message to a running agent.

        Not supported by cursor-agent CLI - raises NotImplementedError.
        """
        raise NotImplementedError(
            "Cursor CLI does not support sending messages to running agents"
        )

    def get_status(self, agent_id: str) -> AgentStatus:
        """
        Get the current status of an agent by checking poll().

        Returns AgentStatus with is_finished based on process poll().
        """
        process = self._processes.get(agent_id)
        if not process:
            return AgentStatus(agent_id=agent_id, is_finished=False)

        return_code = process.poll()
        is_finished = return_code is not None

        return AgentStatus(
            agent_id=agent_id,
            is_finished=is_finished,
        )

    def stop_agent(self, session_name: str) -> bool:
        """
        Stop a running cursor-agent process.

        Terminates the subprocess using SIGTERM with SIGKILL fallback.

        Args:
            session_name: Session/agent identifier to stop

        Returns:
            True if agent was stopped, False if not found or already stopped
        """
        process = self._processes.get(session_name)
        if not process:
            return False

        if process.poll() is not None:
            # Already finished
            del self._processes[session_name]
            return False

        # Terminate with SIGTERM, fallback to SIGKILL
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

        del self._processes[session_name]
        return True
