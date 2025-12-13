"""
Main orchestrator that coordinates decomposition, agent spawning, and voting.
"""

from dataclasses import dataclass
from typing import Callable, Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .decomposer import Subtask, decompose_task, DecompositionResult
from .schaltwerk import SchaltwerkClient, get_client
from .voting import VoteResult, find_consensus, format_vote_summary


@dataclass
class SubtaskResult:
    """Result of processing a single subtask."""

    subtask: Subtask
    sessions: list[str]
    vote_result: VoteResult
    winner_session: str | None
    merged: bool


@dataclass
class OrchestrationResult:
    """Final result of the orchestration process."""

    query: str
    decomposition: DecompositionResult
    subtask_results: list[SubtaskResult]
    overall_success: bool


class Orchestrator:
    """
    Coordinates the multi-agent consensus workflow.

    1. Decompose task into subtasks
    2. Spawn N agents per subtask
    3. Wait for completion
    4. Vote on outputs
    5. Merge winners
    """

    def __init__(
        self,
        agent_count: int = 3,
        timeout: int = 600,
        console: Console | None = None,
    ):
        self.agent_count = agent_count
        self.timeout = timeout
        self.console = console or Console()
        self.client = get_client()

    def run(self, query: str) -> OrchestrationResult:
        """Execute the full orchestration workflow."""
        self.console.print(f"\n[bold blue]📋 Task:[/] {query}\n")

        # Step 1: Decompose
        self.console.print("[bold]🔍 Decomposing task...[/]")
        decomposition = self._decompose(query)

        if decomposition.is_atomic:
            self.console.print("   → Atomic task, no decomposition needed")
        else:
            self.console.print(
                f"   → Split into {len(decomposition.subtasks)} subtasks:"
            )
            for st in decomposition.subtasks:
                self.console.print(f"      • {st.id}: {st.description}")

        # Step 2-5: Process each subtask
        subtask_results = []
        for subtask in decomposition.subtasks:
            result = self._process_subtask(subtask)
            subtask_results.append(result)

        # Determine overall success
        overall_success = all(r.merged for r in subtask_results)

        return OrchestrationResult(
            query=query,
            decomposition=decomposition,
            subtask_results=subtask_results,
            overall_success=overall_success,
        )

    def _decompose(self, query: str) -> DecompositionResult:
        """Decompose the query into subtasks."""
        return decompose_task(query)

    def _process_subtask(self, subtask: Subtask) -> SubtaskResult:
        """Process a single subtask through agents and voting."""
        self.console.print(f"\n[bold cyan]Processing:[/] {subtask.id}")

        # Spawn agents
        sessions = self._spawn_agents(subtask)

        # Wait for completion
        self._wait_for_agents(sessions)

        # Get diffs and vote
        vote_result = self._vote_on_outputs(sessions)

        # Handle result
        winner_session = None
        merged = False

        if vote_result.consensus_reached and vote_result.winner:
            winner_session = vote_result.winner.sessions[0]
            self.console.print(
                f"[green]✅ Consensus reached[/] ({vote_result.confidence:.1%})"
            )

            # Merge winner
            merged = self._merge_winner(winner_session, subtask)

            # Cleanup losers
            self._cleanup_sessions(
                [s for s in sessions if s != winner_session]
            )
        else:
            self.console.print(
                f"[yellow]⚠️  No consensus[/] - manual review needed"
            )
            self.console.print(format_vote_summary(vote_result))

        return SubtaskResult(
            subtask=subtask,
            sessions=sessions,
            vote_result=vote_result,
            winner_session=winner_session,
            merged=merged,
        )

    def _spawn_agents(self, subtask: Subtask) -> list[str]:
        """Spawn N agents for a subtask."""
        self.console.print(f"   🚀 Spawning {self.agent_count} agents...")

        sessions = []
        for i in range(self.agent_count):
            session_name = f"{subtask.id}-agent-{i}"

            # Create spec
            self.client.create_spec(session_name, subtask.prompt)

            # Start agent
            self.client.start_agent(session_name, skip_permissions=True)

            sessions.append(session_name)
            self.console.print(f"      ✓ {session_name} started")

        return sessions

    def _wait_for_agents(self, sessions: list[str]) -> None:
        """Wait for all agents to complete."""
        self.console.print("   ⏳ Waiting for completion...")

        def on_complete(name: str, status) -> None:
            self.console.print(f"      ✓ {name} completed")

        try:
            self.client.wait_for_completion(sessions, callback=on_complete)
        except TimeoutError as e:
            self.console.print(f"[red]   ✗ Timeout: {e}[/]")
            raise

    def _vote_on_outputs(self, sessions: list[str]) -> VoteResult:
        """Collect diffs and vote on outputs."""
        self.console.print("   🗳️  Voting on outputs...")

        # Collect diffs
        sessions_with_diffs = {}
        for session in sessions:
            diff = self.client.get_full_diff(session)
            sessions_with_diffs[session] = diff

            # Show summary
            lines_added = diff.count("\n+")
            lines_removed = diff.count("\n-")
            self.console.print(
                f"      {session}: +{lines_added} -{lines_removed} lines"
            )

        # Vote
        return find_consensus(sessions_with_diffs)

    def _merge_winner(self, session: str, subtask: Subtask) -> bool:
        """Merge the winning session."""
        self.console.print(f"   🔀 Merging {session}...")

        try:
            commit_msg = (
                f"feat: {subtask.description}\n\n"
                f"Implemented via swarm consensus ({self.agent_count} agents)"
            )
            self.client.merge_session(session, commit_msg)
            self.console.print(f"      ✓ Merged to main")
            return True
        except Exception as e:
            self.console.print(f"[red]      ✗ Merge failed: {e}[/]")
            return False

    def _cleanup_sessions(self, sessions: list[str]) -> None:
        """Cancel and clean up sessions."""
        if not sessions:
            return

        self.console.print("   🧹 Cleaning up...")
        for session in sessions:
            try:
                self.client.cancel_session(session, force=True)
                self.console.print(f"      ✓ Cancelled {session}")
            except Exception as e:
                self.console.print(f"[yellow]      ⚠ Could not cancel {session}: {e}[/]")


def run_swarm(query: str, agents: int = 3, timeout: int = 600) -> OrchestrationResult:
    """Convenience function to run the orchestrator."""
    orchestrator = Orchestrator(agent_count=agents, timeout=timeout)
    return orchestrator.run(query)
