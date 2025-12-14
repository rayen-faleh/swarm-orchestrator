"""
Main orchestrator that coordinates decomposition, agent spawning, and voting.
"""

import time
import uuid
from dataclasses import dataclass
from typing import Callable, Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .decomposer import Subtask, decompose_task, DecompositionResult
from .schaltwerk import SchaltwerkClient, get_client
from .voting import VoteResult, find_consensus, format_vote_summary
from .swarm_mcp.server import SwarmMCPServer


# Agent prompt template with MCP tool instructions
# References the detailed instructions file which agents will read
AGENT_PROMPT_TEMPLATE = """# Task

{task_prompt}

---

# CRITICAL: Swarm Agent Coordination

You are **Agent `{agent_id}`** working on **Task `{task_id}`** as part of a {agent_count}-agent swarm.

## READ THE INSTRUCTIONS CAREFULLY

@docs/SWARM_AGENT_INSTRUCTIONS.md

The file above contains MANDATORY instructions for:
1. How to implement your solution
2. How to signal completion using `finished_work` MCP tool
3. How to wait for other agents (they work CONCURRENTLY)
4. How to review and vote using `cast_vote` MCP tool

## Your Identifiers (USE EXACTLY)

| Field | Value |
|-------|-------|
| task_id | `{task_id}` |
| agent_id | `{agent_id}` |
| agent_count | {agent_count} |

## Quick Reference

```
Phase 1: Implement → commit your code
Phase 2: Call finished_work(task_id="{task_id}", agent_id="{agent_id}", implementation="<diff>")
Phase 2.5: Poll get_all_implementations until success (other agents are concurrent!)
Phase 3: Call cast_vote(task_id="{task_id}", agent_id="{agent_id}", voted_for="<best>", reason="<why>")
```

**START NOW**: Begin with Phase 1 - implement the task above.
"""


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
        state_file: str = ".swarm/state.json",
        auto_merge: bool = False,
    ):
        self.agent_count = agent_count
        self.timeout = timeout
        self.console = console or Console()
        self.client = get_client(timeout=timeout)
        self.swarm_server = SwarmMCPServer(persistence_path=state_file)
        self._poll_interval = 5  # seconds between completion checks
        self.auto_merge = auto_merge

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

        # Step 2-5: Process each subtask SEQUENTIALLY
        # User must merge each winner before the next subtask starts
        subtask_results = []
        total_subtasks = len(decomposition.subtasks)

        for idx, subtask in enumerate(decomposition.subtasks):
            subtask_num = idx + 1
            is_last = (subtask_num == total_subtasks)

            # Show progress for multi-subtask workflows
            if total_subtasks > 1:
                self.console.print(f"\n{'━' * 50}")
                self.console.print(
                    f"[bold]📦 Subtask {subtask_num}/{total_subtasks}[/]"
                )

            result = self._process_subtask_with_mcp(subtask)
            subtask_results.append(result)

            # If there are more subtasks and this one succeeded, wait for user to merge
            if not is_last and result.vote_result.consensus_reached and result.winner_session:
                self._wait_for_user_merge(result.winner_session, subtask_num, total_subtasks)

        # Determine overall success based on consensus (not merge - user merges manually)
        overall_success = all(r.vote_result.consensus_reached for r in subtask_results)

        return OrchestrationResult(
            query=query,
            decomposition=decomposition,
            subtask_results=subtask_results,
            overall_success=overall_success,
        )

    def _decompose(self, query: str) -> DecompositionResult:
        """Decompose the query into subtasks."""
        return decompose_task(query, timeout=self.timeout)

    def _wait_for_user_merge(
        self,
        winner_session: str,
        current_subtask: int,
        total_subtasks: int,
    ) -> None:
        """
        Wait for user to merge the winner before continuing to next subtask.

        This ensures the codebase has the changes from the current subtask
        before agents start working on the next subtask (which may depend on it).
        """
        self.console.print("\n" + "═" * 50)
        self.console.print(
            f"[bold yellow]⏸️  WAITING FOR MERGE[/] "
            f"(Subtask {current_subtask}/{total_subtasks} complete)"
        )
        self.console.print("═" * 50)
        self.console.print(
            f"\n[bold]The next subtask may depend on this one's changes.[/]"
        )
        self.console.print(
            f"Please review and merge the winner before continuing:\n"
        )
        self.console.print(f"   [cyan]Winner session:[/] {winner_session}")
        self.console.print(f"   [dim]To review:[/]  schaltwerk diff {winner_session}")
        self.console.print(f"   [dim]To merge:[/]   schaltwerk merge {winner_session}")
        self.console.print("")
        self.console.print(
            f"[bold]After merging, type 'continue' to proceed to subtask "
            f"{current_subtask + 1}/{total_subtasks}:[/]"
        )

        while True:
            try:
                user_input = input("\n> ").strip().lower()
                if user_input == "continue":
                    self.console.print(
                        f"\n[green]✓ Continuing to next subtask...[/]\n"
                    )
                    break
                elif user_input in ("quit", "exit", "q"):
                    self.console.print("\n[yellow]Orchestration paused by user.[/]")
                    raise KeyboardInterrupt("User requested exit")
                else:
                    self.console.print(
                        f"[dim]Type 'continue' after merging, or 'quit' to exit.[/]"
                    )
            except EOFError:
                # Handle non-interactive mode
                self.console.print(
                    "\n[yellow]Non-interactive mode: auto-continuing...[/]"
                )
                break

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

    def _cleanup_sessions(self, sessions: list[str], keep_session: str | None = None) -> None:
        """
        Cancel and clean up sessions.

        Fetches actual session names from Schaltwerk since it adds suffixes.

        Args:
            sessions: List of session name prefixes to clean up
            keep_session: Optional session prefix to keep (the winner)
        """
        if not sessions:
            return

        self.console.print("   🧹 Cleaning up...")

        # Fetch actual session list from Schaltwerk
        all_sessions = self.client.get_session_list()

        for session_prefix in sessions:
            # Skip if this is the winner we want to keep
            if keep_session and session_prefix.startswith(keep_session):
                continue

            # Find the actual session name (Schaltwerk adds suffixes like -zd, -ft)
            actual_session = None
            for s in all_sessions:
                # Match by prefix - the actual name starts with our requested name
                if s.name.startswith(session_prefix):
                    actual_session = s.name
                    break

            if not actual_session:
                self.console.print(f"      ✓ {session_prefix} (already cleaned)")
                continue

            try:
                self.client.cancel_session(actual_session, force=True)
                self.console.print(f"      ✓ Cancelled {actual_session}")
            except Exception as e:
                error_str = str(e).lower()
                if "not found" in error_str:
                    self.console.print(f"      ✓ {actual_session} (already cleaned)")
                else:
                    self.console.print(f"[yellow]      ⚠ Could not cancel {actual_session}: {e}[/]")

    # =========================================================================
    # MCP-based methods (new flow with agent-to-agent coordination)
    # =========================================================================

    def _generate_task_id(self, subtask: Subtask, index: int = 0) -> str:
        """Generate a unique task ID for a subtask."""
        short_uuid = str(uuid.uuid4())[:8]
        return f"{subtask.id}-{short_uuid}"

    def _generate_agent_prompt(
        self,
        subtask: Subtask,
        task_id: str,
        agent_id: str,
    ) -> str:
        """Generate the agent prompt with MCP tool instructions."""
        return AGENT_PROMPT_TEMPLATE.format(
            task_prompt=subtask.prompt,
            task_id=task_id,
            agent_id=agent_id,
            agent_count=self.agent_count,
        )

    def _spawn_agents_with_mcp(
        self,
        subtask: Subtask,
        task_id: str,
    ) -> list[str]:
        """Spawn agents with MCP coordination enabled."""
        self.console.print(f"   🚀 Spawning {self.agent_count} agents...")

        # Register task with swarm server
        task = self.swarm_server.create_task(
            task_id=task_id,
            agent_count=self.agent_count,
            session_prefix=task_id,
        )

        sessions = []
        for agent_id in task.agent_ids:
            requested_name = task.session_names[agent_id]

            # Generate prompt with MCP instructions
            prompt = self._generate_agent_prompt(subtask, task_id, agent_id)

            # Create spec with enhanced prompt
            self.client.create_spec(requested_name, prompt)

            # Start agent - Schaltwerk may add a suffix to the name
            result = self.client.start_agent(requested_name, skip_permissions=True)

            # Get the actual session name from the result (Schaltwerk adds suffix)
            actual_name = requested_name
            if isinstance(result, dict):
                # Try common response field names for the actual session name
                actual_name = (
                    result.get("session_name") or
                    result.get("name") or
                    result.get("session") or
                    requested_name
                )

            # Update task's session_names mapping with actual name
            task.session_names[agent_id] = actual_name

            sessions.append(actual_name)
            self.console.print(f"      ✓ {actual_name} started (agent: {agent_id})")

        # Save updated session names to state
        self.swarm_server.state._auto_save()

        return sessions

    def _wait_for_mcp_completion(self, task_id: str) -> None:
        """Wait for all agents to signal completion via MCP.

        No timeout - agents can take as long as needed to implement their solutions.
        """
        self.console.print("   ⏳ Waiting for agents to finish (no timeout - agents work at their own pace)...")

        start_time = time.time()
        last_status = {}
        last_heartbeat = time.time()
        heartbeat_interval = 60  # Show status every 60 seconds

        while True:
            task = self.swarm_server.state.get_task(task_id)
            if not task:
                raise ValueError(f"Task {task_id} not found")

            # Report progress when agents finish
            for agent_id in task.agent_ids:
                status = task.agent_statuses.get(agent_id)
                if status and agent_id not in last_status:
                    self.console.print(f"      ✓ {agent_id} finished")
                    last_status[agent_id] = status

            if task.all_agents_finished():
                elapsed = int(time.time() - start_time)
                self.console.print(f"      All agents finished! (took {elapsed}s)")
                return

            # Periodic heartbeat to show we're still waiting
            if time.time() - last_heartbeat > heartbeat_interval:
                elapsed = int(time.time() - start_time)
                finished = len(last_status)
                total = len(task.agent_ids)
                self.console.print(f"      [dim]... still waiting ({finished}/{total} done, {elapsed}s elapsed)[/]")
                last_heartbeat = time.time()

            time.sleep(self._poll_interval)

    def _wait_for_mcp_votes(self, task_id: str) -> dict:
        """Wait for all agents to cast their votes via MCP.

        No timeout - agents need time to review all implementations before voting.
        """
        self.console.print("   🗳️  Waiting for agents to vote...")

        start_time = time.time()
        last_votes = set()
        last_heartbeat = time.time()
        heartbeat_interval = 60  # Show status every 60 seconds

        while True:
            task = self.swarm_server.state.get_task(task_id)
            if not task:
                raise ValueError(f"Task {task_id} not found")

            # Report progress
            for voter_id in task.votes:
                if voter_id not in last_votes:
                    vote = task.votes[voter_id]
                    self.console.print(
                        f"      ✓ {voter_id} voted for {vote.voted_for}"
                    )
                    last_votes.add(voter_id)

            if task.all_agents_voted():
                elapsed = int(time.time() - start_time)
                self.console.print(f"      All votes are in! (took {elapsed}s)")
                return self.swarm_server.state.get_vote_results(task_id)

            # Periodic heartbeat to show we're still waiting
            if time.time() - last_heartbeat > heartbeat_interval:
                elapsed = int(time.time() - start_time)
                voted = len(last_votes)
                total = len(task.agent_ids)
                self.console.print(f"      [dim]... still waiting for votes ({voted}/{total} voted, {elapsed}s elapsed)[/]")
                last_heartbeat = time.time()

            time.sleep(self._poll_interval)

    def _get_winner_session(self, task_id: str, winner_agent_id: str) -> str:
        """Get the actual Schaltwerk session name for the winning agent."""
        task = self.swarm_server.state.get_task(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")

        # Get the session name prefix we used
        session_prefix = task.session_names.get(winner_agent_id, winner_agent_id)

        # Find the actual session name from Schaltwerk (it adds suffixes)
        all_sessions = self.client.get_session_list()
        for s in all_sessions:
            if s.name.startswith(session_prefix):
                return s.name

        # Fallback to prefix if not found
        return session_prefix

    def _process_subtask_with_mcp(self, subtask: Subtask) -> SubtaskResult:
        """Process a subtask using the new MCP-based flow."""
        self.console.print(f"\n[bold cyan]Processing:[/] {subtask.id}")

        # Generate task ID
        task_id = self._generate_task_id(subtask)

        # Spawn agents with MCP-enabled prompts
        sessions = self._spawn_agents_with_mcp(subtask, task_id)

        # Wait for all agents to finish (via MCP signals)
        self._wait_for_mcp_completion(task_id)

        # Wait for all agents to vote (via MCP)
        vote_results = self._wait_for_mcp_votes(task_id)

        # Process results
        winner_session = None
        merged = False

        if vote_results.get("success") and vote_results.get("winner"):
            winner_agent_id = vote_results["winner"]
            winner_session = self._get_winner_session(task_id, winner_agent_id)

            vote_counts = vote_results.get("vote_counts", {})
            total = sum(vote_counts.values())
            winner_votes = vote_counts.get(winner_agent_id, 0)
            confidence = winner_votes / total if total > 0 else 0

            self.console.print(
                f"[green]✅ Consensus reached[/] "
                f"({winner_agent_id} with {winner_votes}/{total} votes)"
            )

            # Show vote breakdown
            for voter in vote_results.get("votes", []):
                self.console.print(
                    f"      {voter['voter']} → {voter['voted_for']}: {voter['reason'][:50]}..."
                )

            # Cleanup losing sessions, keep winner for review/merge
            # Note: sessions contains prefixes, winner_session is the actual name with suffix
            # Get the winner's prefix to filter it out
            task = self.swarm_server.state.get_task(task_id)
            winner_prefix = task.session_names.get(winner_agent_id, winner_agent_id) if task else winner_agent_id
            losing_prefixes = [s for s in sessions if s != winner_prefix]
            if losing_prefixes:
                self.console.print("   🧹 Cleaning up losing agent sessions...")
                self._cleanup_sessions(losing_prefixes)

            # Auto-merge if flag is set, otherwise leave for manual review
            if self.auto_merge:
                self.console.print(f"\n   [bold cyan]🔀 Auto-merging winner:[/] {winner_session}")
                merged = self._merge_winner(winner_session, subtask)
                if merged:
                    # Clean up winner session after successful merge
                    self._cleanup_sessions([winner_session])
                    self.console.print(f"   [green]✓ Successfully merged and cleaned up[/]")
            else:
                merged = False
                self.console.print(f"\n   [bold cyan]📋 Winner ready for review:[/] {winner_session}")
                self.console.print(f"   [dim]Review the changes, then merge manually when ready.[/]")
                self.console.print(f"\n   [dim]To review:[/]  schaltwerk diff {winner_session}")
                self.console.print(f"   [dim]To merge:[/]   schaltwerk merge {winner_session}")
                self.console.print(f"   [dim]To cancel:[/]  schaltwerk cancel {winner_session}")
        else:
            self.console.print("[yellow]⚠️  No clear winner - manual review needed[/]")
            self.console.print("   [dim]All sessions kept for manual review.[/]")

        # Create a compatible VoteResult for the return value
        from .voting import VoteGroup

        task = self.swarm_server.state.get_task(task_id)
        groups = []
        if task:
            for agent_id, impl in task.implementations.items():
                session = task.session_names.get(agent_id, agent_id)
                groups.append(VoteGroup(
                    diff_hash=agent_id,
                    diff_content=impl,
                    sessions=[session],
                ))

        # consensus_reached reflects if agents agreed, not if we merged
        consensus_reached = winner_session is not None
        vote_result = VoteResult(
            groups=groups,
            winner=groups[0] if winner_session and groups else None,
            total_votes=self.agent_count,
            consensus_reached=consensus_reached,
            confidence=vote_results.get("vote_counts", {}).get(
                vote_results.get("winner", ""), 0
            ) / self.agent_count if vote_results.get("winner") else 0,
        )

        return SubtaskResult(
            subtask=subtask,
            sessions=sessions,
            vote_result=vote_result,
            winner_session=winner_session,
            merged=merged,
        )


def run_swarm(query: str, agents: int = 3, timeout: int = 600) -> OrchestrationResult:
    """Convenience function to run the orchestrator."""
    orchestrator = Orchestrator(agent_count=agents, timeout=timeout)
    return orchestrator.run(query)
