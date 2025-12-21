"""
Main orchestrator that coordinates decomposition, agent spawning, and voting.
"""

import select
import sys
import termios
import time
import tty
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Generator, Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .decomposer import Subtask, decompose_task, DecompositionResult, validate_decomposition, ExplorationResult
from .exploration import ExplorationExecutor, needs_exploration
from .schaltwerk import SchaltwerkClient, get_client
from .voting import VoteResult, find_consensus, format_vote_summary
from .swarm_mcp.server import SwarmMCPServer
from .config import SwarmConfig
from .backends import (
    WorktreeBackend,
    AgentBackend,
    SchaltwerkWorktreeBackend,
    SchaltwerkAgentBackend,
    GitNativeWorktreeBackend,
    CursorCLIAgentBackend,
    GitNativeWorktreeBackend,
    GitNativeAgentBackend,
)


# Agent prompt template with embedded MCP tool instructions
# Full instructions are embedded so the package works for all users
AGENT_PROMPT_TEMPLATE = """# Task

{task_prompt}
{exploration_context}
---

# CRITICAL: Swarm Agent Coordination

You are **Agent `{agent_id}`** working on **Task `{task_id}`** as part of a {agent_count}-agent swarm.

> **CRITICAL**: You are one of multiple agents working on the same task concurrently. Follow these instructions EXACTLY to coordinate with other agents.

---

## The Stakes: Winner Takes All

You are competing against **{agent_count} highly capable AI agents** - all working on the exact same task. When everyone finishes:

1. **Every agent reviews ALL implementations** (including yours)
2. **Every agent votes** for the best solution
3. **Majority wins** - that implementation gets merged
4. **All others are DISCARDED** - their work is deleted

### What This Means For You

| If You Win | If You Lose |
|------------|-------------|
| Your code gets merged | Your code is **deleted** |
| Your solution ships | Your work was for nothing |
| You solved the problem | You wasted the effort |

### How To Win

The agents judging your work are **smart and thorough**. They will evaluate:

- **Does it actually work?** - Broken code loses immediately
- **Is it complete?** - Partial solutions lose to complete ones
- **Is it well-tested?** - Untested code is risky, tested code wins
- **Is it clean?** - Readable, maintainable code beats clever hacks
- **Does it handle edge cases?** - Robust solutions beat fragile ones

### How To Lose

**Over-engineering is a losing strategy.** The reviewers will penalize:

- **Unnecessary abstractions** - Don't create helpers for one-time operations
- **Excessive boilerplate** - More code = more bugs = more risk
- **Features nobody asked for** - Solve the task, not imaginary future requirements
- **Verbose comments on obvious code** - If the code needs that many comments, simplify it
- **"Just in case" code** - Dead code paths, unused parameters, speculative features
- **Gold-plating** - Perfect is the enemy of done

> **The winning formula**: Solve the problem **exactly**. Not 80% of it. Not 150% of it. **100%** - complete, working, and nothing more.

### What Reviewers Are Looking For

```
❌ LOSES                              ✅ WINS
─────────────────────────────────────────────────────────
500 lines with abstractions          50 lines that just work
"Flexible" and "extensible"          Focused and purposeful
Handles hypothetical edge cases      Handles real edge cases
Comments explaining the obvious      Self-documenting code
Factory factories                    Direct implementation
"Production ready" boilerplate       Clean, minimal solution
```

> **Bottom line**: The best code is code that doesn't exist. Every line you write is a liability. Write exactly what's needed - creative, elegant, minimal - and ship it. **The smartest solution wins, not the longest.**

---

## Your Identity

| Field | Value |
|-------|-------|
| task_id | `{task_id}` |
| agent_id | `{agent_id}` |
| agent_count | {agent_count} |

**IMPORTANT**: Always use these EXACT IDs when calling MCP tools. Do not modify or guess IDs.

---

## Workflow Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  PHASE 1        │     │  PHASE 2        │     │  PHASE 3        │
│  Implement      │ ──▶ │  Signal Done    │ ──▶ │  Vote           │
│  (work alone)   │     │  (coordinate)   │     │  (after ALL done)│
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

---

## Phase 1: Implementation

> **CRITICAL**: You MUST commit your changes before proceeding to Phase 2. The merge will FAIL if you have uncommitted changes.

### Development Approach: Test-Driven Development (TDD)

When applicable, follow TDD:
1. **Write tests FIRST** - Define expected behavior before implementation
2. **Run tests** - Verify they fail (red)
3. **Implement** - Write minimal code to pass tests
4. **Run tests again** - Verify they pass (green)
5. **Refactor** - Clean up while keeping tests green

### Implementation Steps

1. **Understand the task** - Read requirements carefully
2. **Write tests first** (when applicable) - Define expected behavior
3. **Implement your solution** - Write clean, working code
4. **Run tests** - Ensure everything passes
5. **COMMIT your changes** - This is MANDATORY

### Git Commit Requirements

> ⚠️ **WARNING**: You MUST commit before calling `finished_work`. Uncommitted changes will cause merge failures.

```bash
# Stage your changes
git add -A

# Commit with a clear, descriptive message
git commit -m "feat: <brief description of what you implemented>

- <bullet point of key change 1>
- <bullet point of key change 2>
- <bullet point of key change 3>"
```

**Commit Message Guidelines**:
- Use conventional commit format: `feat:`, `fix:`, `refactor:`, `test:`
- First line: Brief summary (50 chars max)
- Body: Bullet points explaining key changes
- Be specific about WHAT changed and WHY

### Quality Checklist (Your Competitors Will Check All of These)

Before committing, verify - **because the other agents WILL**:
- [ ] All tests pass (run the test suite) - *broken tests = instant loss*
- [ ] Code compiles/runs without errors - *crashes = instant loss*
- [ ] Edge cases are handled - *fragile code loses to robust code*
- [ ] Error handling is included - *unhandled errors = amateur work*
- [ ] Code is readable and maintainable - *clever hacks lose to clean code*
- [ ] Changes are committed (not just staged) - *uncommitted = disqualified*

> **Ask yourself**: If a smart, critical reviewer looked at my code, would they find flaws? Fix them NOW, before voting.

### Verify Your Commit

```bash
# Check that changes are committed (should show nothing to commit)
git status

# Verify your commit exists
git log -1 --oneline
```

**DO NOT** proceed to Phase 2 until:
1. All tests pass
2. Your changes are COMMITTED (not just staged)
3. `git status` shows a clean working tree

---

## Phase 2: Signal Completion

> **CRITICAL**: You MUST call `finished_work` after committing. Other agents are waiting.

### Step 1: Get Your Diff
```bash
git diff HEAD~1
```

### Step 2: Call the MCP Tool
```
finished_work(
    task_id="{task_id}",
    agent_id="{agent_id}",
    implementation="<paste your git diff here>"
)
```

### Understanding the Response
The tool returns:
```json
{{
    "success": true,
    "agents_remaining": 2,
    "all_finished": false
}}
```

- `agents_remaining`: How many agents haven't finished yet
- `all_finished`: `true` when ALL agents are done

### What To Do Next

| `all_finished` | Action |
|----------------|--------|
| `false` | **WAIT** using `sleep 30`, then poll `get_all_implementations` |
| `true` | **PROCEED** to Phase 3 immediately |

---

## Phase 2.5: Waiting for Other Agents

> **IMPORTANT**: Other agents work CONCURRENTLY. You must wait for them.
> **DO NOT** call `get_all_implementations` immediately after `finished_work` - use `sleep` first!

### Polling Strategy

**CRITICAL**: Always sleep BEFORE polling, not after. This gives other agents time to finish.

```bash
# Step 1: Sleep first (REQUIRED - do not skip!)
sleep 30

# Step 2: Then check if all agents are done
get_all_implementations(task_id="{task_id}")

# Step 3: If not all finished, sleep and retry
# Repeat until success
```

### Example Workflow
```bash
# After calling finished_work, WAIT before polling:
sleep 30

# Then call the MCP tool:
get_all_implementations(task_id="{task_id}")

# If it fails (others not done), sleep again and retry:
sleep 30
get_all_implementations(task_id="{task_id}")
```

**If it returns an error** like `"Not all agents have finished yet (2/3)"`:
- This is NORMAL - other agents are still working
- Run `sleep 30` in your terminal
- Try `get_all_implementations` again

**If it returns success** with implementations:
- All agents are done
- Proceed to Phase 3

---

## Phase 3: Review and Vote

> **CRITICAL**: Only proceed here when `get_all_implementations` succeeds.

### Step 1: Quick Scan (Use the Summary!)

The `get_all_implementations` response provides **structured summaries** for quick comparison:

```json
{{
    "success": true,
    "implementations": [
        {{
            "agent_id": "{task_id}-agent-0",
            "session_name": "{task_id}-agent-0",
            "summary": {{
                "files_changed": ["src/foo.py", "tests/test_foo.py"],
                "stats": {{
                    "files_count": 2,
                    "added": 48,
                    "deleted": 12,
                    "total": 60
                }}
            }},
            "condensed_diff": "diff --git a/src/foo.py...\\n+new code\\n-old code"
        }}
    ]
}}
```

**Build a comparison table mentally:**

| Agent | Files | +Added | -Deleted | Total Changes |
|-------|-------|--------|----------|---------------|
| agent-0 | 2 | +48 | -12 | 60 |
| agent-1 | 5 | +120 | -45 | 165 |
| agent-2 | 2 | +30 | -8 | 38 |

**Quick red flags:**
- ❌ Too many files changed = over-engineered
- ❌ High total changes for simple task = bloated
- ⭐ Minimal changes that solve the problem = elegant

### Step 2: Review Condensed Diffs

The `condensed_diff` field contains a **minimal diff** with:
- No context lines (only changed lines)
- Normalized whitespace
- File headers preserved

This is much smaller than full git diff - focus on **what actually changed**.

### Step 3: Evaluate Each Solution

**Vote for the solution that best balances these criteria:**

| Criteria | Question to Ask | Red Flags |
|----------|-----------------|-----------|
| **Correctness** | Does it actually solve the problem? | Fails tests, wrong output, crashes |
| **Completeness** | Are ALL requirements met? | Missing features, partial implementation |
| **Elegance** | Is it the *smartest* solution? | Over-engineered, too many abstractions |
| **Minimalism** | Does it do exactly what's needed? | Unnecessary code, boilerplate, dead paths |
| **Robustness** | Does it handle *real* edge cases? | Fragile, unhandled errors |

> **Prefer**: A 50-line solution that works perfectly over a 500-line "enterprise" solution.
> **Penalize**: Over-engineering, unnecessary abstractions, speculative features, verbose comments on obvious code.

### Step 4: Cast Your Vote
```
cast_vote(
    task_id="{task_id}",
    agent_id="{agent_id}",
    voted_for="<agent_id of the BEST implementation>",
    reason="<brief explanation - 1-2 sentences>"
)
```

**RULES**:
- You CANNOT vote for yourself - the system will reject self-votes
- You MUST vote for exactly ONE other agent
- Your vote is FINAL - no changes allowed
- Be OBJECTIVE - vote for the best implementation among your competitors

---

## MCP Tools Reference

### `finished_work`
Signal that you completed your implementation.
```
finished_work(
    task_id: string,      # Your task ID
    agent_id: string,     # Your agent ID
    implementation: string # Your git diff
)
```

### `get_all_implementations`
Get all agents' implementations (only works when all are done).
```
get_all_implementations(
    task_id: string       # Your task ID
)
```

### `cast_vote`
Vote for the best implementation.
```
cast_vote(
    task_id: string,      # Your task ID
    agent_id: string,     # Your agent ID
    voted_for: string,    # Agent ID you're voting for
    reason: string        # Why this is the best
)
```

### `get_vote_results`
Check voting status (optional, for debugging).
```
get_vote_results(
    task_id: string       # Your task ID
)
```

---

## Error Handling

### "Task not found"
- Check your `task_id` is exactly: `{task_id}`
- Do not modify or guess the ID

### "Agent not found"
- Check your `agent_id` is exactly: `{agent_id}`
- Do not modify or guess the ID

### "Not all agents have finished"
- This is NORMAL during the waiting period
- Wait 10-15 seconds and retry

### "Already voted"
- You can only vote once
- Your vote has been recorded

---

## Timeline Example

```
Time    Agent-0          Agent-1          Agent-2
────────────────────────────────────────────────────
0:00    Start impl       Start impl       Start impl
0:45    Still working    Finish impl      Still working
0:46                     finished_work (2 remaining)
0:47                     sleep 30...
1:00    Finish impl                       Still working
1:01    finished_work (1 remaining)
1:17                     get_all (fail)
1:17                     sleep 30...
1:31                                      Finish impl
1:32    sleep 30...                       finished_work (0!)
1:47                     get_all ✓
1:48    (wakes up)                        sleep 30...
2:02    get_all ✓
2:18                                      get_all ✓
2:20    Review & vote    Review & vote    Review & vote
────────────────────────────────────────────────────
        CONSENSUS REACHED - Winner determined
```

---

## Summary Checklist

- [ ] **Phase 1**: Implement and commit your solution
- [ ] **Phase 2**: Call `finished_work` with your diff
- [ ] **Phase 2.5**: Run `sleep 30`, then poll `get_all_implementations` (repeat until success)
- [ ] **Phase 3**: Review all solutions and call `cast_vote`

---

## Final Reminder: This Is A Competition

> **{agent_count} agents. {agent_count} implementations. Only 1 winner.**
>
> The other agents are skilled, thorough, and motivated. They will scrutinize your code for any weakness. A single failing test, an unhandled edge case, or sloppy code could cost you the win.
>
> **Don't just finish. Win.**

---

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
        skip_exploration: bool = False,
        config: SwarmConfig | None = None,
        worktree_backend: WorktreeBackend | None = None,
        agent_backend: AgentBackend | None = None,
    ):
        self.agent_count = agent_count
        self.timeout = timeout
        self.console = console or Console()
        self.config = config or SwarmConfig()
        self.client = get_client(timeout=timeout)
        self.swarm_server = SwarmMCPServer(
            persistence_path=state_file,
            compression_config=self.config,
        )
        self._poll_interval = 5  # seconds between completion checks
        self.auto_merge = auto_merge
        self.skip_exploration = skip_exploration
        self._exploration_result: Optional[ExplorationResult] = None

        # Initialize backends based on config or explicit injection
        self._worktree_backend = worktree_backend or self._create_worktree_backend()
        self._agent_backend = agent_backend or self._create_agent_backend()

    def _create_worktree_backend(self) -> WorktreeBackend:
        """Create worktree backend based on config."""
        if self.config.worktree_backend == "schaltwerk":
            return SchaltwerkWorktreeBackend(self.client)
        if self.config.worktree_backend == "git-native":
            return GitNativeWorktreeBackend()
        raise ValueError(f"Unknown worktree backend: {self.config.worktree_backend}")

    def _create_agent_backend(self) -> AgentBackend:
        """Create agent backend based on config."""
        if self.config.agent_backend == "schaltwerk":
            return SchaltwerkAgentBackend(self.client)
        if self.config.agent_backend == "cursor-cli":
            return CursorCLIAgentBackend()
        if self.config.agent_backend == "git-native":
            return GitNativeAgentBackend(cli_tool=self.config.cli_tool)
        raise ValueError(f"Unknown agent backend: {self.config.agent_backend}")

    def _display_decomposition(self, decomposition: DecompositionResult) -> None:
        """Display enhanced decomposition details with scope information."""
        if decomposition.is_atomic:
            self.console.print("   → [green]Atomic task[/], no decomposition needed")
            subtask = decomposition.subtasks[0]
            self._display_subtask_scope(subtask, indent="   ")
        else:
            self.console.print(
                f"   → Split into [cyan]{len(decomposition.subtasks)}[/] sequential subtasks"
            )

            # Show reasoning if provided
            if decomposition.reasoning:
                self.console.print(f"   [dim]Reasoning: {decomposition.reasoning}[/]")

            # Show total estimated LOC
            total_loc = decomposition.total_estimated_loc()
            self.console.print(f"   [dim]Total estimated: ~{total_loc} LOC[/]")

            # Display each subtask with scope
            self.console.print("")
            for i, st in enumerate(decomposition.subtasks, 1):
                deps = f" [dim](depends on: {', '.join(st.depends_on)})[/]" if st.depends_on else ""
                self.console.print(f"   [bold]{i}. {st.title}[/]{deps}")
                self.console.print(f"      [dim]{st.description}[/]")
                self._display_subtask_scope(st, indent="      ")
                self.console.print("")

        # Validate and show warnings
        is_valid, warnings = validate_decomposition(decomposition)
        if warnings:
            self.console.print("   [yellow]⚠️  Scope warnings:[/]")
            for w in warnings:
                if "EXCEEDS LIMIT" in w:
                    self.console.print(f"      [red]• {w}[/]")
                else:
                    self.console.print(f"      [yellow]• {w}[/]")

    def _display_subtask_scope(self, subtask: Subtask, indent: str = "") -> None:
        """Display scope information for a subtask."""
        scope = subtask.scope
        files_str = ", ".join(scope.files[:3])
        if len(scope.files) > 3:
            files_str += f" (+{len(scope.files) - 3} more)"

        self.console.print(
            f"{indent}[dim]Scope: ~{scope.estimated_loc} LOC | "
            f"Files: {files_str or 'TBD'}[/]"
        )

    def run(self, query: str) -> OrchestrationResult:
        """Execute the full orchestration workflow."""
        self.console.print(f"\n[bold blue]📋 Task:[/] {query}\n")

        # Step 0: Exploration (if needed)
        self._exploration_result = self._explore(query)

        # Step 1: Decompose (with exploration context)
        self.console.print("[bold]🔍 Decomposing task...[/]")
        decomposition = self._decompose(query)

        # Display enhanced decomposition details
        self._display_decomposition(decomposition)

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
                    f"[bold]📦 Subtask {subtask_num}/{total_subtasks}: {subtask.title}[/]"
                )
                self.console.print(f"   [dim]{subtask.description}[/]")
                self._display_subtask_scope(subtask, indent="   ")

                # Show success criteria
                if subtask.success_criteria:
                    self.console.print(f"   [dim]Success criteria:[/]")
                    for criterion in subtask.success_criteria[:3]:  # Show first 3
                        self.console.print(f"   [dim]  • {criterion}[/]")
                    if len(subtask.success_criteria) > 3:
                        self.console.print(f"   [dim]  • (+{len(subtask.success_criteria) - 3} more)[/]")

            result = self._process_subtask_with_mcp(subtask)
            subtask_results.append(result)

            # If there are more subtasks and this one succeeded, wait for user to merge
            if not is_last and result.vote_result.consensus_reached and result.winner_session:
                self._wait_for_user_merge(result.winner_session, subtask_num, total_subtasks)

        # Determine overall success based on consensus (not merge - user merges manually)
        overall_success = all(r.vote_result.consensus_reached for r in subtask_results)

        # Launch post-completion dashboard when auto_merge=False and there are winners
        if not self.auto_merge:
            winner_sessions = [
                r.winner_session for r in subtask_results if r.winner_session
            ]
            if winner_sessions:
                self._launch_post_completion_dashboard(winner_sessions)

        return OrchestrationResult(
            query=query,
            decomposition=decomposition,
            subtask_results=subtask_results,
            overall_success=overall_success,
        )

    def _explore(self, query: str) -> Optional[ExplorationResult]:
        """Run exploration phase if needed."""
        if self.skip_exploration:
            self.console.print("[dim]⏭️  Skipping exploration (--skip-exploration)[/]")
            return None

        if not needs_exploration(query):
            self.console.print("[dim]⏭️  Skipping exploration (simple task detected)[/]")
            return None

        self.console.print("[bold]🔬 Exploring codebase...[/]")
        executor = ExplorationExecutor(timeout=self.timeout, model=self.config.exploration_model)
        result = executor.explore(query)

        if result.context_summary:
            self.console.print(f"   [dim]Context: {result.context_summary[:100]}...[/]" if len(result.context_summary) > 100 else f"   [dim]Context: {result.context_summary}[/]")
            if result.code_insights:
                self.console.print(f"   [dim]Found {len(result.code_insights)} relevant file(s)[/]")

        return result

    def _decompose(self, query: str) -> DecompositionResult:
        """Decompose the query into subtasks."""
        return decompose_task(
            query,
            timeout=self.timeout,
            exploration_result=self._exploration_result,
            cli_tool=self.config.cli_tool,
        )

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
        Cancel and clean up sessions using the configured worktree backend.

        Args:
            sessions: List of session name prefixes to clean up
            keep_session: Optional session prefix to keep (the winner)
        """
        if not sessions:
            return

        self.console.print("   🧹 Cleaning up...")

        # Fetch actual session list from worktree backend
        all_sessions = self._worktree_backend.list_sessions()

        for session_prefix in sessions:
            # Skip if this is the winner we want to keep
            if keep_session and session_prefix.startswith(keep_session):
                continue

            # Find the actual session name (some backends add suffixes)
            actual_session = None
            for s in all_sessions:
                # Match by prefix - the actual name starts with our requested name
                if s.name.startswith(session_prefix):
                    actual_session = s.name
                    break

            if not actual_session:
                self.console.print(f"      ✓ {session_prefix} (already cleaned)")
                continue

            # Stop agent first (closes Terminal window for git-native backend)
            try:
                self._agent_backend.stop_agent(actual_session)
            except Exception:
                pass  # Continue cleanup even if stop fails

            try:
                self._worktree_backend.delete_session(actual_session, force=True)
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
        exploration_result: Optional[ExplorationResult] = None,
    ) -> str:
        """Generate the agent prompt with MCP tool instructions."""
        exploration_context = self._format_exploration_context(exploration_result)

        return AGENT_PROMPT_TEMPLATE.format(
            task_prompt=subtask.prompt,
            exploration_context=exploration_context,
            task_id=task_id,
            agent_id=agent_id,
            agent_count=self.agent_count,
        )

    def _format_exploration_context(self, exploration_result: Optional[ExplorationResult]) -> str:
        """Format exploration results for embedding in agent prompts."""
        if not exploration_result:
            return ""

        sections = []

        if exploration_result.context_summary:
            sections.append(f"\n## Exploration Context\n{exploration_result.context_summary}")

        if exploration_result.code_insights:
            files_section = "\n## Relevant Files"
            for insight in exploration_result.code_insights:
                files_section += f"\n- **{insight.file_path}**: {insight.description}"
                if insight.patterns:
                    files_section += f" (patterns: {', '.join(insight.patterns)})"
            sections.append(files_section)

        if exploration_result.web_findings:
            web_section = "\n## Reference Documentation"
            for finding in exploration_result.web_findings:
                web_section += f"\n- [{finding.source}]({finding.source}): {finding.summary}"
            sections.append(web_section)

        return "\n".join(sections) + "\n" if sections else ""

    def _spawn_agents_with_mcp(
        self,
        subtask: Subtask,
        task_id: str,
    ) -> list[str]:
        """Spawn agents with MCP coordination enabled.

        Uses configured _worktree_backend for session/worktree creation
        and _agent_backend for agent spawning.
        """
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

            # Generate prompt with MCP instructions and exploration context
            prompt = self._generate_agent_prompt(subtask, task_id, agent_id, self._exploration_result)

            # Create session/worktree using configured backend
            session_info = self._worktree_backend.create_session(requested_name, prompt)

            # Spawn agent in the worktree using configured backend
            self._agent_backend.spawn_agent(requested_name, prompt)

            # Use the session name from the worktree backend
            actual_name = session_info.name

            # Update task's session_names mapping with actual name
            task.session_names[agent_id] = actual_name

            sessions.append(actual_name)
            self.console.print(f"      ✓ {actual_name} started (agent: {agent_id})")

        # Save updated session names to state
        self.swarm_server.state._auto_save()

        return sessions

    @contextmanager
    def _raw_mode(self) -> Generator[bool, None, None]:
        """Context manager to set terminal to cbreak mode for the duration.

        Yields True if raw mode was enabled, False otherwise (non-TTY).
        Terminal is restored on exit.
        """
        import io

        old_settings = None
        fd = None
        try:
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            tty.setcbreak(fd)
            yield True
        except (termios.error, OSError, io.UnsupportedOperation):
            # Not a TTY, pseudofile, or terminal functions unavailable
            yield False
        finally:
            if old_settings is not None and fd is not None:
                try:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                except (termios.error, OSError, io.UnsupportedOperation):
                    pass

    def _check_key_press(self) -> str | None:
        """Non-blocking check for keyboard input. Returns key if pressed, None otherwise.

        Note: Terminal must already be in cbreak/raw mode for this to detect
        single keypresses. Use within _raw_mode() context.
        """
        try:
            if select.select([sys.stdin], [], [], 0)[0]:
                return sys.stdin.read(1)
        except (OSError, ValueError):
            # stdin not readable or file descriptor issues
            pass
        return None

    def _show_watch_dashboard(self) -> None:
        """Show the interactive watch dashboard, returning when user presses 'q'."""
        from .tui import SessionsDashboard

        dashboard = SessionsDashboard(
            backend=self._worktree_backend,
            agent_backend=self._agent_backend,
        )
        dashboard.run()
        # After returning from dashboard, show resume message
        self.console.print("\n   [dim]Resumed run command. Press 'w' to open dashboard again.[/]")

    def _launch_post_completion_dashboard(self, winner_sessions: list[str]) -> None:
        """Launch the TUI dashboard after all subtasks complete for manual merge review.

        Displays guidance message and launches SessionsDashboard so user can
        review diffs and merge the winning session(s) before exiting.
        """
        from .tui import SessionsDashboard

        self.console.print("\n" + "═" * 50)
        self.console.print("[bold green]✅ All subtasks complete![/]")
        self.console.print("═" * 50)
        self.console.print("\n[bold]Winner session(s) ready for review and merge:[/]")
        for session in winner_sessions:
            self.console.print(f"   • {session}")
        self.console.print("\n[dim]Use the dashboard to review diffs and merge. Press 'q' to exit.[/]\n")

        dashboard = SessionsDashboard(
            backend=self._worktree_backend,
            agent_backend=self._agent_backend,
        )
        dashboard.run()

    def _wait_for_mcp_completion(self, task_id: str) -> None:
        """Wait for all agents to signal completion via MCP.

        No timeout - agents can take as long as needed to implement their solutions.
        Press 'w' to toggle the watch dashboard while waiting.
        """
        self.console.print("   ⏳ Waiting for agents to finish (no timeout - agents work at their own pace)...")
        self.console.print("   [dim]Press 'w' to open watch dashboard[/]")

        start_time = time.time()
        last_status = {}
        last_heartbeat = time.time()
        heartbeat_interval = 60  # Show status every 60 seconds

        with self._raw_mode():
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

                # Check for 'w' key to toggle watch dashboard
                key = self._check_key_press()
                if key == "w":
                    self._show_watch_dashboard()

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
        Press 'w' to toggle the watch dashboard while waiting.
        """
        self.console.print("   🗳️  Waiting for agents to vote...")
        self.console.print("   [dim]Press 'w' to open watch dashboard[/]")

        start_time = time.time()
        last_votes = set()
        last_heartbeat = time.time()
        heartbeat_interval = 60  # Show status every 60 seconds

        with self._raw_mode():
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

                # Check for 'w' key to toggle watch dashboard
                key = self._check_key_press()
                if key == "w":
                    self._show_watch_dashboard()

                # Periodic heartbeat to show we're still waiting
                if time.time() - last_heartbeat > heartbeat_interval:
                    elapsed = int(time.time() - start_time)
                    voted = len(last_votes)
                    total = len(task.agent_ids)
                    self.console.print(f"      [dim]... still waiting for votes ({voted}/{total} voted, {elapsed}s elapsed)[/]")
                    last_heartbeat = time.time()

                time.sleep(self._poll_interval)

    def _get_winner_session(self, task_id: str, winner_agent_id: str) -> str:
        """Get the actual session name for the winning agent."""
        task = self.swarm_server.state.get_task(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")

        # Get the session name prefix we used
        session_prefix = task.session_names.get(winner_agent_id, winner_agent_id)

        # Find the actual session name from worktree backend (some add suffixes)
        all_sessions = self._worktree_backend.list_sessions()
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
