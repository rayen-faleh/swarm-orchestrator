# Multi-Agent Consensus System v2 - Schaltwerk + Claude Code

## Vision

A lightweight orchestration layer that applies MAKER paper principles (redundant execution + voting) using **existing Claude Code instances** via Schaltwerk, rather than building custom agents from scratch.

---

## Why This Approach?

| Aspect | Original Plan (LangGraph + Ollama) | New Plan (Schaltwerk + Claude Code) |
|--------|-----------------------------------|-------------------------------------|
| **LOC** | ~1500+ | ~300-500 |
| **Agent Quality** | Local Qwen3 8B | Claude Sonnet/Opus |
| **Tooling** | Must build file ops, search, etc. | Built into Claude Code |
| **Time to PoC** | Days | Hours |
| **Complexity** | High (full agent system) | Low (orchestration only) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER QUERY                               │
│            "Add authentication to the Express app"               │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        ORCHESTRATOR                              │
│                    (Python CLI ~200 LOC)                         │
│                                                                  │
│  1. Decompose query into subtasks (Claude API)                   │
│  2. For each subtask, spawn N redundant Claude Code agents       │
│  3. Wait for completion                                          │
│  4. Compare outputs via git diff                                 │
│  5. Vote on consensus                                            │
│  6. Merge winning solution                                       │
└─────────────────────────────┬───────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│  Claude Code  │     │  Claude Code  │     │  Claude Code  │
│   Agent 1     │     │   Agent 2     │     │   Agent 3     │
│  (worktree)   │     │  (worktree)   │     │  (worktree)   │
│               │     │               │     │               │
│ schaltwerk/   │     │ schaltwerk/   │     │ schaltwerk/   │
│ task-1-a1     │     │ task-1-a2     │     │ task-1-a3     │
└───────┬───────┘     └───────┬───────┘     └───────┬───────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      VOTING / COMPARISON                         │
│                                                                  │
│  - Git diff each worktree against base                           │
│  - Group identical diffs                                         │
│  - Majority wins (2/3 or 3/5)                                    │
│  - If no consensus: return all candidates for human review       │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      CONSENSUS OUTPUT                            │
│                                                                  │
│  - Merge winning worktree to main                                │
│  - Report confidence score                                       │
│  - Clean up other worktrees                                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Components

### 1. Orchestrator CLI (`swarm_orchestrator.py`)

**~200 LOC** - The brain of the system

```python
#!/usr/bin/env python3
"""
Swarm Orchestrator - Multi-agent consensus using Schaltwerk + Claude Code
"""

import subprocess
import json
import hashlib
from anthropic import Anthropic

def decompose_task(query: str) -> list[dict]:
    """Use Claude to break query into subtasks."""
    client = Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": DECOMPOSE_PROMPT.format(query=query)}]
    )
    return json.loads(response.content[0].text)

def spawn_agents(subtask: dict, agent_count: int = 3) -> list[str]:
    """Spawn N Claude Code agents via Schaltwerk for the same subtask."""
    sessions = []
    for i in range(agent_count):
        session_name = f"{subtask['id']}-agent-{i}"
        # Create spec
        create_spec(session_name, subtask['prompt'])
        # Start agent
        start_agent(session_name)
        sessions.append(session_name)
    return sessions

def wait_for_completion(sessions: list[str]) -> list[dict]:
    """Poll Schaltwerk until all sessions complete."""
    # ... polling logic

def compare_outputs(sessions: list[str]) -> dict:
    """Compare git diffs from each session, group by similarity."""
    diffs = {}
    for session in sessions:
        diff = get_diff(session)
        diff_hash = hashlib.md5(diff.encode()).hexdigest()
        if diff_hash not in diffs:
            diffs[diff_hash] = {"diff": diff, "sessions": [], "count": 0}
        diffs[diff_hash]["sessions"].append(session)
        diffs[diff_hash]["count"] += 1
    return diffs

def vote(diffs: dict, threshold: int = 2) -> dict | None:
    """Return winning diff if majority exists."""
    for diff_hash, data in diffs.items():
        if data["count"] >= threshold:
            return data
    return None  # No consensus

def main(query: str, agents: int = 3):
    subtasks = decompose_task(query)

    for subtask in subtasks:
        sessions = spawn_agents(subtask, agents)
        wait_for_completion(sessions)
        diffs = compare_outputs(sessions)
        winner = vote(diffs, threshold=(agents // 2) + 1)

        if winner:
            merge_winner(winner["sessions"][0])
            cleanup_losers([s for s in sessions if s not in winner["sessions"]])
        else:
            print(f"No consensus for {subtask['id']} - manual review needed")
```

### 2. Decomposition Prompt

```python
DECOMPOSE_PROMPT = """
You are a task decomposer for software engineering tasks.

Given a coding task, determine if it should be:
1. ATOMIC - A single, focused change (one file, one function, one feature)
2. COMPLEX - Multiple independent changes that can be done separately

Output JSON:
{
  "is_atomic": true/false,
  "subtasks": [
    {
      "id": "task-1",
      "description": "Brief description",
      "prompt": "Detailed prompt for Claude Code agent"
    }
  ]
}

For ATOMIC tasks: Return single subtask with the original query as prompt.
For COMPLEX tasks: Break into 2-5 independent subtasks.

TASK: {query}
"""
```

### 3. Voting Logic (`voting.py`)

**~100 LOC** - Compare and vote on outputs

```python
def normalize_diff(diff: str) -> str:
    """Normalize diff for comparison (ignore timestamps, whitespace variations)."""
    lines = []
    for line in diff.split('\n'):
        # Skip diff metadata lines
        if line.startswith('diff --git') or line.startswith('index '):
            continue
        if line.startswith('@@'):
            # Normalize hunk headers
            line = re.sub(r'@@ .* @@', '@@', line)
        lines.append(line.rstrip())
    return '\n'.join(lines)

def hash_diff(diff: str) -> str:
    """Create hash of normalized diff for grouping."""
    normalized = normalize_diff(diff)
    return hashlib.sha256(normalized.encode()).hexdigest()

def group_by_similarity(sessions: list[str]) -> dict[str, list[str]]:
    """Group sessions by their diff output."""
    groups = {}
    for session in sessions:
        diff = get_session_diff(session)
        diff_hash = hash_diff(diff)
        if diff_hash not in groups:
            groups[diff_hash] = []
        groups[diff_hash].append(session)
    return groups

def find_consensus(groups: dict, min_votes: int) -> str | None:
    """Return winning session if consensus reached."""
    for diff_hash, sessions in groups.items():
        if len(sessions) >= min_votes:
            return sessions[0]  # Return first matching session
    return None
```

### 4. Schaltwerk Integration (`schaltwerk_client.py`)

**~100 LOC** - Wrapper around Schaltwerk MCP calls

```python
import subprocess
import json

def run_mcp_command(tool: str, params: dict) -> dict:
    """Execute Schaltwerk MCP command via claude CLI."""
    # This could use the MCP client directly or shell out to claude
    cmd = ["claude", "mcp", "schaltwerk", tool, json.dumps(params)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return json.loads(result.stdout)

def create_spec(name: str, content: str) -> dict:
    return run_mcp_command("schaltwerk_spec_create", {
        "name": name,
        "content": content
    })

def start_agent(session_name: str) -> dict:
    return run_mcp_command("schaltwerk_draft_start", {
        "session_name": session_name,
        "agent_type": "claude",
        "skip_permissions": True
    })

def get_session_status(session_name: str) -> dict:
    result = run_mcp_command("schaltwerk_list", {"json": True})
    for session in result["sessions"]:
        if session["name"] == session_name:
            return session
    return None

def get_diff_summary(session_name: str) -> dict:
    return run_mcp_command("schaltwerk_diff_summary", {
        "session": session_name
    })

def merge_session(session_name: str, message: str) -> dict:
    return run_mcp_command("schaltwerk_merge_session", {
        "session_name": session_name,
        "mode": "squash",
        "commit_message": message,
        "cancel_after_merge": True
    })

def cancel_session(session_name: str) -> dict:
    return run_mcp_command("schaltwerk_cancel", {
        "session_name": session_name,
        "force": True
    })
```

---

## Project Structure

```
swarm/
├── PLAN_V2_SCHALTWERK.md      # This plan
├── src/
│   └── swarm_orchestrator/
│       ├── __init__.py
│       ├── cli.py              # Click CLI entry point (~50 LOC)
│       ├── orchestrator.py     # Main orchestration logic (~150 LOC)
│       ├── decomposer.py       # Task decomposition (~50 LOC)
│       ├── voting.py           # Diff comparison & voting (~100 LOC)
│       └── schaltwerk.py       # Schaltwerk MCP wrapper (~100 LOC)
└── tests/
    ├── test_decomposer.py
    ├── test_voting.py
    └── test_orchestrator.py
```

**Total: ~450 LOC** (vs ~1500 LOC in original plan)

---

## Implementation Phases

### Phase 1: Core Orchestrator (2-3 hours)

**Goal:** Basic end-to-end flow working

1. **Decomposer** - Claude API call to break down tasks
2. **Schaltwerk wrapper** - Spawn agents, check status, get diffs
3. **Simple voting** - Hash-based diff comparison
4. **CLI** - Basic `swarm run "query"` command

### Phase 2: Robustness (2-3 hours)

**Goal:** Handle edge cases and improve reliability

1. **Polling with timeout** - Don't wait forever for agents
2. **Error handling** - Graceful failures, retries
3. **Diff normalization** - Ignore irrelevant differences
4. **Logging** - Track what's happening

### Phase 3: Polish (1-2 hours)

**Goal:** Nice UX and reporting

1. **Progress output** - Show what's happening in real-time
2. **Confidence reporting** - Show vote distribution
3. **Manual review mode** - When no consensus, present options

---

## User Stories

### US-1: Basic Orchestration

**As a** developer
**I want** to run `swarm "Add a login endpoint"`
**So that** multiple Claude Code agents work on it and I get the consensus solution

**Acceptance Criteria:**
- [ ] CLI accepts query string
- [ ] Query is decomposed (or marked atomic)
- [ ] 3 Claude Code agents spawn via Schaltwerk
- [ ] System waits for completion
- [ ] Diffs are compared
- [ ] Winner is merged if consensus
- [ ] Losers are cleaned up

### US-2: Voting & Consensus

**As a** developer
**I want** the system to identify when agents agree
**So that** I get reliable output backed by consensus

**Acceptance Criteria:**
- [ ] Diffs are normalized before comparison
- [ ] Identical diffs are grouped
- [ ] Majority (2/3 or 3/5) determines winner
- [ ] Confidence score is reported
- [ ] No consensus triggers manual review prompt

### US-3: Task Decomposition

**As a** developer
**I want** complex tasks broken into subtasks
**So that** each part gets focused attention and consensus

**Acceptance Criteria:**
- [ ] Simple tasks stay atomic
- [ ] Complex tasks split into 2-5 subtasks
- [ ] Each subtask processed independently
- [ ] Results aggregated at the end

---

## CLI Interface

```bash
# Basic usage
swarm run "Add user authentication with JWT"

# With options
swarm run "Refactor the database layer" --agents 5 --timeout 600

# Dry run (just decompose, don't execute)
swarm decompose "Build a REST API for todos"

# Check status of running swarm
swarm status

# Manual vote selection (when no consensus)
swarm review task-1
```

**Output Example:**
```
🐝 Swarm Orchestrator v0.1.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 Task: Add user authentication with JWT

🔍 Decomposing task...
   → Atomic task, no decomposition needed

🚀 Spawning 3 Claude Code agents...
   ✓ auth-agent-0 started
   ✓ auth-agent-1 started
   ✓ auth-agent-2 started

⏳ Waiting for completion...
   ✓ auth-agent-0 completed (2m 34s)
   ✓ auth-agent-1 completed (2m 41s)
   ✓ auth-agent-2 completed (2m 38s)

🗳️  Voting on outputs...
   Agent 0: +145 -12 lines (hash: a3f2c...)
   Agent 1: +145 -12 lines (hash: a3f2c...)
   Agent 2: +152 -15 lines (hash: b7d1e...)

✅ CONSENSUS REACHED (2/3 agents agree)
   Confidence: 66.7%
   Winner: auth-agent-0

🔀 Merging winning solution...
   ✓ Merged to main branch

🧹 Cleaning up...
   ✓ Cancelled auth-agent-1
   ✓ Cancelled auth-agent-2

Done! 🎉
```

---

## Testing Strategy

### Unit Tests

```python
# test_decomposer.py
def test_atomic_task_not_decomposed():
    result = decompose("Write a function to check if number is prime")
    assert result["is_atomic"] == True
    assert len(result["subtasks"]) == 1

def test_complex_task_decomposed():
    result = decompose("Build a REST API with auth, CRUD, and tests")
    assert result["is_atomic"] == False
    assert 2 <= len(result["subtasks"]) <= 5

# test_voting.py
def test_unanimous_consensus():
    groups = {"hash1": ["a", "b", "c"]}
    winner = find_consensus(groups, min_votes=2)
    assert winner == "a"

def test_no_consensus():
    groups = {"h1": ["a"], "h2": ["b"], "h3": ["c"]}
    winner = find_consensus(groups, min_votes=2)
    assert winner is None

def test_diff_normalization():
    diff1 = "@@  -1,5 +1,6 @@\n+new line"
    diff2 = "@@ -1,5 +1,6 @@\n+new line  "
    assert hash_diff(diff1) == hash_diff(diff2)
```

### Integration Tests

```python
@pytest.mark.integration
def test_end_to_end_atomic():
    result = run_swarm("Write an is_prime function", agents=3)
    assert result["consensus"] == True
    assert result["confidence"] >= 0.66

@pytest.mark.integration
def test_end_to_end_complex():
    result = run_swarm("Add logging and error handling", agents=3)
    assert len(result["subtasks"]) >= 2
```

---

## Dependencies

```toml
[project]
name = "swarm-orchestrator"
version = "0.1.0"
requires-python = ">=3.11"

dependencies = [
    "anthropic>=0.40.0",     # Claude API for decomposition
    "click>=8.1.0",           # CLI framework
    "rich>=13.0.0",           # Pretty terminal output
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
]

[project.scripts]
swarm = "swarm_orchestrator.cli:main"
```

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Agents take too long | Medium | Medium | Timeout + parallel execution |
| No consensus reached | Medium | Low | Manual review fallback |
| Schaltwerk API changes | Low | High | Thin wrapper layer |
| Claude API rate limits | Low | Medium | Retry logic + backoff |
| Diff comparison too strict | Medium | Medium | Normalization + fuzzy matching |

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Consensus rate | >70% of tasks |
| Time to completion | <5min for simple tasks |
| Code quality | Passes linting/tests |
| LOC | <500 total |

---

## Next Steps

1. **Create project structure** with uv
2. **Implement decomposer** (Claude API call)
3. **Implement Schaltwerk wrapper** (spawn, poll, diff)
4. **Implement voting** (hash comparison)
5. **Create CLI** (Click)
6. **Test with simple task** (is_prime function)

---

## Progress Tracker

```
Phase 1: [░░░░░░░░░░] 0%  - Core Orchestrator
Phase 2: [░░░░░░░░░░] 0%  - Robustness
Phase 3: [░░░░░░░░░░] 0%  - Polish

Overall: [░░░░░░░░░░] 0%
```
