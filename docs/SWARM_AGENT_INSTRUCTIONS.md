# Swarm Agent Instructions

> **CRITICAL**: You are one of multiple agents working on the same task concurrently. Follow these instructions EXACTLY to coordinate with other agents.

---

## Your Identity

- **Task ID**: `{task_id}`
- **Agent ID**: `{agent_id}`
- **Total Agents**: {agent_count}

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

### Quality Checklist

Before committing, verify:
- [ ] All tests pass (run the test suite)
- [ ] Code compiles/runs without errors
- [ ] Edge cases are handled
- [ ] Error handling is included
- [ ] Code is readable and maintainable
- [ ] Changes are committed (not just staged)

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
| `false` | **WAIT and POLL** - Call `get_all_implementations` periodically until it succeeds |
| `true` | **PROCEED** to Phase 3 immediately |

---

## Phase 2.5: Waiting for Other Agents

> **IMPORTANT**: Other agents work CONCURRENTLY. You must wait for them.

### Polling Strategy
```
while not all_finished:
    wait 10-15 seconds
    call get_all_implementations(task_id="{task_id}")
    if success: break
```

### Example Polling Call
```
get_all_implementations(task_id="{task_id}")
```

**If it returns an error** like `"Not all agents have finished yet (2/3)"`:
- This is NORMAL - other agents are still working
- Wait 10-15 seconds
- Try again

**If it returns success** with implementations:
- All agents are done
- Proceed to Phase 3

---

## Phase 3: Review and Vote

> **CRITICAL**: Only proceed here when `get_all_implementations` succeeds.

### Step 1: Review All Implementations
The `get_all_implementations` response looks like:
```json
{{
    "success": true,
    "implementations": [
        {{"agent_id": "{task_id}-agent-0", "implementation": "...diff..."}},
        {{"agent_id": "{task_id}-agent-1", "implementation": "...diff..."}},
        {{"agent_id": "{task_id}-agent-2", "implementation": "...diff..."}}
    ]
}}
```

### Step 2: Evaluate Each Solution
Consider:
| Criteria | Weight |
|----------|--------|
| **Correctness** | Does it solve the problem? |
| **Completeness** | Are all requirements met? |
| **Code Quality** | Is it clean and readable? |
| **Error Handling** | Are edge cases covered? |
| **Simplicity** | Is it unnecessarily complex? |

### Step 3: Cast Your Vote
```
cast_vote(
    task_id="{task_id}",
    agent_id="{agent_id}",
    voted_for="<agent_id of the BEST implementation>",
    reason="<brief explanation - 1-2 sentences>"
)
```

**RULES**:
- You MAY vote for yourself if your solution is genuinely best
- You MUST vote for exactly ONE agent
- Your vote is FINAL - no changes allowed
- Be OBJECTIVE - vote for quality, not familiarity

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
0:46                     Call finished_work (2 remaining)
0:47                     Poll... waiting
1:00    Finish impl                       Still working
1:01    Call finished_work (1 remaining)
1:02    Poll... waiting  Poll... waiting
1:30                                      Finish impl
1:31                                      Call finished_work (0 remaining!)
1:32    get_all_impl ✓   get_all_impl ✓   get_all_impl ✓
1:33    Review...        Review...        Review...
1:35    cast_vote        cast_vote        cast_vote
────────────────────────────────────────────────────
        CONSENSUS REACHED - Winner determined
```

---

## Summary Checklist

- [ ] **Phase 1**: Implement and commit your solution
- [ ] **Phase 2**: Call `finished_work` with your diff
- [ ] **Phase 2.5**: Poll `get_all_implementations` until it succeeds
- [ ] **Phase 3**: Review all solutions and call `cast_vote`

> **Remember**: You are part of a team. Your vote helps determine the best solution through consensus.
