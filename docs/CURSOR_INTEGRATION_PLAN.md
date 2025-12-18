# Cursor Integration Plan for Swarm Orchestrator

This document outlines the available Cursor endpoints and proposes an integration strategy for the swarm-orchestrator multi-agent consensus system.

## Executive Summary

Cursor provides **two primary integration paths**:
1. **Cursor CLI** (`cursor-agent`) - Local/headless execution with full file system access
2. **Background Agents API** - Cloud-based execution with automatic PR creation

For swarm-orchestrator, the **CLI approach is recommended** as it aligns with our existing worktree-based architecture and consensus voting flow.

---

## Available Cursor Endpoints

### 1. Cursor CLI (`cursor-agent`)

The Cursor CLI brings AI coding assistance to the terminal and supports headless automation.

**Installation:**
```bash
curl https://cursor.com/install -fsS | bash
```

**Key Commands:**

| Command | Purpose |
|---------|---------|
| `cursor-agent "prompt"` | Interactive session with initial prompt |
| `cursor-agent -p "prompt"` | Non-interactive/headless mode |
| `cursor-agent ls` | List previous conversations |
| `cursor-agent resume` | Resume latest session |
| `cursor-agent --resume="id"` | Resume specific conversation |

**Flags for Automation:**

| Flag | Purpose |
|------|---------|
| `-p, --print` | Enable non-interactive scripting |
| `--force` | Allow file modifications without confirmation |
| `--output-format` | Response format: `text`, `json`, `stream-json` |
| `--stream-partial-output` | Enable incremental streaming |
| `--model` | Specify model (e.g., `gpt-5`, `claude-4-sonnet`) |

**Authentication:**
```bash
export CURSOR_API_KEY="sk_XXXX..."
# or inline:
cursor-agent --api-key sk_XXXX... -p "prompt"
```

**Source:** [Cursor CLI Docs](https://cursor.com/docs/cli/overview), [Headless Mode](https://cursor.com/docs/cli/headless)

---

### 2. Background Agents API

REST API for cloud-based agent execution. Agents run on Cursor's infrastructure and create PRs automatically.

**Base URL:** `https://api.cursor.com/v0`

**Authentication:** Basic Auth with API key (include trailing colon)
```bash
Authorization: Basic $(echo -n "YOUR_API_KEY:" | base64)
```

#### Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `POST` | `/v0/agents` | Launch a new agent |
| `GET` | `/v0/agents` | List agents (paginated) |
| `GET` | `/v0/agents/{id}` | Get agent status |
| `GET` | `/v0/agents/{id}/conversation` | Get message history |
| `POST` | `/v0/agents/{id}/followup` | Send follow-up instruction |
| `POST` | `/v0/agents/{id}/stop` | Pause agent (resumable) |
| `DELETE` | `/v0/agents/{id}` | Delete agent permanently |
| `GET` | `/v0/me` | API key info |
| `GET` | `/v0/models` | List available models |
| `GET` | `/v0/repositories` | List connected repos (rate-limited) |

#### Launch Agent Request

```json
POST /v0/agents
{
  "prompt": {
    "text": "Implement user authentication using JWT",
    "images": []  // Optional, max 5, base64 encoded
  },
  "source": {
    "repository": "owner/repo",
    "ref": "main"  // Optional branch/commit
  },
  "target": {
    "branchName": "feature/auth"  // Optional
  },
  "model": "auto",  // Or specific model
  "webhook": {
    "url": "https://your-server.com/webhook",
    "secret": "min-32-char-secret"  // Optional
  }
}
```

#### Agent Status Response

```json
{
  "id": "agent_abc123",
  "status": "running",  // pending, running, completed, failed, stopped
  "summary": "Implementing JWT authentication...",
  "repository": "owner/repo",
  "branch": "feature/auth",
  "pullRequest": {
    "url": "https://github.com/owner/repo/pull/42",
    "number": 42
  }
}
```

**Rate Limits:**
- Repository listing: 1/user/minute, 30/user/hour
- Other endpoints: Based on subscription tier

**Source:** [Background Agents API](https://cursor.com/docs/cloud-agent/api/overview)

---

### 3. GitHub Actions Integration

Cursor CLI can be used in CI/CD pipelines:

```yaml
- name: Install Cursor CLI
  run: |
    curl https://cursor.com/install -fsS | bash
    echo "$HOME/.cursor/bin" >> $GITHUB_PATH

- name: Run Cursor Agent
  env:
    CURSOR_API_KEY: ${{ secrets.CURSOR_API_KEY }}
  run: |
    cursor-agent -p "Your prompt here" --force
```

**Permission Configuration:**
```json
{
  "permissions": {
    "allow": ["Read(**/*.py)", "Write(src/**/*)", "Shell(pytest)"],
    "deny": ["Shell(git push)", "Write(.env*)"]
  }
}
```

**Source:** [GitHub Actions Docs](https://cursor.com/docs/cli/github-actions)

---

### 4. Pricing & Limits

| Plan | Price | Background Agents | Notes |
|------|-------|-------------------|-------|
| Hobby | Free | Limited | Entry-level limits |
| Pro | $20/mo | Included | $20 credit pool for non-Auto models |
| Pro Plus | ~$60/mo | $70 credit | Additional bonus usage |
| Ultra | $200/mo | ~20× Pro | Priority access |
| Teams | $40/user/mo | ~500 requests/user | SSO, admin controls |

Background Agents are charged at API pricing for the selected model.

**Source:** [Cursor Pricing](https://cursor.com/pricing), [Models & Pricing](https://cursor.com/docs/account/pricing)

---

## Integration Analysis

### Option A: CLI-Based Agent Backend (Recommended)

Use `cursor-agent` in headless mode within Schaltwerk worktrees.

**Architecture:**
```
┌─────────────────────────────────────────────────────────────┐
│                    Swarm Orchestrator                        │
├─────────────────────────────────────────────────────────────┤
│  SchaltwerkWorktreeBackend  │  CursorCLIAgentBackend        │
│  (creates worktrees)        │  (runs cursor-agent -p)       │
└─────────────────────────────────────────────────────────────┘
```

**Pros:**
- Works with existing worktree isolation
- Full control over execution environment
- Agents can use our swarm MCP for coordination
- No automatic PR creation (aligns with consensus flow)
- Local execution = faster iteration

**Cons:**
- Requires Cursor CLI installed on host
- Requires Cursor Pro subscription
- No built-in parallelization (we handle it)

**Implementation:**

```python
class CursorCLIAgentBackend(AgentBackend):
    """Execute agents using Cursor CLI in worktrees."""

    def __init__(self, model: str = "auto"):
        self.model = model
        self.processes: dict[str, subprocess.Popen] = {}

    def spawn_agent(self, session_name: str, prompt: str) -> str:
        worktree_path = self._get_worktree_path(session_name)

        # Write prompt to file for complex prompts
        prompt_file = Path(worktree_path) / ".swarm-prompt.md"
        prompt_file.write_text(prompt)

        proc = subprocess.Popen(
            [
                "cursor-agent",
                "-p", f"@{prompt_file}",  # Read from file
                "--force",
                "--model", self.model,
                "--output-format", "stream-json",
            ],
            cwd=worktree_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, "CURSOR_API_KEY": os.getenv("CURSOR_API_KEY")},
        )

        self.processes[session_name] = proc
        return session_name

    def wait_for_completion(
        self, agent_ids: list[str], timeout: int | None = None
    ) -> dict[str, AgentStatus]:
        results = {}
        for agent_id in agent_ids:
            proc = self.processes.get(agent_id)
            if proc:
                try:
                    stdout, stderr = proc.communicate(timeout=timeout)
                    results[agent_id] = AgentStatus(
                        agent_id=agent_id,
                        is_finished=True,
                        implementation=stdout.decode(),
                    )
                except subprocess.TimeoutExpired:
                    proc.kill()
                    results[agent_id] = AgentStatus(
                        agent_id=agent_id,
                        is_finished=False,
                    )
        return results

    def get_status(self, agent_id: str) -> AgentStatus:
        proc = self.processes.get(agent_id)
        if not proc:
            return AgentStatus(agent_id=agent_id, is_finished=True)

        return AgentStatus(
            agent_id=agent_id,
            is_finished=proc.poll() is not None,
        )

    def send_message(self, agent_id: str, message: str) -> None:
        # Cursor CLI doesn't support mid-session messages
        # Would need to use resume functionality
        raise NotImplementedError("Cursor CLI doesn't support mid-session messages")
```

---

### Option B: Cloud API Agent Backend

Use Background Agents API for cloud-based parallel execution.

**Architecture:**
```
┌─────────────────────────────────────────────────────────────┐
│                    Swarm Orchestrator                        │
├─────────────────────────────────────────────────────────────┤
│  CursorCloudWorktreeBackend │  CursorCloudAgentBackend      │
│  (API creates branches)     │  (API spawns agents)          │
└─────────────────────────────────────────────────────────────┘
        │                              │
        └──────────┬───────────────────┘
                   ▼
         Cursor Background Agents API
                   │
                   ▼
         GitHub (branches, PRs)
```

**Pros:**
- True parallel cloud execution
- No local resources required
- Automatic PR creation
- Works without local Cursor installation

**Cons:**
- Agents auto-create PRs (conflicts with consensus voting)
- GitHub-centric (requires repo connection)
- Less control over execution environment
- Can't use our swarm MCP for coordination
- Higher latency (cloud round-trips)

**Implementation:**

```python
import httpx

class CursorCloudAgentBackend(AgentBackend):
    """Execute agents using Cursor Background Agents API."""

    BASE_URL = "https://api.cursor.com/v0"

    def __init__(self, repository: str, model: str = "auto"):
        self.repository = repository
        self.model = model
        self.api_key = os.getenv("CURSOR_API_KEY")
        self._client = httpx.Client(
            base_url=self.BASE_URL,
            auth=(self.api_key, ""),  # Basic auth with trailing colon
        )
        self.agents: dict[str, str] = {}  # session_name -> agent_id

    def spawn_agent(self, session_name: str, prompt: str) -> str:
        response = self._client.post(
            "/agents",
            json={
                "prompt": {"text": prompt},
                "source": {"repository": self.repository},
                "target": {"branchName": f"swarm/{session_name}"},
                "model": self.model,
            },
        )
        response.raise_for_status()
        agent_id = response.json()["id"]
        self.agents[session_name] = agent_id
        return session_name

    def wait_for_completion(
        self, agent_ids: list[str], timeout: int | None = None
    ) -> dict[str, AgentStatus]:
        import time
        start = time.time()
        results = {}

        while True:
            all_done = True
            for session_name in agent_ids:
                if session_name in results and results[session_name].is_finished:
                    continue

                status = self.get_status(session_name)
                results[session_name] = status
                if not status.is_finished:
                    all_done = False

            if all_done:
                break

            if timeout and (time.time() - start) > timeout:
                break

            time.sleep(10)  # Poll every 10 seconds

        return results

    def get_status(self, agent_id: str) -> AgentStatus:
        cursor_agent_id = self.agents.get(agent_id)
        if not cursor_agent_id:
            return AgentStatus(agent_id=agent_id, is_finished=True)

        response = self._client.get(f"/agents/{cursor_agent_id}")
        data = response.json()

        finished_states = {"completed", "failed", "stopped"}
        return AgentStatus(
            agent_id=agent_id,
            is_finished=data["status"] in finished_states,
            implementation=data.get("summary"),
        )

    def send_message(self, agent_id: str, message: str) -> None:
        cursor_agent_id = self.agents.get(agent_id)
        if cursor_agent_id:
            self._client.post(
                f"/agents/{cursor_agent_id}/followup",
                json={"text": message},
            )
```

---

### Option C: Hybrid Approach

Use CLI for execution but API for enhanced monitoring.

**Use Cases:**
- CLI for agent execution in worktrees (full control)
- API webhooks for progress notifications
- API for model selection and validation

---

## Recommended Integration Path

### Phase 1: CLI Backend (MVP)

1. Implement `CursorCLIAgentBackend`
2. Test with existing `SchaltwerkWorktreeBackend`
3. Validate consensus flow works with Cursor agents

**Configuration:**
```json
{
  "worktree_backend": "schaltwerk",
  "agent_backend": "cursor-cli",
  "llm_backend": "claude-cli"
}
```

**CLI Usage:**
```bash
swarm run "implement feature X" --agent-backend cursor-cli
```

### Phase 2: Agent Prompt Adaptation

Modify agent instructions to work with Cursor's tool-calling model:
- Cursor uses different tool schemas than Claude Code
- May need to inject our swarm MCP differently
- Test coordination between Cursor agents

### Phase 3: Cloud Backend (Optional)

If local resources become a bottleneck:
1. Implement `CursorCloudAgentBackend`
2. Create `CursorCloudWorktreeBackend` (API-managed branches)
3. Adapt consensus flow to work with auto-generated PRs

---

## Challenges & Mitigations

| Challenge | Impact | Mitigation |
|-----------|--------|------------|
| Cursor CLI requires subscription | Barrier to entry | Document requirements clearly |
| Different tool schemas | Agent instructions may fail | Create Cursor-specific prompt templates |
| No mid-session messaging | Can't send follow-ups | Use resume functionality or restructure flow |
| Cloud agents auto-create PRs | Conflicts with voting flow | Use CLI backend or adapt PR-based voting |
| Rate limits on Pro tier | May throttle heavy usage | Implement backoff, recommend higher tiers |

---

## Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `CURSOR_API_KEY` | Yes | Authentication for CLI and API |

---

## Next Steps

1. [ ] Implement `CursorCLIAgentBackend` skeleton
2. [ ] Add `cursor-cli` to valid agent backends in config
3. [ ] Create Cursor-specific agent prompt template
4. [ ] Test single-agent execution in worktree
5. [ ] Test multi-agent consensus with Cursor agents
6. [ ] Document setup requirements in README

---

## References

- [Cursor CLI Documentation](https://cursor.com/docs/cli/overview)
- [Headless Mode](https://cursor.com/docs/cli/headless)
- [Background Agents API](https://cursor.com/docs/cloud-agent/api/overview)
- [GitHub Actions Integration](https://cursor.com/docs/cli/github-actions)
- [Cursor Pricing](https://cursor.com/pricing)
- [Community MCP Wrapper](https://github.com/mjdierkes/cursor-background-agent-api)
