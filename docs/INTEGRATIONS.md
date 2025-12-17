# Backend Integrations Guide

This document covers the swarm-orchestrator's pluggable backend architecture and proposes new integrations.

## Architecture Overview

The orchestrator uses three pluggable backend interfaces:

```
┌─────────────────────────────────────────────────────────────┐
│                      Orchestrator                           │
├─────────────────────────────────────────────────────────────┤
│  WorktreeBackend  │  AgentBackend  │  LLMBackend           │
│  (isolation)      │  (execution)   │  (decomposition)      │
└─────────────────────────────────────────────────────────────┘
```

**WorktreeBackend**: Creates isolated environments for agents (git worktrees, containers, etc.)

**AgentBackend**: Spawns and manages coding agents (Claude Code, Aider, OpenCode, etc.)

**LLMBackend**: Handles task decomposition and exploration (Claude, GPT-4, Gemini, etc.)

## Current Implementations

| Backend | Implementation | Description |
|---------|----------------|-------------|
| Worktree | `SchaltwerkWorktreeBackend` | Uses Schaltwerk MCP for git worktrees (macOS) |
| Agent | `SchaltwerkAgentBackend` | Spawns Claude Code via Schaltwerk |
| LLM | `ClaudeCLIBackend` | Uses `claude` CLI (Max/Pro subscription) |
| LLM | `AnthropicAPIBackend` | Uses Anthropic API directly |

## Implementing Custom Backends

### WorktreeBackend Interface

```python
from swarm_orchestrator.backends import WorktreeBackend, SessionInfo, DiffResult

class MyWorktreeBackend(WorktreeBackend):
    def create_session(self, name: str, content: str) -> SessionInfo:
        """Create isolated environment with spec content."""
        ...

    def delete_session(self, name: str, force: bool = False) -> None:
        """Clean up session resources."""
        ...

    def get_session(self, name: str) -> SessionInfo | None:
        """Get session info by name."""
        ...

    def list_sessions(self, filter_type: str = "all") -> list[SessionInfo]:
        """List sessions with optional filtering."""
        ...

    def get_diff(self, session_name: str) -> DiffResult:
        """Get changes made in session."""
        ...

    def merge_session(self, name: str, commit_message: str) -> None:
        """Merge session changes back to parent branch."""
        ...
```

### AgentBackend Interface

```python
from swarm_orchestrator.backends import AgentBackend, AgentStatus

class MyAgentBackend(AgentBackend):
    def spawn_agent(self, session_name: str, prompt: str) -> str:
        """Start agent in session, return agent ID."""
        ...

    def wait_for_completion(
        self, agent_ids: list[str], timeout: int | None = None
    ) -> dict[str, AgentStatus]:
        """Wait for agents to finish work."""
        ...

    def send_message(self, agent_id: str, message: str) -> None:
        """Send follow-up message to running agent."""
        ...

    def get_status(self, agent_id: str) -> AgentStatus:
        """Check agent's current status."""
        ...
```

### LLMBackend Interface

```python
from swarm_orchestrator.backends import LLMBackend, DecomposeResult

class MyLLMBackend(LLMBackend):
    def decompose(self, query: str, context: str | None = None) -> DecomposeResult:
        """Break task into subtasks."""
        ...

    def explore(self, query: str) -> str:
        """Explore codebase for context."""
        ...
```

## Proposed Integrations

### 1. Git Worktree Backend (Native)

**Purpose**: Cross-platform worktree management without Schaltwerk dependency.

**Implementation**:
```python
class GitWorktreeBackend(WorktreeBackend):
    def __init__(self, base_path: str = ".swarm/worktrees"):
        self.base_path = Path(base_path)

    def create_session(self, name: str, content: str) -> SessionInfo:
        branch = f"swarm/{name}"
        worktree_path = self.base_path / name
        subprocess.run(["git", "worktree", "add", "-b", branch, str(worktree_path)])
        (worktree_path / ".swarm-spec.md").write_text(content)
        return SessionInfo(name=name, status="active", branch=branch,
                          worktree_path=str(worktree_path))
```

**Configuration**:
```json
{
  "worktree_backend": "git-native",
  "worktree_base_path": ".swarm/worktrees"
}
```

### 2. Docker Container Backend

**Purpose**: Full isolation via containers, supports any OS and prevents file system conflicts.

**Implementation**:
```python
class DockerWorktreeBackend(WorktreeBackend):
    def __init__(self, image: str = "swarm-agent:latest"):
        self.image = image
        self.containers: dict[str, str] = {}

    def create_session(self, name: str, content: str) -> SessionInfo:
        container_id = subprocess.check_output([
            "docker", "run", "-d", "--name", f"swarm-{name}",
            "-v", f"{os.getcwd()}:/workspace:ro",
            self.image, "sleep", "infinity"
        ]).decode().strip()
        self.containers[name] = container_id
        return SessionInfo(name=name, status="running", branch=name,
                          metadata={"container_id": container_id})
```

**Configuration**:
```json
{
  "worktree_backend": "docker",
  "docker_image": "swarm-agent:latest"
}
```

### 3. Aider Agent Backend

**Purpose**: Use [Aider](https://aider.chat) as the coding agent.

**Implementation**:
```python
class AiderAgentBackend(AgentBackend):
    def __init__(self, model: str = "claude-3-5-sonnet"):
        self.model = model
        self.processes: dict[str, subprocess.Popen] = {}

    def spawn_agent(self, session_name: str, prompt: str) -> str:
        worktree_path = self._get_worktree_path(session_name)
        proc = subprocess.Popen(
            ["aider", "--model", self.model, "--message", prompt, "--yes"],
            cwd=worktree_path
        )
        self.processes[session_name] = proc
        return session_name
```

**Configuration**:
```json
{
  "agent_backend": "aider",
  "aider_model": "claude-3-5-sonnet"
}
```

### 4. OpenCode Agent Backend

**Purpose**: Use [OpenCode](https://github.com/opencode-ai/opencode) agents.

**Implementation**:
```python
class OpenCodeAgentBackend(AgentBackend):
    def spawn_agent(self, session_name: str, prompt: str) -> str:
        worktree_path = self._get_worktree_path(session_name)
        proc = subprocess.Popen(
            ["opencode", "--non-interactive", "--prompt", prompt],
            cwd=worktree_path
        )
        self.processes[session_name] = proc
        return session_name
```

**Configuration**:
```json
{
  "agent_backend": "opencode"
}
```

### 5. Cursor Agent Backend

**Purpose**: Integrate with Cursor's agent mode via CLI.

**Implementation**:
```python
class CursorAgentBackend(AgentBackend):
    def spawn_agent(self, session_name: str, prompt: str) -> str:
        worktree_path = self._get_worktree_path(session_name)
        # Write prompt to file for Cursor to pick up
        prompt_file = Path(worktree_path) / ".cursor-prompt.md"
        prompt_file.write_text(prompt)
        subprocess.Popen(["cursor", "--folder", worktree_path])
        return session_name
```

### 6. OpenAI LLM Backend

**Purpose**: Use GPT-4 for task decomposition.

**Implementation**:
```python
from openai import OpenAI

class OpenAIBackend(LLMBackend):
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.client = OpenAI()

    def decompose(self, query: str, context: str | None = None) -> DecomposeResult:
        prompt = f"{query}\n\nContext:\n{context}" if context else query
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": DECOMPOSE_PROMPT.format(query=prompt)}]
        )
        return self._parse_response(response.choices[0].message.content)

    def explore(self, query: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": query}]
        )
        return response.choices[0].message.content
```

**Configuration**:
```json
{
  "llm_backend": "openai",
  "llm_model": "gpt-4o"
}
```

### 7. Google Gemini LLM Backend

**Purpose**: Use Gemini models for decomposition.

**Implementation**:
```python
import google.generativeai as genai

class GeminiBackend(LLMBackend):
    def __init__(self, model: str = "gemini-1.5-pro"):
        self.model = genai.GenerativeModel(model)

    def decompose(self, query: str, context: str | None = None) -> DecomposeResult:
        prompt = f"{query}\n\nContext:\n{context}" if context else query
        response = self.model.generate_content(DECOMPOSE_PROMPT.format(query=prompt))
        return self._parse_response(response.text)

    def explore(self, query: str) -> str:
        response = self.model.generate_content(query)
        return response.text
```

**Configuration**:
```json
{
  "llm_backend": "gemini",
  "llm_model": "gemini-1.5-pro"
}
```

## Configuration Examples

### Full Configuration File

Create `.swarm/config.json`:

```json
{
  "worktree_backend": "schaltwerk",
  "agent_backend": "schaltwerk",
  "llm_backend": "claude-cli",
  "llm_model": "claude-sonnet-4-20250514",
  "llm_timeout": 120
}
```

### Cross-Platform Setup (No Schaltwerk)

```json
{
  "worktree_backend": "git-native",
  "agent_backend": "aider",
  "llm_backend": "anthropic-api",
  "llm_model": "claude-sonnet-4-20250514"
}
```

### Docker-Based Isolation

```json
{
  "worktree_backend": "docker",
  "docker_image": "python:3.12-slim",
  "agent_backend": "opencode",
  "llm_backend": "openai",
  "llm_model": "gpt-4o"
}
```

## Registering Custom Backends

To add a new backend, update `config.py` validation and `orchestrator.py` factory methods:

```python
# config.py - Add to validation sets
valid_worktree = {"schaltwerk", "git-native", "docker"}
valid_agent = {"schaltwerk", "aider", "opencode", "cursor"}
valid_llm = {"claude-cli", "anthropic-api", "openai", "gemini"}

# orchestrator.py - Add to factory methods
def _create_worktree_backend(self) -> WorktreeBackend:
    if self.config.worktree_backend == "git-native":
        return GitWorktreeBackend()
    elif self.config.worktree_backend == "docker":
        return DockerWorktreeBackend(self.config.docker_image)
    # ... existing schaltwerk case
```
