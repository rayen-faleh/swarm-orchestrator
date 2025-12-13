# Swarm Orchestrator

Multi-agent consensus system using Schaltwerk + Claude Code.

## Overview

Applies MAKER paper principles (redundant execution + voting) using existing Claude Code instances via Schaltwerk, rather than building custom agents from scratch.

## Installation

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Usage

```bash
# Run a task through multi-agent consensus
swarm run "Add user authentication with JWT"

# Decompose a task (dry run)
swarm decompose "Build a REST API"

# Check status
swarm status
```

## How It Works

1. **Decompose** - Task is analyzed and broken into subtasks (if complex)
2. **Spawn** - N Claude Code agents work on each subtask in parallel worktrees
3. **Wait** - System waits for all agents to complete
4. **Vote** - Outputs are compared via git diff
5. **Merge** - Majority solution is merged to main branch

## Development

```bash
# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov

# Lint
uv run ruff check src/
```

## Architecture

```
User Query
    │
    ▼
┌──────────────┐
│ Decomposer   │ ← Claude API
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Schaltwerk   │ ← Spawns Claude Code agents
│ (N agents)   │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Voting       │ ← Compare diffs, find majority
└──────┬───────┘
       │
       ▼
 Consensus Output
```

## License

MIT
