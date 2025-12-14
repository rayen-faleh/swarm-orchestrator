# Swarm Orchestrator

Multi-agent consensus system using Schaltwerk + Claude Code.

## Overview

Swarm Orchestrator applies MAKER paper principles (redundant execution + voting) using existing Claude Code instances via Schaltwerk. Multiple AI agents work on the same task in parallel, then vote on the best solution - the majority winner gets merged.

## Prerequisites

- Python 3.11+
- [Claude Code](https://claude.ai/claude-code) installed and configured
- [Schaltwerk](https://github.com/anthropics/schaltwerk) MCP server available
- Git repository (swarm operates on git repos)

## Installation

Install swarm-orchestrator into your project:

```bash
# Using uv (recommended)
uv pip install swarm-orchestrator

# Or using pip
pip install swarm-orchestrator
```

For development:

```bash
# Clone the repository
git clone https://github.com/anthropics/swarm-orchestrator.git
cd swarm-orchestrator

# Install with dev dependencies
uv pip install -e ".[dev]"
```

## Setup

Initialize swarm-orchestrator in your project:

```bash
cd /path/to/your/project
swarm init
```

This creates:
- `.mcp.json` - MCP server configuration for Claude Code
- `.swarm/` - Directory for task state persistence

After initialization, **restart Claude Code** to load the new MCP server.

## Usage

### Run a Task

Execute a task through the multi-agent consensus system:

```bash
# Basic usage (3 agents by default)
swarm run "Add user authentication with JWT"

# Specify number of agents
swarm run "Refactor the database layer" --agents 5

# Auto-merge the winning solution
swarm run "Fix the login bug" --auto-merge
```

### Decompose a Task (Dry Run)

Preview how a task would be broken down without spawning agents:

```bash
swarm decompose "Build a REST API with CRUD operations"
```

### Check Status

View the status of running swarm sessions:

```bash
swarm status
```

### CLI Reference

```
swarm --help                    Show all commands
swarm --version                 Show version

swarm run <query>               Run task through consensus
  --agents, -a <n>              Number of agents (default: 3)
  --timeout, -t <secs>          Decomposition timeout (default: 120)
  --auto-merge                  Auto-merge winning solution

swarm decompose <query>         Preview task decomposition
swarm status                    Show session status
swarm init                      Initialize in current project
  --force, -f                   Overwrite existing config
```

## How It Works

```
User Query
    │
    ▼
┌──────────────┐
│  Decompose   │  Break complex tasks into subtasks
└──────┬───────┘
       │
       ▼
┌──────────────┐
│    Spawn     │  N Claude Code agents per subtask
│  (parallel)  │  Each in isolated git worktree
└──────┬───────┘
       │
       ▼
┌──────────────┐
│    Wait      │  Agents work concurrently
└──────┬───────┘
       │
       ▼
┌──────────────┐
│    Vote      │  Agents review all implementations
│              │  and vote for the best one
└──────┬───────┘
       │
       ▼
┌──────────────┐
│    Merge     │  Majority solution wins
└──────────────┘
```

### Workflow Details

1. **Decompose** - Complex tasks are analyzed and broken into independent subtasks using Claude API
2. **Spawn** - For each subtask, N agents are spawned via Schaltwerk in separate git worktrees
3. **Work** - Agents implement solutions concurrently, isolated from each other
4. **Signal** - Each agent signals completion via MCP tools with their git diff
5. **Vote** - Once all agents finish, each reviews all implementations and votes for the best
6. **Consensus** - The implementation with the most votes wins
7. **Merge** - The winning solution is merged back to the main branch

## Configuration

### Project Structure

After `swarm init`, your project will have:

```
your-project/
├── .mcp.json           # MCP server config (add to .gitignore)
├── .swarm/
│   ├── state.json      # Task state (gitignored)
│   └── .gitignore
└── ...
```

### Multi-Project Support

Each project has isolated configuration. The state file uses absolute paths, so:
- Different projects don't interfere with each other
- Agents in worktrees share the same state file within a project

### Environment Variables

- `ANTHROPIC_API_KEY` - Required for task decomposition via Claude API

## Development

```bash
# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov

# Lint
uv run ruff check src/

# Format
uv run ruff format src/
```

## License

MIT
