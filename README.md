# Swarm Orchestrator

Multi-agent consensus system using Schaltwerk + Claude Code.

## Overview

Swarm Orchestrator applies MAKER paper principles (redundant execution + voting) using existing Claude Code instances via Schaltwerk. Multiple AI agents work on the same task in parallel, then vote on the best solution - the majority winner gets merged.

## Prerequisites

Before installing Swarm Orchestrator, ensure you have the following:

### Required

| Requirement | Version | Description |
|-------------|---------|-------------|
| **Python** | 3.11+ | Python 3.11, 3.12, or 3.13 supported |
| **Git** | Any recent | Swarm operates on git repositories |
| **Anthropic API Key** | - | Set as `ANTHROPIC_API_KEY` environment variable |

### External Dependencies

| Dependency | Purpose | Platform |
|------------|---------|----------|
| **[Claude Code](https://claude.ai/claude-code)** | AI agents run as Claude Code instances | All |
| **[Schaltwerk](https://github.com/2mawi2/schaltwerk)** | Manages git worktrees for parallel agents | macOS only |

#### Installing Claude Code

```bash
npm install -g @anthropic-ai/claude-code
```

#### Installing Schaltwerk

[Schaltwerk](https://github.com/2mawi2/schaltwerk) is a desktop app that manages Claude Code agents in isolated git worktrees. Currently **macOS only**.

```bash
# Install via Homebrew
brew install --cask 2mawi2/tap/schaltwerk
```

> **Note**: Schaltwerk must be running and configured as an MCP server in your Claude Code setup. Swarm Orchestrator uses Schaltwerk to create isolated git worktrees for each agent, enabling true parallel execution without conflicts.

## Installation

### Quick Install (Recommended)

For most users, install directly from PyPI:

```bash
# Using uv (fastest)
uv pip install swarm-orchestrator

# Using pip
pip install swarm-orchestrator

# Using pipx (isolated environment)
pipx install swarm-orchestrator
```

### Verify Installation

```bash
swarm --version
swarm --help
```

### Installation Methods Comparison

| Method | Best For | Isolation | Command |
|--------|----------|-----------|---------|
| `uv pip install` | Speed, daily use | Project venv | `uv pip install swarm-orchestrator` |
| `pip install` | Standard Python | Project venv | `pip install swarm-orchestrator` |
| `pipx install` | CLI tool users | Global isolated | `pipx install swarm-orchestrator` |

### Development Installation

For contributors or those who want to modify the source:

```bash
# 1. Clone the repository
git clone https://github.com/rayen-faleh/swarm-orchestrator.git
cd swarm-orchestrator

# 2. Create virtual environment (recommended)
uv venv
source .venv/bin/activate  # Linux/macOS
# or: .venv\Scripts\activate  # Windows

# 3. Install with dev dependencies
uv pip install -e ".[dev]"

# 4. Verify installation
swarm --version
uv run pytest  # Run tests
```

### Troubleshooting Installation

**"Command not found: swarm"**
- Ensure your virtual environment is activated
- Or use `python -m swarm_orchestrator.cli` instead

**"No module named 'anthropic'"**
- Dependencies didn't install correctly. Run: `uv pip install swarm-orchestrator --reinstall`

**Python version errors**
- Swarm requires Python 3.11+. Check with: `python --version`

## Quick Start

Get up and running in 4 steps:

### Step 1: Set Your API Key

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

Add to your shell profile (`~/.bashrc`, `~/.zshrc`) for persistence.

### Step 2: Initialize in Your Project

```bash
cd /path/to/your/project
swarm init
```

This creates:
- `.mcp.json` - MCP server configuration for Claude Code
- `.swarm/` - Directory for task state persistence

### Step 3: Restart Claude Code

After initialization, **restart Claude Code** to load the new MCP server configuration.

### Step 4: Run Your First Task

```bash
swarm run "Add a hello world function to main.py"
```

Watch as multiple agents work on your task, vote on solutions, and produce a consensus result!

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
