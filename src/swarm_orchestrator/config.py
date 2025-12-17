"""
Configuration system for swarm-orchestrator.

Provides SwarmConfig dataclass for backend selection and load_config function
for loading configuration from .swarm/config.json.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# Backend registry: centralized definitions with descriptions for help text
BACKENDS = {
    "worktree": {
        "schaltwerk": "Uses Schaltwerk MCP for git worktree isolation (default)",
    },
    "agent": {
        "schaltwerk": "Uses Schaltwerk MCP to spawn Claude agents (default)",
        "cursor-cli": "Uses Cursor CLI for agent execution - requires cursor-agent installed and CURSOR_API_KEY",
    },
    "llm": {
        "claude-cli": "Uses 'claude' CLI tool - requires Claude Code installed (default)",
        "anthropic-api": "Uses Anthropic API directly - requires ANTHROPIC_API_KEY env var",
    },
}


def get_backend_choices(backend_type: str) -> list[str]:
    """Get list of valid choices for a backend type."""
    return list(BACKENDS.get(backend_type, {}).keys())


def format_backend_help(backend_type: str, intro: str = "") -> str:
    """Format help text for a backend type with all options described."""
    options = BACKENDS.get(backend_type, {})
    if not options:
        return intro
    lines = [intro] if intro else []
    for name, desc in options.items():
        lines.append(f"  {name}: {desc}")
    return "\n".join(lines)


@dataclass
class SwarmConfig:
    """
    Configuration for swarm-orchestrator backends.

    Allows users to select which backends to use for worktree management,
    agent execution, and LLM calls.

    Attributes:
        worktree_backend: Backend for worktree/isolation ("schaltwerk")
        agent_backend: Backend for agent execution ("schaltwerk")
        llm_backend: Backend for LLM calls ("claude-cli" or "anthropic-api")
        llm_model: Model to use for API backend (default: claude-sonnet-4-20250514)
        llm_timeout: Timeout for LLM calls in seconds (default: 120)
    """

    worktree_backend: str = "schaltwerk"
    agent_backend: str = "schaltwerk"
    llm_backend: str = "claude-cli"
    llm_model: str = "claude-sonnet-4-20250514"
    llm_timeout: int = 120

    def __post_init__(self):
        """Validate configuration values."""
        valid_worktree = set(get_backend_choices("worktree"))
        valid_agent = set(get_backend_choices("agent"))
        valid_llm = set(get_backend_choices("llm"))

        if self.worktree_backend not in valid_worktree:
            raise ValueError(
                f"Invalid worktree_backend: {self.worktree_backend}. "
                f"Valid options: {valid_worktree}"
            )
        if self.agent_backend not in valid_agent:
            raise ValueError(
                f"Invalid agent_backend: {self.agent_backend}. "
                f"Valid options: {valid_agent}"
            )
        if self.llm_backend not in valid_llm:
            raise ValueError(
                f"Invalid llm_backend: {self.llm_backend}. "
                f"Valid options: {valid_llm}"
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SwarmConfig":
        """Create SwarmConfig from a dictionary."""
        return cls(
            worktree_backend=data.get("worktree_backend", "schaltwerk"),
            agent_backend=data.get("agent_backend", "schaltwerk"),
            llm_backend=data.get("llm_backend", "claude-cli"),
            llm_model=data.get("llm_model", "claude-sonnet-4-20250514"),
            llm_timeout=data.get("llm_timeout", 120),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "worktree_backend": self.worktree_backend,
            "agent_backend": self.agent_backend,
            "llm_backend": self.llm_backend,
            "llm_model": self.llm_model,
            "llm_timeout": self.llm_timeout,
        }


def load_config(config_path: str | Path | None = None) -> SwarmConfig:
    """
    Load configuration from file or return defaults.

    Args:
        config_path: Path to config file. If None, looks for .swarm/config.json

    Returns:
        SwarmConfig with loaded or default values
    """
    if config_path is None:
        config_path = Path(".swarm/config.json")
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        return SwarmConfig()

    try:
        data = json.loads(config_path.read_text())
        return SwarmConfig.from_dict(data)
    except (json.JSONDecodeError, KeyError) as e:
        raise ValueError(f"Invalid config file {config_path}: {e}")


def save_config(config: SwarmConfig, config_path: str | Path | None = None) -> None:
    """
    Save configuration to file.

    Args:
        config: SwarmConfig to save
        config_path: Path to config file. If None, saves to .swarm/config.json
    """
    if config_path is None:
        config_path = Path(".swarm/config.json")
    else:
        config_path = Path(config_path)

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config.to_dict(), indent=2))
