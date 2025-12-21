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
        "git-native": "Uses native git worktree commands without MCP dependency",
    },
    "agent": {
        "schaltwerk": "Uses Schaltwerk MCP to spawn Claude agents (default)",
        "cursor-cli": "Uses Cursor CLI for agent execution - requires cursor-agent installed and authentication via 'swarm cursor login' or CURSOR_API_KEY env var",
        "git-native": "Uses native git worktrees with Claude CLI for agent execution without MCP dependency",
    },
    "cli_tool": {
        "claude": "Uses Claude Code CLI - default for both decomposition and agent execution",
        "opencode": "Uses OpenCode CLI for agent execution (decomposition uses Claude CLI)",
        "cursor": "Uses Cursor CLI for agent execution (decomposition uses Claude CLI)",
        "anthropic-api": "Uses Anthropic API directly for decomposition - requires ANTHROPIC_API_KEY env var",
    },
}

# Mapping from old llm_backend values to new cli_tool values for backwards compatibility
_LLM_BACKEND_MIGRATION = {
    "claude-cli": "claude",
    "anthropic-api": "anthropic-api",
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
    agent execution, and CLI tool for both decomposition and agent execution.

    Attributes:
        worktree_backend: Backend for worktree/isolation ("schaltwerk" or "git-native")
        agent_backend: Backend for agent execution ("schaltwerk", "cursor-cli", or "git-native")
        cli_tool: Unified CLI tool for decomposition and agent execution.
                  Options: "claude", "opencode", "cursor", "anthropic-api"
        llm_model: Model to use for API backend (default: claude-sonnet-4-20250514)
        llm_timeout: Timeout for LLM calls in seconds (default: 120)
        exploration_model: Model for exploration phase (default: claude-haiku-3-5)
        enable_diff_compression: Whether to compress diffs (default: True)
        compression_min_tokens: Minimum tokens before compression kicks in (default: 500)
        compression_target_ratio: Target compression ratio, 0.1-1.0 (default: 0.3)
    """

    worktree_backend: str = "schaltwerk"
    agent_backend: str = "schaltwerk"
    cli_tool: str = "claude"
    llm_model: str = "claude-sonnet-4-20250514"
    llm_timeout: int = 120
    exploration_model: str = "claude-haiku-3-5"
    enable_diff_compression: bool = True
    compression_min_tokens: int = 500
    compression_target_ratio: float = 0.3

    def __post_init__(self):
        """Validate configuration values."""
        valid_worktree = set(get_backend_choices("worktree"))
        valid_agent = set(get_backend_choices("agent"))
        valid_cli_tool = set(get_backend_choices("cli_tool"))

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
        if self.cli_tool not in valid_cli_tool:
            raise ValueError(
                f"Invalid cli_tool: {self.cli_tool}. "
                f"Valid options: {valid_cli_tool}"
            )
        if not (0.1 <= self.compression_target_ratio <= 1.0):
            raise ValueError(
                f"Invalid compression_target_ratio: {self.compression_target_ratio}. "
                f"Must be between 0.1 and 1.0"
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SwarmConfig":
        """Create SwarmConfig from a dictionary.

        Includes backwards-compatible migration from old llm_backend field.
        If both llm_backend and cli_tool are present, cli_tool takes precedence.
        """
        # Migrate from old llm_backend field if present
        cli_tool = data.get("cli_tool")
        if cli_tool is None and "llm_backend" in data:
            # Migrate old llm_backend value to new cli_tool
            old_llm_backend = data["llm_backend"]
            cli_tool = _LLM_BACKEND_MIGRATION.get(old_llm_backend, "claude")
        elif cli_tool is None:
            cli_tool = "claude"

        return cls(
            worktree_backend=data.get("worktree_backend", "schaltwerk"),
            agent_backend=data.get("agent_backend", "schaltwerk"),
            cli_tool=cli_tool,
            llm_model=data.get("llm_model", "claude-sonnet-4-20250514"),
            llm_timeout=data.get("llm_timeout", 120),
            exploration_model=data.get("exploration_model", "claude-haiku-3-5"),
            enable_diff_compression=data.get("enable_diff_compression", True),
            compression_min_tokens=data.get("compression_min_tokens", 500),
            compression_target_ratio=data.get("compression_target_ratio", 0.3),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "worktree_backend": self.worktree_backend,
            "agent_backend": self.agent_backend,
            "cli_tool": self.cli_tool,
            "llm_model": self.llm_model,
            "llm_timeout": self.llm_timeout,
            "exploration_model": self.exploration_model,
            "enable_diff_compression": self.enable_diff_compression,
            "compression_min_tokens": self.compression_min_tokens,
            "compression_target_ratio": self.compression_target_ratio,
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
