"""
Backend interfaces for swarm-orchestrator.

This package defines abstract base classes for pluggable backends:
- WorktreeBackend: Git worktree/isolation management
- AgentBackend: Agent execution and lifecycle
- LLMBackend: LLM calls for decomposition and exploration

Concrete implementations:
- SchaltwerkWorktreeBackend: Schaltwerk MCP-based worktree management
- SchaltwerkAgentBackend: Schaltwerk MCP-based agent spawning
"""

from .base import (
    WorktreeBackend,
    AgentBackend,
    LLMBackend,
    SessionInfo,
    DiffResult,
    AgentStatus,
    DecomposeResult,
)
from .schaltwerk import (
    SchaltwerkWorktreeBackend,
    SchaltwerkAgentBackend,
)
from .git_native import GitNativeWorktreeBackend, GitNativeAgentBackend
from .cursor import CursorCLIAgentBackend
from .llm import (
    ClaudeCLIBackend,
    AnthropicAPIBackend,
    LLMBackendError,
)

__all__ = [
    # Abstract interfaces
    "WorktreeBackend",
    "AgentBackend",
    "LLMBackend",
    # Data models
    "SessionInfo",
    "DiffResult",
    "AgentStatus",
    "DecomposeResult",
    # Schaltwerk implementations
    "SchaltwerkWorktreeBackend",
    "SchaltwerkAgentBackend",
    # Git-native implementations
    "GitNativeWorktreeBackend",
    "GitNativeAgentBackend",
    # Cursor CLI implementation
    "CursorCLIAgentBackend",
    # LLM implementations
    "ClaudeCLIBackend",
    "AnthropicAPIBackend",
    "LLMBackendError",
]
