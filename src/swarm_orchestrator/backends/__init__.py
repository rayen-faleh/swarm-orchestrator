"""
Backend interfaces for swarm-orchestrator.

This package defines abstract base classes for pluggable backends:
- WorktreeBackend: Git worktree/isolation management
- AgentBackend: Agent execution and lifecycle
- LLMBackend: LLM calls for decomposition and exploration
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

__all__ = [
    "WorktreeBackend",
    "AgentBackend",
    "LLMBackend",
    "SessionInfo",
    "DiffResult",
    "AgentStatus",
    "DecomposeResult",
]
