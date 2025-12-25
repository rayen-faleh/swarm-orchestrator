"""
Swarm Orchestrator - Multi-agent consensus using Schaltwerk + Claude Code

Applies MAKER paper principles (redundant execution + voting) using existing
Claude Code instances via Schaltwerk.

Research-backed task decomposition based on:
- MAKER (arXiv:2511.09030): Maximal Agentic Decomposition
- Agentless (arXiv:2407.01489): Hierarchical localization
- Select-Then-Decompose (arXiv:2510.17922): Adaptive strategies
- SWE-bench: Real-world software patch statistics
"""

__version__ = "0.2.0"

from .decomposer import (
    Subtask,
    SubtaskScope,
    DecompositionResult,
    Decomposer,
    decompose_task,
    validate_decomposition,
    SCOPE_LIMITS,
)
from .exploration import (
    ExplorationExecutor,
    needs_exploration,
)
from .orchestrator import (
    Orchestrator,
    OrchestrationResult,
    SubtaskResult,
    run_swarm,
)

__all__ = [
    # Version
    "__version__",
    # Decomposer
    "Subtask",
    "SubtaskScope",
    "DecompositionResult",
    "Decomposer",
    "decompose_task",
    "validate_decomposition",
    "SCOPE_LIMITS",
    # Exploration
    "ExplorationExecutor",
    "needs_exploration",
    # Orchestrator
    "Orchestrator",
    "OrchestrationResult",
    "SubtaskResult",
    "run_swarm",
]
