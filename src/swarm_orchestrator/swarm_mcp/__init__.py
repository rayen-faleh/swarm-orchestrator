"""
Swarm MCP Server - Agent coordination via MCP protocol.

Provides tools for:
- Signaling work completion (finished_work)
- Getting all implementations (get_all_implementations)
- Casting votes (cast_vote)
- Getting vote results (get_vote_results)
"""

from .state import SwarmState, TaskState, AgentStatus, VoteRecord
from .server import SwarmMCPServer
from .tools import (
    FinishedWorkTool,
    GetAllImplementationsTool,
    CastVoteTool,
    GetVoteResultsTool,
)

__all__ = [
    "SwarmState",
    "TaskState",
    "AgentStatus",
    "VoteRecord",
    "SwarmMCPServer",
    "FinishedWorkTool",
    "GetAllImplementationsTool",
    "CastVoteTool",
    "GetVoteResultsTool",
]
