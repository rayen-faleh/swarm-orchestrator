"""
MCP tools for the Swarm orchestrator.

These tools are exposed via the MCP server and can be called by agents
to coordinate their work.
"""

from abc import ABC, abstractmethod
from typing import Any

from .state import SwarmState


class BaseTool(ABC):
    """Base class for MCP tools."""

    name: str
    description: str
    input_schema: dict

    def __init__(self, state: SwarmState):
        self.state = state

    @abstractmethod
    def execute(self, arguments: dict) -> dict:
        """Execute the tool with the given arguments."""
        pass


class FinishedWorkTool(BaseTool):
    """
    Tool for agents to signal they have finished their work.

    When called, marks the agent as finished and stores their implementation.
    Returns the number of agents still working.
    """

    name = "finished_work"
    description = (
        "Signal that you have finished implementing the task. "
        "Call this after you have completed your implementation and committed your changes. "
        "You must provide your implementation (the git diff of your changes)."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "The ID of the task you are working on",
            },
            "agent_id": {
                "type": "string",
                "description": "Your agent ID (provided in your initial prompt)",
            },
            "implementation": {
                "type": "string",
                "description": "Your implementation as a git diff",
            },
        },
        "required": ["task_id", "agent_id", "implementation"],
    }

    def execute(self, arguments: dict) -> dict:
        task_id = arguments.get("task_id")
        agent_id = arguments.get("agent_id")
        implementation = arguments.get("implementation")

        # Validate required parameters
        missing = []
        if not task_id:
            missing.append("task_id")
        if not agent_id:
            missing.append("agent_id")
        if not implementation:
            missing.append("implementation")

        if missing:
            return {
                "success": False,
                "error": f"Required parameters missing: {', '.join(missing)}",
            }

        return self.state.mark_agent_finished(task_id, agent_id, implementation)


class GetAllImplementationsTool(BaseTool):
    """
    Tool to get all implementations from other agents.

    Only available after all agents have finished their work.
    Returns a list of implementations with agent IDs and session names.
    """

    name = "get_all_implementations"
    description = (
        "Get all implementations from all agents working on this task. "
        "This is only available after all agents have finished their work. "
        "Use this to review other implementations before casting your vote."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "The ID of the task",
            },
        },
        "required": ["task_id"],
    }

    def execute(self, arguments: dict) -> dict:
        task_id = arguments.get("task_id")

        if not task_id:
            return {"success": False, "error": "Required parameter missing: task_id"}

        return self.state.get_all_implementations(task_id)


class CastVoteTool(BaseTool):
    """
    Tool for agents to cast their vote for the best implementation.

    Agents can vote for any implementation, including their own.
    Must provide a reason for the vote.
    """

    name = "cast_vote"
    description = (
        "Cast your vote for the best implementation. "
        "You should review all implementations using get_all_implementations first. "
        "You can vote for any agent, including yourself if you believe your implementation is best. "
        "Provide a clear reason for your choice."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "The ID of the task",
            },
            "agent_id": {
                "type": "string",
                "description": "Your agent ID",
            },
            "voted_for": {
                "type": "string",
                "description": "The agent ID you are voting for",
            },
            "reason": {
                "type": "string",
                "description": "Your reason for voting for this implementation",
            },
        },
        "required": ["task_id", "agent_id", "voted_for", "reason"],
    }

    def execute(self, arguments: dict) -> dict:
        task_id = arguments.get("task_id")
        agent_id = arguments.get("agent_id")
        voted_for = arguments.get("voted_for")
        reason = arguments.get("reason", "")

        # Validate required parameters
        missing = []
        if not task_id:
            missing.append("task_id")
        if not agent_id:
            missing.append("agent_id")
        if not voted_for:
            missing.append("voted_for")

        if missing:
            return {
                "success": False,
                "error": f"Required parameters missing: {', '.join(missing)}",
            }

        return self.state.cast_vote(task_id, agent_id, voted_for, reason)


class GetVoteResultsTool(BaseTool):
    """
    Tool to get the current vote results.

    Returns vote counts and the winner (if all agents have voted).
    """

    name = "get_vote_results"
    description = (
        "Get the current vote results for the task. "
        "Shows how many votes each implementation has received and the winner if all votes are in."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "The ID of the task",
            },
        },
        "required": ["task_id"],
    }

    def execute(self, arguments: dict) -> dict:
        task_id = arguments.get("task_id")

        if not task_id:
            return {"success": False, "error": "Required parameter missing: task_id"}

        return self.state.get_vote_results(task_id)


# Registry of all available tools
TOOL_REGISTRY = {
    "finished_work": FinishedWorkTool,
    "get_all_implementations": GetAllImplementationsTool,
    "cast_vote": CastVoteTool,
    "get_vote_results": GetVoteResultsTool,
}
