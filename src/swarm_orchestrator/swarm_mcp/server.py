"""
Swarm MCP Server implementation.

Provides a JSON-RPC 2.0 server for agent coordination.
Agents connect to this server to signal completion, get implementations, and vote.
"""

import json
import sys
from typing import Any, Optional

from .state import SwarmState, TaskState
from .tools import (
    TOOL_REGISTRY,
    FinishedWorkTool,
    GetAllImplementationsTool,
    CastVoteTool,
    GetVoteResultsTool,
)
from ..config import SwarmConfig, load_config


class SwarmMCPServer:
    """
    MCP server for swarm agent coordination.

    Exposes tools via JSON-RPC 2.0 protocol:
    - finished_work: Signal work completion
    - get_all_implementations: Get all agent implementations
    - cast_vote: Vote for best implementation
    - get_vote_results: Get voting results
    """

    def __init__(
        self,
        persistence_path: Optional[str] = None,
        compression_config: Optional[SwarmConfig] = None,
    ):
        """
        Initialize the MCP server.

        Args:
            persistence_path: Optional path to persist state across restarts.
            compression_config: Optional SwarmConfig with compression settings.
        """
        self.state = SwarmState(
            persistence_path=persistence_path,
            compression_config=compression_config,
        )
        self._tools = self._create_tools()

    def _create_tools(self) -> dict:
        """Create tool instances."""
        return {
            name: cls(self.state) for name, cls in TOOL_REGISTRY.items()
        }

    def list_tools(self) -> list[dict]:
        """
        List all available tools.

        Returns list of tool definitions for MCP protocol.
        """
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.input_schema,
            }
            for tool in self._tools.values()
        ]

    def call_tool(self, name: str, arguments: dict) -> dict:
        """
        Call a tool by name.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        if name not in self._tools:
            return {"success": False, "error": f"Unknown tool: {name}"}

        return self._tools[name].execute(arguments)

    def create_task(
        self,
        task_id: str,
        agent_count: int,
        session_prefix: str,
    ) -> TaskState:
        """
        Create a new task with agents.

        Convenience method that generates agent IDs and session names.

        Args:
            task_id: Unique task identifier
            agent_count: Number of agents to create
            session_prefix: Prefix for session names

        Returns:
            Created TaskState
        """
        agent_ids = [f"{session_prefix}-agent-{i}" for i in range(agent_count)]
        session_names = {aid: aid for aid in agent_ids}

        return self.state.create_task(task_id, agent_ids, session_names)

    def handle_request(self, request: dict) -> dict:
        """
        Handle a JSON-RPC 2.0 request.

        Args:
            request: JSON-RPC request object

        Returns:
            JSON-RPC response object
        """
        request_id = request.get("id")
        method = request.get("method", "")
        params = request.get("params", {})

        try:
            if method == "initialize":
                result = self._handle_initialize(params)
            elif method == "tools/list":
                result = {"tools": self.list_tools()}
            elif method == "tools/call":
                tool_name = params.get("name", "")
                arguments = params.get("arguments", {})
                tool_result = self.call_tool(tool_name, arguments)
                result = {
                    "content": [
                        {"type": "text", "text": json.dumps(tool_result)}
                    ]
                }
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}",
                    },
                }

            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result,
            }

        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": str(e),
                },
            }

    def _handle_initialize(self, params: dict) -> dict:
        """Handle MCP initialize request."""
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {},
            },
            "serverInfo": {
                "name": "swarm-orchestrator",
                "version": "0.1.0",
            },
        }

    def run_stdio(self) -> None:
        """
        Run the server in stdio mode.

        Reads JSON-RPC requests from stdin and writes responses to stdout.
        """
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break

                line = line.strip()
                if not line:
                    continue

                request = json.loads(line)
                response = self.handle_request(request)

                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()

            except json.JSONDecodeError as e:
                error_response = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": -32700,
                        "message": f"Parse error: {e}",
                    },
                }
                sys.stdout.write(json.dumps(error_response) + "\n")
                sys.stdout.flush()
            except KeyboardInterrupt:
                break


def main():
    """Entry point for running the server."""
    import argparse

    parser = argparse.ArgumentParser(description="Swarm MCP Server")
    parser.add_argument(
        "--state-file",
        help="Path to state persistence file",
        default=None,
    )
    parser.add_argument(
        "--config",
        help="Path to config file (default: .swarm/config.json)",
        default=None,
    )
    args = parser.parse_args()

    config = load_config(args.config)
    server = SwarmMCPServer(
        persistence_path=args.state_file,
        compression_config=config,
    )
    server.run_stdio()


if __name__ == "__main__":
    main()
