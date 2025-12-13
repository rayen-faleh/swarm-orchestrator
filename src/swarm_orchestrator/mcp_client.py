"""
MCP (Model Context Protocol) client for communicating with MCP servers via stdio.

This implements a simple JSON-RPC 2.0 client that spawns and communicates with
MCP servers like Schaltwerk.
"""

import json
import subprocess
import threading
import queue
from dataclasses import dataclass
from typing import Any, Optional
from pathlib import Path


@dataclass
class MCPConfig:
    """Configuration for an MCP server."""
    command: str
    args: list[str]
    env: dict[str, str]


class MCPClient:
    """
    Client for communicating with MCP servers via stdio using JSON-RPC 2.0.

    Usage:
        client = MCPClient.from_config_file(".mcp.json", "schaltwerk")
        client.start()
        result = client.call_tool("schaltwerk_list", {})
        client.stop()
    """

    def __init__(self, config: MCPConfig):
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self.request_id = 0
        self._response_queue: queue.Queue = queue.Queue()
        self._reader_thread: Optional[threading.Thread] = None
        self._running = False

    @classmethod
    def from_config_file(cls, config_path: str, server_name: str) -> "MCPClient":
        """Create an MCP client from a .mcp.json config file."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"MCP config file not found: {config_path}")

        with open(path) as f:
            config_data = json.load(f)

        servers = config_data.get("mcpServers", {})
        if server_name not in servers:
            raise ValueError(f"Server '{server_name}' not found in config")

        server_config = servers[server_name]
        return cls(MCPConfig(
            command=server_config["command"],
            args=server_config.get("args", []),
            env=server_config.get("env", {}),
        ))

    def start(self) -> None:
        """Start the MCP server process."""
        if self.process is not None:
            return

        cmd = [self.config.command] + self.config.args
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            env={**dict(__import__('os').environ), **self.config.env},
        )

        self._running = True
        self._reader_thread = threading.Thread(target=self._read_responses, daemon=True)
        self._reader_thread.start()

        # Initialize the connection
        self._initialize()

    def stop(self) -> None:
        """Stop the MCP server process."""
        self._running = False
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None

    def _initialize(self) -> dict:
        """Send the initialize request to the MCP server."""
        return self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "swarm-orchestrator",
                "version": "0.1.0"
            }
        })

    def _read_responses(self) -> None:
        """Background thread to read responses from the server."""
        while self._running and self.process and self.process.stdout:
            try:
                line = self.process.stdout.readline()
                if not line:
                    break

                # Try to parse as JSON-RPC response
                try:
                    response = json.loads(line.strip())
                    self._response_queue.put(response)
                except json.JSONDecodeError:
                    # Not JSON, might be a log line - ignore
                    pass
            except Exception:
                break

    def _send_request(self, method: str, params: dict, timeout: float = 30.0) -> dict:
        """Send a JSON-RPC request and wait for the response."""
        if not self.process or not self.process.stdin:
            raise RuntimeError("MCP server not started")

        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method,
            "params": params,
        }

        # Send request
        request_str = json.dumps(request) + "\n"
        self.process.stdin.write(request_str)
        self.process.stdin.flush()

        # Wait for response with matching ID
        while True:
            try:
                response = self._response_queue.get(timeout=timeout)
                if response.get("id") == self.request_id:
                    if "error" in response:
                        raise RuntimeError(f"MCP error: {response['error']}")
                    return response.get("result", {})
                # Put back responses for other requests
                self._response_queue.put(response)
            except queue.Empty:
                raise TimeoutError(f"Timeout waiting for response to {method}")

    def call_tool(self, tool_name: str, arguments: dict, timeout: float = 60.0) -> Any:
        """Call an MCP tool and return the result."""
        result = self._send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments,
        }, timeout=timeout)

        # Extract content from the result
        content = result.get("content", [])
        if content and len(content) > 0:
            first_content = content[0]
            if first_content.get("type") == "text":
                text = first_content.get("text", "")
                # Try to parse as JSON
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return text
        return result

    def list_tools(self) -> list[dict]:
        """List available tools from the MCP server."""
        result = self._send_request("tools/list", {})
        return result.get("tools", [])

    def __enter__(self) -> "MCPClient":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
