"""
Tests for the MCP client module.

TDD approach: These tests define the expected behavior of the MCP client.
"""

import pytest
import json
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path

from swarm_orchestrator.mcp_client import MCPClient, MCPConfig


class TestMCPConfig:
    """Tests for MCPConfig dataclass."""

    def test_create_config(self):
        """Should create a config with all fields."""
        config = MCPConfig(
            command="node",
            args=["/path/to/server.js"],
            env={"FOO": "bar"},
        )

        assert config.command == "node"
        assert config.args == ["/path/to/server.js"]
        assert config.env == {"FOO": "bar"}


class TestMCPClientFromConfig:
    """Tests for MCPClient.from_config_file()."""

    def test_loads_valid_config(self, tmp_path):
        """Should load config from a valid .mcp.json file."""
        config_data = {
            "mcpServers": {
                "test-server": {
                    "type": "stdio",
                    "command": "node",
                    "args": ["/path/to/server.js"],
                    "env": {"KEY": "value"},
                }
            }
        }

        config_file = tmp_path / ".mcp.json"
        config_file.write_text(json.dumps(config_data))

        client = MCPClient.from_config_file(str(config_file), "test-server")

        assert client.config.command == "node"
        assert client.config.args == ["/path/to/server.js"]
        assert client.config.env == {"KEY": "value"}

    def test_raises_on_missing_file(self):
        """Should raise FileNotFoundError for missing config."""
        with pytest.raises(FileNotFoundError):
            MCPClient.from_config_file("/nonexistent/.mcp.json", "server")

    def test_raises_on_missing_server(self, tmp_path):
        """Should raise ValueError for unknown server name."""
        config_data = {"mcpServers": {"other-server": {}}}
        config_file = tmp_path / ".mcp.json"
        config_file.write_text(json.dumps(config_data))

        with pytest.raises(ValueError, match="not found"):
            MCPClient.from_config_file(str(config_file), "missing-server")

    def test_handles_missing_optional_fields(self, tmp_path):
        """Should handle missing args and env fields."""
        config_data = {
            "mcpServers": {
                "minimal": {
                    "command": "python",
                }
            }
        }

        config_file = tmp_path / ".mcp.json"
        config_file.write_text(json.dumps(config_data))

        client = MCPClient.from_config_file(str(config_file), "minimal")

        assert client.config.command == "python"
        assert client.config.args == []
        assert client.config.env == {}

    def test_loads_from_settings_json_format(self, tmp_path):
        """Should load config from settings.json with mcpServers at top level."""
        # This is the format used by ~/.claude/settings.json
        config_data = {
            "mcpServers": {
                "schaltwerk": {
                    "command": "npx",
                    "args": ["-y", "@schaltwerk/mcp"],
                    "env": {"REPO_PATH": "/path/to/repo"},
                }
            },
            "permissions": {},  # Other settings.json fields should be ignored
            "enabledFeatures": ["mcp"],
        }

        settings_file = tmp_path / "settings.json"
        settings_file.write_text(json.dumps(config_data))

        client = MCPClient.from_config_file(str(settings_file), "schaltwerk")

        assert client.config.command == "npx"
        assert client.config.args == ["-y", "@schaltwerk/mcp"]
        assert client.config.env == {"REPO_PATH": "/path/to/repo"}


class TestMCPClientLifecycle:
    """Tests for MCP client start/stop lifecycle."""

    @pytest.fixture
    def mock_popen(self):
        """Mock subprocess.Popen."""
        with patch("swarm_orchestrator.mcp_client.subprocess.Popen") as mock:
            process = MagicMock()
            process.stdin = MagicMock()
            process.stdout = MagicMock()
            process.stderr = MagicMock()
            process.stdout.readline.return_value = ""
            mock.return_value = process
            yield mock

    def test_start_spawns_process(self, mock_popen):
        """Should spawn the server process on start."""
        config = MCPConfig(command="node", args=["server.js"], env={})
        client = MCPClient(config)

        # Mock the initialize response
        mock_popen.return_value.stdout.readline.side_effect = [
            '{"jsonrpc": "2.0", "id": 1, "result": {}}\n',
            "",  # End of stream
        ]

        client.start()

        mock_popen.assert_called_once()
        call_args = mock_popen.call_args
        assert call_args[0][0] == ["node", "server.js"]

    def test_stop_terminates_process(self, mock_popen):
        """Should terminate the process on stop."""
        config = MCPConfig(command="node", args=[], env={})
        client = MCPClient(config)

        mock_popen.return_value.stdout.readline.side_effect = [
            '{"jsonrpc": "2.0", "id": 1, "result": {}}\n',
            "",
        ]

        client.start()
        client.stop()

        mock_popen.return_value.terminate.assert_called_once()

    def test_context_manager(self, mock_popen):
        """Should work as a context manager."""
        config = MCPConfig(command="node", args=[], env={})

        mock_popen.return_value.stdout.readline.side_effect = [
            '{"jsonrpc": "2.0", "id": 1, "result": {}}\n',
            "",
        ]

        with MCPClient(config) as client:
            assert client.process is not None

        mock_popen.return_value.terminate.assert_called()


class TestMCPClientCallTool:
    """Tests for MCPClient.call_tool()."""

    @pytest.fixture
    def started_client(self):
        """Create a started client with mocked process."""
        with patch("swarm_orchestrator.mcp_client.subprocess.Popen") as mock_popen:
            process = MagicMock()
            process.stdin = MagicMock()
            process.stdout = MagicMock()
            process.stderr = MagicMock()
            mock_popen.return_value = process

            config = MCPConfig(command="node", args=[], env={})
            client = MCPClient(config)

            # Mock initialize response
            process.stdout.readline.side_effect = [
                '{"jsonrpc": "2.0", "id": 1, "result": {}}\n',
            ]

            client.start()
            client._running = True

            # Reset readline for subsequent calls
            process.stdout.readline.side_effect = None

            yield client, process

    def test_call_tool_sends_request(self, started_client):
        """Should send a properly formatted JSON-RPC request."""
        client, process = started_client

        # Queue up the response
        client._response_queue.put({
            "jsonrpc": "2.0",
            "id": 2,
            "result": {"content": [{"type": "text", "text": '{"success": true}'}]},
        })

        result = client.call_tool("test_tool", {"arg": "value"})

        # Check the request was sent
        process.stdin.write.assert_called()
        written = process.stdin.write.call_args[0][0]
        request = json.loads(written.strip())

        assert request["method"] == "tools/call"
        assert request["params"]["name"] == "test_tool"
        assert request["params"]["arguments"] == {"arg": "value"}

    def test_call_tool_parses_json_response(self, started_client):
        """Should parse JSON content from tool response."""
        client, process = started_client

        client._response_queue.put({
            "jsonrpc": "2.0",
            "id": 2,
            "result": {
                "content": [{"type": "text", "text": '{"data": "test"}'}]
            },
        })

        result = client.call_tool("test_tool", {})

        assert result == {"data": "test"}

    def test_call_tool_returns_text_on_parse_failure(self, started_client):
        """Should return raw text if JSON parsing fails."""
        client, process = started_client

        client._response_queue.put({
            "jsonrpc": "2.0",
            "id": 2,
            "result": {
                "content": [{"type": "text", "text": "plain text response"}]
            },
        })

        result = client.call_tool("test_tool", {})

        assert result == "plain text response"

    def test_call_tool_raises_on_error(self, started_client):
        """Should raise RuntimeError on MCP error response."""
        client, process = started_client

        client._response_queue.put({
            "jsonrpc": "2.0",
            "id": 2,
            "error": {"code": -32600, "message": "Invalid request"},
        })

        with pytest.raises(RuntimeError, match="MCP error"):
            client.call_tool("test_tool", {})
