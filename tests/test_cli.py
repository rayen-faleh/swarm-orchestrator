"""
Tests for the CLI commands.

TDD approach: These tests define the expected behavior of CLI commands.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from click.testing import CliRunner

from swarm_orchestrator.cli import main, init
from swarm_orchestrator.installation import (
    InstallationContext,
    InstallationType,
    PythonProject,
)


class TestInitCommand:
    """Tests for the 'swarm init' command."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def temp_project(self, tmp_path):
        """Create a temporary project directory."""
        return tmp_path

    def test_init_creates_mcp_config(self, runner, temp_project):
        """Should create .mcp.json with swarm-orchestrator server config."""
        with runner.isolated_filesystem(temp_dir=temp_project):
            result = runner.invoke(main, ["init", "--non-interactive"])

            assert result.exit_code == 0
            assert Path(".mcp.json").exists()

            config = json.loads(Path(".mcp.json").read_text())
            assert "mcpServers" in config
            assert "swarm-orchestrator" in config["mcpServers"]

    def test_init_adds_to_existing_config(self, runner, temp_project):
        """Should add swarm-orchestrator to existing .mcp.json."""
        with runner.isolated_filesystem(temp_dir=temp_project):
            # Create existing config
            existing = {
                "mcpServers": {
                    "schaltwerk": {"command": "schaltwerk"}
                }
            }
            Path(".mcp.json").write_text(json.dumps(existing))

            result = runner.invoke(main, ["init", "--non-interactive"])

            assert result.exit_code == 0

            config = json.loads(Path(".mcp.json").read_text())
            assert "schaltwerk" in config["mcpServers"]
            assert "swarm-orchestrator" in config["mcpServers"]

    def test_init_config_uses_uv_for_local_python_project(self, runner, temp_project):
        """Config uses uv run for local install in Python project."""
        with runner.isolated_filesystem(temp_dir=temp_project):
            # Create a Python project marker
            Path("pyproject.toml").write_text("[project]\nname = 'test'")

            # Mock local install in Python project
            def mock_detect(directory=None):
                return InstallationContext(
                    installation_type=InstallationType.LOCAL,
                    python_project=PythonProject(
                        root=Path.cwd().resolve(),
                        has_pyproject_toml=True,
                    ),
                    swarm_location=Path.cwd() / ".venv",
                )

            with patch("swarm_orchestrator.cli.detect_installation_context", mock_detect):
                result = runner.invoke(main, ["init", "--non-interactive"])

            config = json.loads(Path(".mcp.json").read_text())
            server = config["mcpServers"]["swarm-orchestrator"]

            # Uses uv to run from main repo's environment
            assert server["command"] == "uv"
            assert "run" in server["args"]
            assert "--project" in server["args"]
            assert "-m" in server["args"]
            assert "swarm_orchestrator.swarm_mcp.server" in server["args"]

    def test_init_config_uses_direct_command_for_global_install(self, runner, temp_project):
        """Config uses direct swarm command for global (pipx/system) install."""
        with runner.isolated_filesystem(temp_dir=temp_project):
            # Mock global install
            def mock_detect(directory=None):
                return InstallationContext(
                    installation_type=InstallationType.PIPX,
                    python_project=None,
                    swarm_location=Path.home() / ".local" / "pipx" / "venvs" / "swarm",
                )

            with patch("swarm_orchestrator.cli.detect_installation_context", mock_detect):
                result = runner.invoke(main, ["init", "--non-interactive"])

            config = json.loads(Path(".mcp.json").read_text())
            server = config["mcpServers"]["swarm-orchestrator"]

            # Uses direct swarm command
            assert server["command"] == "swarm"
            assert "server" in server["args"]
            assert "--state-file" in server["args"]
            # Should NOT have uv-specific args
            assert "run" not in server["args"]
            assert "--project" not in server["args"]

    def test_init_config_uses_direct_command_for_non_python_project(self, runner, temp_project):
        """Config uses direct swarm command for non-Python project."""
        with runner.isolated_filesystem(temp_dir=temp_project):
            # No pyproject.toml or other Python project markers

            # Mock local install but NOT in Python project
            def mock_detect(directory=None):
                return InstallationContext(
                    installation_type=InstallationType.LOCAL,
                    python_project=None,  # Not a Python project
                    swarm_location=Path.cwd() / ".venv",
                )

            with patch("swarm_orchestrator.cli.detect_installation_context", mock_detect):
                result = runner.invoke(main, ["init", "--non-interactive"])

            config = json.loads(Path(".mcp.json").read_text())
            server = config["mcpServers"]["swarm-orchestrator"]

            # Uses direct swarm command (not uv)
            assert server["command"] == "swarm"
            assert "server" in server["args"]

    def test_init_creates_state_directory(self, runner, temp_project):
        """Should create .swarm directory for state persistence."""
        with runner.isolated_filesystem(temp_dir=temp_project):
            result = runner.invoke(main, ["init", "--non-interactive"])

            assert result.exit_code == 0
            assert Path(".swarm").is_dir()

    def test_init_state_file_path_in_config(self, runner, temp_project):
        """Config should include absolute state file path in args."""
        with runner.isolated_filesystem(temp_dir=temp_project):
            result = runner.invoke(main, ["init", "--non-interactive"])

            config = json.loads(Path(".mcp.json").read_text())
            server = config["mcpServers"]["swarm-orchestrator"]

            assert "--state-file" in server["args"]
            # State file should be an absolute path containing .swarm/state.json
            state_file_idx = server["args"].index("--state-file") + 1
            state_file_path = server["args"][state_file_idx]
            assert ".swarm/state.json" in state_file_path
            assert Path(state_file_path).is_absolute()

    def test_init_skips_if_already_configured(self, runner, temp_project):
        """Should skip if swarm-orchestrator already in config."""
        with runner.isolated_filesystem(temp_dir=temp_project):
            existing = {
                "mcpServers": {
                    "swarm-orchestrator": {
                        "command": "python",
                        "args": ["-m", "swarm_orchestrator.swarm_mcp.server"],
                    }
                }
            }
            Path(".mcp.json").write_text(json.dumps(existing))

            result = runner.invoke(main, ["init"])

            assert result.exit_code == 0
            assert "already configured" in result.output.lower()

    def test_init_force_overwrites(self, runner, temp_project):
        """--force should overwrite existing config."""
        with runner.isolated_filesystem(temp_dir=temp_project):
            existing = {
                "mcpServers": {
                    "swarm-orchestrator": {
                        "command": "old-command",
                        "args": [],
                    }
                }
            }
            Path(".mcp.json").write_text(json.dumps(existing))

            # Mock global install so we expect swarm command
            def mock_detect(directory=None):
                return InstallationContext(
                    installation_type=InstallationType.PIPX,
                    python_project=None,
                    swarm_location=Path.home() / ".local" / "pipx",
                )

            with patch("swarm_orchestrator.cli.detect_installation_context", mock_detect):
                result = runner.invoke(main, ["init", "--force", "--non-interactive"])

            assert result.exit_code == 0

            config = json.loads(Path(".mcp.json").read_text())
            server = config["mcpServers"]["swarm-orchestrator"]
            # Should now use swarm (not old-command)
            assert server["command"] == "swarm"

    def test_init_shows_success_message(self, runner, temp_project):
        """Should show success message with instructions."""
        with runner.isolated_filesystem(temp_dir=temp_project):
            result = runner.invoke(main, ["init", "--non-interactive"])

            assert result.exit_code == 0
            assert "initialized" in result.output.lower() or "success" in result.output.lower()

    def test_init_interactive_prompts_for_backends(self, runner, temp_project):
        """Interactive mode should prompt for all backend selections."""
        with runner.isolated_filesystem(temp_dir=temp_project):
            # Simulate user selecting defaults for all prompts (input "1" for each)
            result = runner.invoke(main, ["init"], input="1\n1\n1\n")

            assert result.exit_code == 0
            # Should prompt for backends
            assert "worktree" in result.output.lower()
            assert "agent" in result.output.lower()
            assert "llm" in result.output.lower()

    def test_init_interactive_creates_config_with_selections(self, runner, temp_project):
        """Interactive mode should create config.json with selected backends."""
        with runner.isolated_filesystem(temp_dir=temp_project):
            # Select option 1 for each backend
            result = runner.invoke(main, ["init"], input="1\n1\n1\n")

            assert result.exit_code == 0
            assert Path(".swarm/config.json").exists()

            config = json.loads(Path(".swarm/config.json").read_text())
            assert "worktree_backend" in config
            assert "agent_backend" in config
            assert "llm_backend" in config

    def test_init_interactive_anthropic_api_prompts_for_model(self, runner, temp_project):
        """When anthropic-api is selected, should prompt for model preference."""
        with runner.isolated_filesystem(temp_dir=temp_project):
            # Select: 1 (worktree), 1 (agent), 2 (anthropic-api), 1 (default model)
            result = runner.invoke(main, ["init"], input="1\n1\n2\n1\n")

            assert result.exit_code == 0
            config = json.loads(Path(".swarm/config.json").read_text())
            assert config["llm_backend"] == "anthropic-api"

    def test_init_non_interactive_when_not_tty(self, runner, temp_project):
        """Non-interactive mode should use defaults when stdin is not a tty."""
        with runner.isolated_filesystem(temp_dir=temp_project):
            # Pass no input - should use defaults without prompting
            result = runner.invoke(main, ["init", "--non-interactive"])

            assert result.exit_code == 0
            # Should create config with defaults
            if Path(".swarm/config.json").exists():
                config = json.loads(Path(".swarm/config.json").read_text())
                assert config["worktree_backend"] == "schaltwerk"
                assert config["agent_backend"] == "schaltwerk"
                assert config["llm_backend"] == "claude-cli"

    def test_init_force_allows_reinit(self, runner, temp_project):
        """--force with interactive should allow re-initialization."""
        with runner.isolated_filesystem(temp_dir=temp_project):
            # First init
            runner.invoke(main, ["init"], input="1\n1\n1\n")

            # Second init with force
            result = runner.invoke(main, ["init", "--force"], input="1\n1\n2\n1\n")

            assert result.exit_code == 0
            config = json.loads(Path(".swarm/config.json").read_text())
            assert config["llm_backend"] == "anthropic-api"


class TestRunCommandWithMCP:
    """Tests for the 'swarm run' command with MCP integration."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()

    def test_run_checks_for_init(self, runner):
        """Should check if swarm is initialized before running."""
        with runner.isolated_filesystem():
            result = runner.invoke(main, ["run", "test task"])

            # Should fail or warn if not initialized
            assert result.exit_code != 0 or "init" in result.output.lower()


class TestServerCommand:
    """Tests for the 'swarm server' command."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()

    def test_server_command_exists(self, runner):
        """Should have a server command available."""
        result = runner.invoke(main, ["server", "--help"])

        # Should either succeed or show help
        assert result.exit_code == 0
        assert "server" in result.output.lower() or "usage" in result.output.lower()

    def test_server_accepts_state_file(self, runner):
        """Should accept --state-file option."""
        result = runner.invoke(main, ["server", "--help"])

        assert "--state-file" in result.output


class TestHelpDocumentation:
    """Tests for CLI help text documentation."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()

    def test_main_help_shows_overview(self, runner):
        """Main help should show configuration overview."""
        result = runner.invoke(main, ["--help"])

        assert result.exit_code == 0
        assert "Multi-agent consensus" in result.output
        assert "Configuration" in result.output
        assert ".swarm/config.json" in result.output

    def test_main_help_lists_backends(self, runner):
        """Main help should mention backend options."""
        result = runner.invoke(main, ["--help"])

        assert result.exit_code == 0
        assert "--worktree-backend" in result.output
        assert "--agent-backend" in result.output
        assert "--llm-backend" in result.output

    def test_run_help_documents_worktree_backend(self, runner):
        """Run help should document worktree backend options."""
        result = runner.invoke(main, ["run", "--help"])

        assert result.exit_code == 0
        assert "--worktree-backend" in result.output
        assert "schaltwerk" in result.output.lower()

    def test_run_help_documents_agent_backend(self, runner):
        """Run help should document agent backend options."""
        result = runner.invoke(main, ["run", "--help"])

        assert result.exit_code == 0
        assert "--agent-backend" in result.output
        assert "schaltwerk" in result.output.lower()

    def test_run_help_documents_llm_backends(self, runner):
        """Run help should document LLM backend options with descriptions."""
        result = runner.invoke(main, ["run", "--help"])

        assert result.exit_code == 0
        assert "--llm-backend" in result.output
        assert "claude-cli" in result.output
        assert "anthropic-api" in result.output
        # Should explain when to use each
        assert "Claude Code" in result.output or "CLI" in result.output
        assert "API" in result.output

    def test_run_help_documents_llm_model(self, runner):
        """Run help should document --llm-model option."""
        result = runner.invoke(main, ["run", "--help"])

        assert result.exit_code == 0
        assert "--llm-model" in result.output
        assert "anthropic-api" in result.output

    def test_run_help_documents_config_file(self, runner):
        """Run help should document config file option."""
        result = runner.invoke(main, ["run", "--help"])

        assert result.exit_code == 0
        assert "--config" in result.output
        assert ".swarm/config.json" in result.output

    def test_init_help_shows_config_format(self, runner):
        """Init help should show config file format."""
        result = runner.invoke(main, ["init", "--help"])

        assert result.exit_code == 0
        assert ".swarm/config.json" in result.output
        # Help text now describes interactive prompts instead of config format
        assert "Worktree backend" in result.output
        assert "LLM backend" in result.output


class TestConfigCommand:
    """Tests for the 'swarm config' command group."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def temp_project(self, tmp_path):
        """Create a temporary project directory."""
        return tmp_path

    def test_config_show_displays_current_settings(self, runner, temp_project):
        """'swarm config show' should display current configuration."""
        with runner.isolated_filesystem(temp_dir=temp_project):
            # First initialize
            runner.invoke(main, ["init", "--non-interactive"])

            result = runner.invoke(main, ["config", "show"])

            assert result.exit_code == 0
            # Should show all config keys
            assert "worktree_backend" in result.output or "worktree-backend" in result.output
            assert "agent_backend" in result.output or "agent-backend" in result.output
            assert "llm_backend" in result.output or "llm-backend" in result.output
            # Should show current values
            assert "schaltwerk" in result.output
            assert "claude-cli" in result.output

    def test_config_show_fails_without_init(self, runner, temp_project):
        """'swarm config show' should fail gracefully if not initialized."""
        with runner.isolated_filesystem(temp_dir=temp_project):
            result = runner.invoke(main, ["config", "show"])

            assert result.exit_code != 0
            # Should suggest running init
            assert "init" in result.output.lower()

    def test_config_set_updates_value(self, runner, temp_project):
        """'swarm config set <key> <value>' should update config."""
        with runner.isolated_filesystem(temp_dir=temp_project):
            # First initialize
            runner.invoke(main, ["init", "--non-interactive"])

            # Change llm-backend to anthropic-api
            result = runner.invoke(main, ["config", "set", "llm-backend", "anthropic-api"])

            assert result.exit_code == 0

            # Verify change persisted
            config = json.loads(Path(".swarm/config.json").read_text())
            assert config["llm_backend"] == "anthropic-api"

    def test_config_set_rejects_invalid_value(self, runner, temp_project):
        """'swarm config set' should reject invalid values with helpful error."""
        with runner.isolated_filesystem(temp_dir=temp_project):
            # First initialize
            runner.invoke(main, ["init", "--non-interactive"])

            result = runner.invoke(main, ["config", "set", "llm-backend", "invalid-backend"])

            assert result.exit_code != 0
            # Should list valid options
            assert "claude-cli" in result.output
            assert "anthropic-api" in result.output

    def test_config_set_rejects_invalid_key(self, runner, temp_project):
        """'swarm config set' should reject invalid keys."""
        with runner.isolated_filesystem(temp_dir=temp_project):
            # First initialize
            runner.invoke(main, ["init", "--non-interactive"])

            result = runner.invoke(main, ["config", "set", "invalid-key", "value"])

            assert result.exit_code != 0
            # Should indicate invalid key
            assert "invalid" in result.output.lower() or "unknown" in result.output.lower()

    def test_config_set_fails_without_init(self, runner, temp_project):
        """'swarm config set' should fail gracefully if not initialized."""
        with runner.isolated_filesystem(temp_dir=temp_project):
            result = runner.invoke(main, ["config", "set", "llm-backend", "anthropic-api"])

            assert result.exit_code != 0
            # Should suggest running init
            assert "init" in result.output.lower()

    def test_config_changes_persist(self, runner, temp_project):
        """Config changes should persist and be reflected in subsequent 'config show'."""
        with runner.isolated_filesystem(temp_dir=temp_project):
            # First initialize
            runner.invoke(main, ["init", "--non-interactive"])

            # Change value
            runner.invoke(main, ["config", "set", "llm-backend", "anthropic-api"])

            # Verify change is reflected in show
            result = runner.invoke(main, ["config", "show"])

            assert result.exit_code == 0
            assert "anthropic-api" in result.output

    def test_config_show_displays_valid_options(self, runner, temp_project):
        """'swarm config show' should display valid options for each setting."""
        with runner.isolated_filesystem(temp_dir=temp_project):
            # First initialize
            runner.invoke(main, ["init", "--non-interactive"])

            result = runner.invoke(main, ["config", "show"])

            assert result.exit_code == 0
            # Should show valid options for llm-backend
            assert "claude-cli" in result.output
            assert "anthropic-api" in result.output

    def test_config_set_llm_model(self, runner, temp_project):
        """'swarm config set' should allow setting llm-model."""
        with runner.isolated_filesystem(temp_dir=temp_project):
            # First initialize
            runner.invoke(main, ["init", "--non-interactive"])

            result = runner.invoke(main, ["config", "set", "llm-model", "claude-opus-4-20250514"])

            assert result.exit_code == 0

            # Verify change persisted
            config = json.loads(Path(".swarm/config.json").read_text())
            assert config["llm_model"] == "claude-opus-4-20250514"
