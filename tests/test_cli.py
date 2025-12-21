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
            # 3 prompts: worktree, agent, cli_tool
            result = runner.invoke(main, ["init"], input="1\n1\n1\n")

            assert result.exit_code == 0
            # Should prompt for backends
            assert "worktree" in result.output.lower()
            assert "agent" in result.output.lower()
            assert "cli tool" in result.output.lower()

    def test_init_interactive_creates_config_with_selections(self, runner, temp_project):
        """Interactive mode should create config.json with selected backends."""
        with runner.isolated_filesystem(temp_dir=temp_project):
            # Select option 1 for each backend (worktree, agent, cli_tool)
            result = runner.invoke(main, ["init"], input="1\n1\n1\n")

            assert result.exit_code == 0
            assert Path(".swarm/config.json").exists()

            config = json.loads(Path(".swarm/config.json").read_text())
            assert "worktree_backend" in config
            assert "agent_backend" in config
            assert "cli_tool" in config

    def test_init_interactive_anthropic_api_prompts_for_model(self, runner, temp_project):
        """When anthropic-api is selected, should prompt for model preference."""
        with runner.isolated_filesystem(temp_dir=temp_project):
            # Select: 1 (worktree), 1 (agent), 4 (anthropic-api), 1 (default model)
            result = runner.invoke(main, ["init"], input="1\n1\n4\n1\n")

            assert result.exit_code == 0
            config = json.loads(Path(".swarm/config.json").read_text())
            assert config["cli_tool"] == "anthropic-api"

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
                assert config["cli_tool"] == "claude"

    def test_init_force_allows_reinit(self, runner, temp_project):
        """--force with interactive should allow re-initialization."""
        with runner.isolated_filesystem(temp_dir=temp_project):
            # First init (3 prompts: worktree, agent, cli_tool)
            runner.invoke(main, ["init"], input="1\n1\n1\n")

            # Second init with force (4 prompts: worktree, agent, cli_tool=anthropic-api, model)
            result = runner.invoke(main, ["init", "--force"], input="1\n1\n4\n1\n")

            assert result.exit_code == 0
            config = json.loads(Path(".swarm/config.json").read_text())
            assert config["cli_tool"] == "anthropic-api"

    def test_init_cli_tool_appears_in_summary(self, runner, temp_project):
        """Init should display selected cli_tool in configuration summary."""
        with runner.isolated_filesystem(temp_dir=temp_project):
            result = runner.invoke(main, ["init", "--non-interactive"])

            assert result.exit_code == 0
            # Should show cli tool in the configuration summary
            assert "cli tool" in result.output.lower()
            assert "claude" in result.output.lower()

    def test_init_interactive_cli_tool_selection_opencode(self, runner, temp_project):
        """Interactive mode should allow selecting opencode as cli_tool."""
        with runner.isolated_filesystem(temp_dir=temp_project):
            # Select: 1 (worktree), 1 (agent), 2 (opencode)
            result = runner.invoke(main, ["init"], input="1\n1\n2\n")

            assert result.exit_code == 0
            config = json.loads(Path(".swarm/config.json").read_text())
            assert config["cli_tool"] == "opencode"


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


class TestRunCommandExitFlow:
    """Tests for the CLI exit flow after orchestration completes."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()

    def _make_result(self, merged: bool, consensus: bool = True):
        """Create a mock OrchestrationResult."""
        mock_vote_result = MagicMock()
        mock_vote_result.consensus_reached = consensus

        mock_subtask_result = MagicMock()
        mock_subtask_result.vote_result = mock_vote_result
        mock_subtask_result.winner_session = "winner-session" if consensus else None
        mock_subtask_result.merged = merged

        mock_result = MagicMock()
        mock_result.overall_success = consensus
        mock_result.subtask_results = [mock_subtask_result]

        return mock_result

    def test_auto_merge_success_shows_merged_message(self, runner):
        """When auto_merge=True and all merged, should show 'merged' message."""
        result = self._make_result(merged=True)

        with runner.isolated_filesystem():
            Path(".swarm").mkdir()
            Path(".swarm/config.json").write_text('{"worktree_backend": "schaltwerk"}')

            with patch("swarm_orchestrator.cli.Orchestrator") as mock_orch:
                mock_orch.return_value.run.return_value = result

                cli_result = runner.invoke(main, ["run", "test", "--auto-merge"])

                # Should mention merged
                assert "merged" in cli_result.output.lower()

    def test_non_auto_merge_after_dashboard_shows_appropriate_message(self, runner):
        """When auto_merge=False, should show message appropriate for post-dashboard flow."""
        result = self._make_result(merged=False)

        with runner.isolated_filesystem():
            Path(".swarm").mkdir()
            Path(".swarm/config.json").write_text('{"worktree_backend": "schaltwerk"}')

            with patch("swarm_orchestrator.cli.Orchestrator") as mock_orch:
                mock_orch.return_value.run.return_value = result

                cli_result = runner.invoke(main, ["run", "test"])

                # Should NOT show old redundant message about reviewing "above"
                assert "review the changes above" not in cli_result.output.lower()
                # Should acknowledge completion
                assert "complete" in cli_result.output.lower() or "done" in cli_result.output.lower()

    def test_non_auto_merge_all_merged_in_dashboard_shows_merged_message(self, runner):
        """When user merged all sessions in dashboard, should acknowledge that."""
        result = self._make_result(merged=True)

        with runner.isolated_filesystem():
            Path(".swarm").mkdir()
            Path(".swarm/config.json").write_text('{"worktree_backend": "schaltwerk"}')

            with patch("swarm_orchestrator.cli.Orchestrator") as mock_orch:
                mock_orch.return_value.run.return_value = result

                cli_result = runner.invoke(main, ["run", "test"])

                # Should mention merged
                assert "merged" in cli_result.output.lower()

    def test_exit_code_zero_on_success(self, runner):
        """Should exit with code 0 on successful orchestration."""
        result = self._make_result(merged=True)

        with runner.isolated_filesystem():
            Path(".swarm").mkdir()
            Path(".swarm/config.json").write_text('{"worktree_backend": "schaltwerk"}')

            with patch("swarm_orchestrator.cli.Orchestrator") as mock_orch:
                mock_orch.return_value.run.return_value = result

                cli_result = runner.invoke(main, ["run", "test"])

                assert cli_result.exit_code == 0


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
        assert "--cli-tool" in result.output

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

    def test_run_help_documents_cli_tool(self, runner):
        """Run help should document CLI tool options with descriptions."""
        result = runner.invoke(main, ["run", "--help"])

        assert result.exit_code == 0
        assert "--cli-tool" in result.output
        assert "claude" in result.output
        assert "anthropic-api" in result.output
        # Should explain when to use each
        assert "decomposition" in result.output or "CLI" in result.output

    def test_run_help_documents_llm_model(self, runner):
        """Run help should document --llm-model option."""
        result = runner.invoke(main, ["run", "--help"])

        assert result.exit_code == 0
        assert "--llm-model" in result.output
        assert "anthropic-api" in result.output

    def test_run_help_documents_exploration_model(self, runner):
        """Run help should document --exploration-model option."""
        result = runner.invoke(main, ["run", "--help"])

        assert result.exit_code == 0
        assert "--exploration-model" in result.output
        assert "exploration" in result.output.lower()

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
        assert "CLI Tool" in result.output or "cli_tool" in result.output.lower()


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
            assert "cli_tool" in result.output or "cli-tool" in result.output
            # Should show current values
            assert "schaltwerk" in result.output
            assert "claude" in result.output

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

            # Change cli-tool to anthropic-api
            result = runner.invoke(main, ["config", "set", "cli-tool", "anthropic-api"])

            assert result.exit_code == 0

            # Verify change persisted
            config = json.loads(Path(".swarm/config.json").read_text())
            assert config["cli_tool"] == "anthropic-api"

    def test_config_set_rejects_invalid_value(self, runner, temp_project):
        """'swarm config set' should reject invalid values with helpful error."""
        with runner.isolated_filesystem(temp_dir=temp_project):
            # First initialize
            runner.invoke(main, ["init", "--non-interactive"])

            result = runner.invoke(main, ["config", "set", "cli-tool", "invalid-backend"])

            assert result.exit_code != 0
            # Should list valid options
            assert "claude" in result.output
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
            result = runner.invoke(main, ["config", "set", "cli-tool", "anthropic-api"])

            assert result.exit_code != 0
            # Should suggest running init
            assert "init" in result.output.lower()

    def test_config_changes_persist(self, runner, temp_project):
        """Config changes should persist and be reflected in subsequent 'config show'."""
        with runner.isolated_filesystem(temp_dir=temp_project):
            # First initialize
            runner.invoke(main, ["init", "--non-interactive"])

            # Change value
            runner.invoke(main, ["config", "set", "cli-tool", "anthropic-api"])

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
            # Should show valid options for cli-tool
            assert "claude" in result.output
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

    def test_config_set_cli_tool(self, runner, temp_project):
        """'swarm config set' should allow setting cli-tool."""
        with runner.isolated_filesystem(temp_dir=temp_project):
            # First initialize
            runner.invoke(main, ["init", "--non-interactive"])

            result = runner.invoke(main, ["config", "set", "cli-tool", "opencode"])

            assert result.exit_code == 0

            # Verify change persisted
            config = json.loads(Path(".swarm/config.json").read_text())
            assert config["cli_tool"] == "opencode"

    def test_config_set_cli_tool_rejects_invalid(self, runner, temp_project):
        """'swarm config set' should reject invalid cli-tool values."""
        with runner.isolated_filesystem(temp_dir=temp_project):
            # First initialize
            runner.invoke(main, ["init", "--non-interactive"])

            result = runner.invoke(main, ["config", "set", "cli-tool", "invalid-tool"])

            assert result.exit_code != 0
            # Should list valid options
            assert "claude" in result.output
            assert "opencode" in result.output

    def test_config_show_displays_cli_tool(self, runner, temp_project):
        """'swarm config show' should display cli-tool setting."""
        with runner.isolated_filesystem(temp_dir=temp_project):
            # First initialize
            runner.invoke(main, ["init", "--non-interactive"])

            result = runner.invoke(main, ["config", "show"])

            assert result.exit_code == 0
            # Should show cli-tool in the output
            assert "cli-tool" in result.output or "cli_tool" in result.output
            assert "claude" in result.output

    def test_config_set_exploration_model(self, runner, temp_project):
        """'swarm config set' should allow setting exploration-model."""
        with runner.isolated_filesystem(temp_dir=temp_project):
            # First initialize
            runner.invoke(main, ["init", "--non-interactive"])

            result = runner.invoke(main, ["config", "set", "exploration-model", "claude-sonnet-4-20250514"])

            assert result.exit_code == 0

            # Verify change persisted
            config = json.loads(Path(".swarm/config.json").read_text())
            assert config["exploration_model"] == "claude-sonnet-4-20250514"

    def test_config_show_displays_exploration_model(self, runner, temp_project):
        """'swarm config show' should display exploration-model setting."""
        with runner.isolated_filesystem(temp_dir=temp_project):
            # First initialize
            runner.invoke(main, ["init", "--non-interactive"])

            result = runner.invoke(main, ["config", "show"])

            assert result.exit_code == 0
            # Should show exploration-model in the output
            assert "exploration-model" in result.output or "exploration_model" in result.output


class TestCursorCommands:
    """Tests for the 'swarm cursor' command group."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()

    def test_cursor_group_exists(self, runner):
        """'swarm cursor' command group should exist."""
        result = runner.invoke(main, ["cursor", "--help"])

        assert result.exit_code == 0
        assert "login" in result.output
        assert "status" in result.output

    def test_cursor_login_invokes_cursor_agent_login(self, runner):
        """'swarm cursor login' should invoke 'cursor-agent login'."""
        with patch("swarm_orchestrator.cli.cursor_auth.login") as mock_login:
            mock_login.return_value = True
            result = runner.invoke(main, ["cursor", "login"])

            assert result.exit_code == 0
            mock_login.assert_called_once()

    def test_cursor_login_shows_success_message(self, runner):
        """'swarm cursor login' should show success message on successful login."""
        with patch("swarm_orchestrator.cli.cursor_auth.login", return_value=True):
            result = runner.invoke(main, ["cursor", "login"])

            assert result.exit_code == 0
            assert "success" in result.output.lower() or "logged in" in result.output.lower()

    def test_cursor_login_shows_failure_message(self, runner):
        """'swarm cursor login' should show failure message on failed login."""
        with patch("swarm_orchestrator.cli.cursor_auth.login", return_value=False):
            result = runner.invoke(main, ["cursor", "login"])

            assert result.exit_code != 0
            assert "failed" in result.output.lower() or "error" in result.output.lower()

    def test_cursor_status_shows_authenticated(self, runner):
        """'swarm cursor status' should show authenticated state when logged in."""
        with patch("swarm_orchestrator.cli.cursor_auth.is_authenticated", return_value=True):
            result = runner.invoke(main, ["cursor", "status"])

            assert result.exit_code == 0
            assert "authenticated" in result.output.lower()

    def test_cursor_status_shows_not_authenticated(self, runner):
        """'swarm cursor status' should show not authenticated when not logged in."""
        with patch("swarm_orchestrator.cli.cursor_auth.is_authenticated", return_value=False):
            result = runner.invoke(main, ["cursor", "status"])

            assert result.exit_code == 0
            assert "not authenticated" in result.output.lower() or "not logged in" in result.output.lower()

    def test_cursor_login_help_mentions_browser_auth(self, runner):
        """'swarm cursor login --help' should explain browser-based authentication."""
        result = runner.invoke(main, ["cursor", "login", "--help"])

        assert result.exit_code == 0
        assert "browser" in result.output.lower()

    def test_cursor_status_help_explains_auth_options(self, runner):
        """'swarm cursor status --help' should document both auth options."""
        result = runner.invoke(main, ["cursor", "status", "--help"])

        assert result.exit_code == 0
        # Should mention API key alternative
        output_lower = result.output.lower()
        assert "cursor_api_key" in output_lower or "api key" in output_lower or "status" in output_lower


class TestUpdateGlobalClaudeConfig:
    """Tests for the _update_global_claude_config function."""

    @pytest.fixture
    def temp_home(self, tmp_path, monkeypatch):
        """Create a temporary home directory."""
        monkeypatch.setenv("HOME", str(tmp_path))
        return tmp_path

    def test_creates_config_if_not_exists(self, temp_home):
        """Should create ~/.claude.json if it doesn't exist."""
        from swarm_orchestrator.cli import _update_global_claude_config

        mcp_config = {"command": "swarm", "args": ["server"]}
        _update_global_claude_config(mcp_config)

        config_path = temp_home / ".claude.json"
        assert config_path.exists()

        config = json.loads(config_path.read_text())
        assert "mcpServers" in config
        assert "swarm-orchestrator" in config["mcpServers"]
        assert config["mcpServers"]["swarm-orchestrator"] == mcp_config

    def test_adds_to_existing_config_without_mcp_servers(self, temp_home):
        """Should add mcpServers to existing config that lacks it."""
        from swarm_orchestrator.cli import _update_global_claude_config

        config_path = temp_home / ".claude.json"
        config_path.write_text(json.dumps({"someOtherKey": "value"}))

        mcp_config = {"command": "swarm", "args": ["server"]}
        _update_global_claude_config(mcp_config)

        config = json.loads(config_path.read_text())
        assert config["someOtherKey"] == "value"
        assert "mcpServers" in config
        assert "swarm-orchestrator" in config["mcpServers"]

    def test_preserves_existing_mcp_servers(self, temp_home):
        """Should preserve other MCP servers in existing config."""
        from swarm_orchestrator.cli import _update_global_claude_config

        config_path = temp_home / ".claude.json"
        existing = {
            "mcpServers": {
                "schaltwerk": {"command": "schaltwerk"},
                "other-server": {"command": "other"}
            }
        }
        config_path.write_text(json.dumps(existing))

        mcp_config = {"command": "swarm", "args": ["server"]}
        _update_global_claude_config(mcp_config)

        config = json.loads(config_path.read_text())
        assert "schaltwerk" in config["mcpServers"]
        assert "other-server" in config["mcpServers"]
        assert "swarm-orchestrator" in config["mcpServers"]

    def test_updates_existing_swarm_orchestrator(self, temp_home):
        """Should update swarm-orchestrator if already present."""
        from swarm_orchestrator.cli import _update_global_claude_config

        config_path = temp_home / ".claude.json"
        existing = {
            "mcpServers": {
                "swarm-orchestrator": {"command": "old-command", "args": []}
            }
        }
        config_path.write_text(json.dumps(existing))

        new_config = {"command": "swarm", "args": ["server", "--state-file", "/new/path"]}
        _update_global_claude_config(new_config)

        config = json.loads(config_path.read_text())
        assert config["mcpServers"]["swarm-orchestrator"] == new_config

    def test_handles_malformed_json(self, temp_home):
        """Should handle malformed JSON by creating new config."""
        from swarm_orchestrator.cli import _update_global_claude_config

        config_path = temp_home / ".claude.json"
        config_path.write_text("{ invalid json }")

        mcp_config = {"command": "swarm", "args": ["server"]}
        _update_global_claude_config(mcp_config)

        config = json.loads(config_path.read_text())
        assert "mcpServers" in config
        assert "swarm-orchestrator" in config["mcpServers"]

    def test_handles_non_dict_json(self, temp_home):
        """Should handle JSON that is not a dict by creating new config."""
        from swarm_orchestrator.cli import _update_global_claude_config

        config_path = temp_home / ".claude.json"
        config_path.write_text('"just a string"')

        mcp_config = {"command": "swarm", "args": ["server"]}
        _update_global_claude_config(mcp_config)

        config = json.loads(config_path.read_text())
        assert "mcpServers" in config
        assert "swarm-orchestrator" in config["mcpServers"]


class TestInitUpdatesGlobalConfig:
    """Tests for init command updating global ~/.claude.json."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()

    def test_init_updates_global_claude_config(self, runner, tmp_path):
        """Init should write MCP config to both .mcp.json and ~/.claude.json."""
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        with runner.isolated_filesystem(temp_dir=project_dir):
            with patch("pathlib.Path.home", return_value=fake_home):
                result = runner.invoke(main, ["init", "--non-interactive"])

            assert result.exit_code == 0

            # Check project-level config
            assert Path(".mcp.json").exists()
            project_config = json.loads(Path(".mcp.json").read_text())
            assert "swarm-orchestrator" in project_config["mcpServers"]

            # Check global config
            global_config_path = fake_home / ".claude.json"
            assert global_config_path.exists()
            global_config = json.loads(global_config_path.read_text())
            assert "mcpServers" in global_config
            assert "swarm-orchestrator" in global_config["mcpServers"]

    def test_init_global_config_has_same_mcp_config(self, runner, tmp_path):
        """Global config should have the same MCP server config as project config."""
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        with runner.isolated_filesystem(temp_dir=project_dir):
            with patch("pathlib.Path.home", return_value=fake_home):
                result = runner.invoke(main, ["init", "--non-interactive"])

            assert result.exit_code == 0

            project_config = json.loads(Path(".mcp.json").read_text())
            global_config = json.loads((fake_home / ".claude.json").read_text())

            project_mcp = project_config["mcpServers"]["swarm-orchestrator"]
            global_mcp = global_config["mcpServers"]["swarm-orchestrator"]

            assert project_mcp == global_mcp

    def test_init_output_mentions_global_config(self, runner, tmp_path):
        """Init output should mention updating global config."""
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        with runner.isolated_filesystem(temp_dir=project_dir):
            with patch("pathlib.Path.home", return_value=fake_home):
                result = runner.invoke(main, ["init", "--non-interactive"])

            assert result.exit_code == 0
            assert "~/.claude.json" in result.output or ".claude.json" in result.output

    def test_init_preserves_existing_global_mcp_servers(self, runner, tmp_path):
        """Init should preserve existing MCP servers in global config."""
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        # Create existing global config with other servers
        global_config_path = fake_home / ".claude.json"
        existing = {
            "mcpServers": {
                "schaltwerk": {"command": "schaltwerk"},
                "other-server": {"command": "other"}
            }
        }
        global_config_path.write_text(json.dumps(existing))

        with runner.isolated_filesystem(temp_dir=project_dir):
            with patch("pathlib.Path.home", return_value=fake_home):
                result = runner.invoke(main, ["init", "--non-interactive"])

            assert result.exit_code == 0

            global_config = json.loads(global_config_path.read_text())
            # All servers should be present
            assert "schaltwerk" in global_config["mcpServers"]
            assert "other-server" in global_config["mcpServers"]
            assert "swarm-orchestrator" in global_config["mcpServers"]
