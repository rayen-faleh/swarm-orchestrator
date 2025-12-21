"""
Tests for the configuration system.

Tests SwarmConfig dataclass, load_config, and save_config functions.
"""

import json
import pytest
from pathlib import Path

from swarm_orchestrator.config import (
    SwarmConfig,
    load_config,
    save_config,
    get_backend_choices,
    BACKENDS,
)


class TestSwarmConfig:
    """Tests for SwarmConfig dataclass."""

    def test_default_values(self):
        """Default config uses schaltwerk backends and claude cli_tool."""
        config = SwarmConfig()

        assert config.worktree_backend == "schaltwerk"
        assert config.agent_backend == "schaltwerk"
        assert config.cli_tool == "claude"
        assert config.llm_model == "claude-sonnet-4-20250514"
        assert config.llm_timeout == 120
        assert config.exploration_model == "claude-haiku-3-5"

    def test_custom_values(self):
        """Config accepts custom values."""
        config = SwarmConfig(
            worktree_backend="schaltwerk",
            agent_backend="schaltwerk",
            cli_tool="anthropic-api",
            llm_model="claude-opus-4-20250514",
            llm_timeout=300,
            exploration_model="claude-sonnet-4-20250514",
        )

        assert config.cli_tool == "anthropic-api"
        assert config.llm_model == "claude-opus-4-20250514"
        assert config.llm_timeout == 300
        assert config.exploration_model == "claude-sonnet-4-20250514"

    def test_invalid_worktree_backend(self):
        """Invalid worktree_backend raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            SwarmConfig(worktree_backend="invalid")

        assert "worktree_backend" in str(exc_info.value)
        assert "invalid" in str(exc_info.value)

    def test_invalid_agent_backend(self):
        """Invalid agent_backend raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            SwarmConfig(agent_backend="invalid")

        assert "agent_backend" in str(exc_info.value)

    def test_from_dict(self):
        """SwarmConfig.from_dict creates config from dictionary."""
        data = {
            "worktree_backend": "schaltwerk",
            "agent_backend": "schaltwerk",
            "cli_tool": "anthropic-api",
            "llm_model": "claude-opus-4-20250514",
            "llm_timeout": 180,
            "exploration_model": "claude-sonnet-4-20250514",
        }

        config = SwarmConfig.from_dict(data)

        assert config.cli_tool == "anthropic-api"
        assert config.llm_model == "claude-opus-4-20250514"
        assert config.llm_timeout == 180
        assert config.exploration_model == "claude-sonnet-4-20250514"

    def test_from_dict_with_defaults(self):
        """SwarmConfig.from_dict uses defaults for missing keys."""
        data = {"cli_tool": "anthropic-api"}

        config = SwarmConfig.from_dict(data)

        assert config.worktree_backend == "schaltwerk"
        assert config.agent_backend == "schaltwerk"
        assert config.cli_tool == "anthropic-api"
        assert config.llm_model == "claude-sonnet-4-20250514"
        assert config.exploration_model == "claude-haiku-3-5"

    def test_to_dict(self):
        """SwarmConfig.to_dict returns dictionary representation."""
        config = SwarmConfig(cli_tool="anthropic-api", llm_timeout=200)

        data = config.to_dict()

        assert data["worktree_backend"] == "schaltwerk"
        assert data["agent_backend"] == "schaltwerk"
        assert data["cli_tool"] == "anthropic-api"
        assert data["llm_timeout"] == 200
        assert data["exploration_model"] == "claude-haiku-3-5"


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_from_default_path(self, tmp_path, monkeypatch):
        """load_config loads from .swarm/config.json by default."""
        monkeypatch.chdir(tmp_path)
        config_dir = tmp_path / ".swarm"
        config_dir.mkdir()
        config_file = config_dir / "config.json"
        config_file.write_text(json.dumps({"cli_tool": "anthropic-api"}))

        config = load_config()

        assert config.cli_tool == "anthropic-api"

    def test_load_from_custom_path(self, tmp_path):
        """load_config loads from specified path."""
        config_file = tmp_path / "custom-config.json"
        config_file.write_text(json.dumps({"llm_timeout": 300}))

        config = load_config(config_file)

        assert config.llm_timeout == 300

    def test_returns_defaults_if_file_missing(self, tmp_path, monkeypatch):
        """load_config returns defaults if config file doesn't exist."""
        monkeypatch.chdir(tmp_path)

        config = load_config()

        assert config.worktree_backend == "schaltwerk"
        assert config.agent_backend == "schaltwerk"
        assert config.cli_tool == "claude"

    def test_invalid_json_raises_error(self, tmp_path):
        """load_config raises ValueError for invalid JSON."""
        config_file = tmp_path / "config.json"
        config_file.write_text("not valid json")

        with pytest.raises(ValueError) as exc_info:
            load_config(config_file)

        assert "Invalid config file" in str(exc_info.value)


class TestSaveConfig:
    """Tests for save_config function."""

    def test_save_to_default_path(self, tmp_path, monkeypatch):
        """save_config saves to .swarm/config.json by default."""
        monkeypatch.chdir(tmp_path)
        config = SwarmConfig(cli_tool="anthropic-api")

        save_config(config)

        config_file = tmp_path / ".swarm" / "config.json"
        assert config_file.exists()
        data = json.loads(config_file.read_text())
        assert data["cli_tool"] == "anthropic-api"

    def test_save_to_custom_path(self, tmp_path):
        """save_config saves to specified path."""
        config = SwarmConfig(llm_timeout=200)
        config_file = tmp_path / "custom" / "config.json"

        save_config(config, config_file)

        assert config_file.exists()
        data = json.loads(config_file.read_text())
        assert data["llm_timeout"] == 200

    def test_save_creates_parent_directories(self, tmp_path):
        """save_config creates parent directories if needed."""
        config = SwarmConfig()
        config_file = tmp_path / "deep" / "nested" / "config.json"

        save_config(config, config_file)

        assert config_file.exists()


class TestConfigIntegration:
    """Integration tests for config system."""

    def test_save_and_load_roundtrip(self, tmp_path):
        """Config can be saved and loaded back correctly."""
        original = SwarmConfig(
            worktree_backend="schaltwerk",
            agent_backend="schaltwerk",
            cli_tool="opencode",
            llm_model="claude-opus-4-20250514",
            llm_timeout=250,
            exploration_model="claude-sonnet-4-20250514",
        )
        config_file = tmp_path / "config.json"

        save_config(original, config_file)
        loaded = load_config(config_file)

        assert loaded.worktree_backend == original.worktree_backend
        assert loaded.agent_backend == original.agent_backend
        assert loaded.cli_tool == original.cli_tool
        assert loaded.llm_model == original.llm_model
        assert loaded.llm_timeout == original.llm_timeout
        assert loaded.exploration_model == original.exploration_model


class TestBackendRegistry:
    """Tests for the centralized backend registry."""

    def test_backends_has_all_types(self):
        """BACKENDS should have worktree, agent, and cli_tool entries."""
        assert "worktree" in BACKENDS
        assert "agent" in BACKENDS
        assert "cli_tool" in BACKENDS

    def test_get_backend_choices_worktree(self):
        """get_backend_choices returns worktree options."""
        choices = get_backend_choices("worktree")
        assert "schaltwerk" in choices

    def test_get_backend_choices_agent(self):
        """get_backend_choices returns agent options."""
        choices = get_backend_choices("agent")
        assert "schaltwerk" in choices

    def test_get_backend_choices_agent_includes_cursor_cli(self):
        """get_backend_choices('agent') includes 'cursor-cli'."""
        choices = get_backend_choices("agent")
        assert "cursor-cli" in choices

    def test_get_backend_choices_unknown(self):
        """get_backend_choices returns empty list for unknown type."""
        choices = get_backend_choices("unknown")
        assert choices == []

    def test_backend_descriptions_exist(self):
        """Each backend option should have a description."""
        for backend_type, options in BACKENDS.items():
            for name, desc in options.items():
                assert desc, f"Missing description for {backend_type}/{name}"
                assert len(desc) > 10, f"Description too short for {backend_type}/{name}"


class TestCursorCLIBackendConfig:
    """Tests for cursor-cli backend registration."""

    def test_backends_agent_contains_cursor_cli(self):
        """BACKENDS['agent'] contains 'cursor-cli' entry."""
        assert "cursor-cli" in BACKENDS["agent"]

    def test_cursor_cli_has_description(self):
        """cursor-cli has a descriptive string."""
        desc = BACKENDS["agent"]["cursor-cli"]
        assert isinstance(desc, str)
        assert len(desc) > 0

    def test_swarm_config_validates_cursor_cli(self):
        """SwarmConfig(agent_backend='cursor-cli') passes validation."""
        config = SwarmConfig(agent_backend="cursor-cli")
        assert config.agent_backend == "cursor-cli"

    def test_cursor_cli_backend_import_works(self):
        """CursorCLIAgentBackend can be imported from backends package."""
        from swarm_orchestrator.backends import CursorCLIAgentBackend
        assert CursorCLIAgentBackend is not None


class TestGitNativeAgentBackendConfig:
    """Tests for git-native agent backend registration."""

    def test_backends_agent_contains_git_native(self):
        """BACKENDS['agent'] contains 'git-native' entry."""
        assert "git-native" in BACKENDS["agent"]

    def test_git_native_has_description(self):
        """git-native has a descriptive string."""
        desc = BACKENDS["agent"]["git-native"]
        assert isinstance(desc, str)
        assert len(desc) > 0

    def test_swarm_config_validates_git_native(self):
        """SwarmConfig(agent_backend='git-native') passes validation."""
        config = SwarmConfig(agent_backend="git-native")
        assert config.agent_backend == "git-native"

    def test_get_backend_choices_agent_includes_git_native(self):
        """get_backend_choices('agent') includes 'git-native'."""
        choices = get_backend_choices("agent")
        assert "git-native" in choices


class TestCLIToolConfig:
    """Tests for cli_tool configuration option (unified CLI abstraction)."""

    def test_backends_contains_cli_tool(self):
        """BACKENDS contains 'cli_tool' entry."""
        assert "cli_tool" in BACKENDS

    def test_cli_tool_has_all_options(self):
        """BACKENDS['cli_tool'] contains all unified CLI options."""
        assert "claude" in BACKENDS["cli_tool"]
        assert "opencode" in BACKENDS["cli_tool"]
        assert "cursor" in BACKENDS["cli_tool"]
        assert "anthropic-api" in BACKENDS["cli_tool"]

    def test_cli_tool_options_have_descriptions(self):
        """Each cli_tool option has a description."""
        for name, desc in BACKENDS["cli_tool"].items():
            assert isinstance(desc, str)
            assert len(desc) > 10

    def test_get_backend_choices_cli_tool(self):
        """get_backend_choices returns cli_tool options."""
        choices = get_backend_choices("cli_tool")
        assert "claude" in choices
        assert "opencode" in choices
        assert "cursor" in choices
        assert "anthropic-api" in choices

    def test_default_cli_tool_is_claude(self):
        """Default cli_tool is 'claude'."""
        config = SwarmConfig()
        assert config.cli_tool == "claude"

    def test_cli_tool_accepts_claude(self):
        """SwarmConfig accepts cli_tool='claude'."""
        config = SwarmConfig(cli_tool="claude")
        assert config.cli_tool == "claude"

    def test_cli_tool_accepts_opencode(self):
        """SwarmConfig accepts cli_tool='opencode'."""
        config = SwarmConfig(cli_tool="opencode")
        assert config.cli_tool == "opencode"

    def test_cli_tool_accepts_cursor(self):
        """SwarmConfig accepts cli_tool='cursor'."""
        config = SwarmConfig(cli_tool="cursor")
        assert config.cli_tool == "cursor"

    def test_cli_tool_accepts_anthropic_api(self):
        """SwarmConfig accepts cli_tool='anthropic-api'."""
        config = SwarmConfig(cli_tool="anthropic-api")
        assert config.cli_tool == "anthropic-api"

    def test_invalid_cli_tool_raises_error(self):
        """Invalid cli_tool raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            SwarmConfig(cli_tool="invalid")

        assert "cli_tool" in str(exc_info.value)
        assert "invalid" in str(exc_info.value)

    def test_from_dict_with_cli_tool(self):
        """from_dict correctly loads cli_tool."""
        data = {"cli_tool": "opencode"}
        config = SwarmConfig.from_dict(data)
        assert config.cli_tool == "opencode"

    def test_from_dict_defaults_cli_tool_to_claude(self):
        """from_dict defaults cli_tool to 'claude' when not specified."""
        data = {}
        config = SwarmConfig.from_dict(data)
        assert config.cli_tool == "claude"

    def test_to_dict_includes_cli_tool(self):
        """to_dict includes cli_tool field."""
        config = SwarmConfig(cli_tool="opencode")
        data = config.to_dict()
        assert "cli_tool" in data
        assert data["cli_tool"] == "opencode"

    def test_cli_tool_roundtrip(self, tmp_path):
        """cli_tool survives save/load roundtrip."""
        original = SwarmConfig(cli_tool="opencode")
        config_file = tmp_path / "config.json"

        save_config(original, config_file)
        loaded = load_config(config_file)

        assert loaded.cli_tool == "opencode"


class TestBackwardsCompatMigration:
    """Tests for backwards-compatible migration of llm_backend to cli_tool."""

    def test_llm_backend_claude_cli_migrates_to_claude(self):
        """Old llm_backend='claude-cli' migrates to cli_tool='claude'."""
        data = {"llm_backend": "claude-cli"}
        config = SwarmConfig.from_dict(data)
        assert config.cli_tool == "claude"

    def test_llm_backend_anthropic_api_migrates(self):
        """Old llm_backend='anthropic-api' migrates to cli_tool='anthropic-api'."""
        data = {"llm_backend": "anthropic-api"}
        config = SwarmConfig.from_dict(data)
        assert config.cli_tool == "anthropic-api"

    def test_cli_tool_takes_precedence_over_llm_backend(self):
        """If both are specified, cli_tool takes precedence."""
        data = {"llm_backend": "anthropic-api", "cli_tool": "opencode"}
        config = SwarmConfig.from_dict(data)
        assert config.cli_tool == "opencode"

    def test_old_config_file_still_works(self, tmp_path):
        """Config files with old llm_backend field still load correctly."""
        config_file = tmp_path / "config.json"
        config_file.write_text('{"llm_backend": "anthropic-api", "llm_model": "claude-opus-4-20250514"}')

        config = load_config(config_file)

        assert config.cli_tool == "anthropic-api"
        assert config.llm_model == "claude-opus-4-20250514"

    def test_to_dict_does_not_include_llm_backend(self):
        """to_dict no longer includes llm_backend field."""
        config = SwarmConfig(cli_tool="anthropic-api")
        data = config.to_dict()
        assert "llm_backend" not in data
        assert data["cli_tool"] == "anthropic-api"


class TestConfigToCommandIntegration:
    """End-to-end tests: config file → GitNativeAgentBackend → command string."""

    def test_config_file_to_claude_command(self, tmp_path):
        """Config file with cli_tool='claude' generates correct claude command."""
        from swarm_orchestrator.backends.git_native import GitNativeAgentBackend

        config_file = tmp_path / "config.json"
        config_file.write_text('{"cli_tool": "claude", "agent_backend": "git-native"}')

        config = load_config(config_file)
        backend = GitNativeAgentBackend(cli_tool=config.cli_tool)
        cmd = backend._generate_command("$PROMPT_FILE")

        assert "claude" in cmd
        assert "--dangerously-skip-permissions" in cmd
        assert "opencode" not in cmd

    def test_config_file_to_opencode_command(self, tmp_path):
        """Config file with cli_tool='opencode' generates correct opencode command."""
        from swarm_orchestrator.backends.git_native import GitNativeAgentBackend

        config_file = tmp_path / "config.json"
        config_file.write_text('{"cli_tool": "opencode", "agent_backend": "git-native"}')

        config = load_config(config_file)
        backend = GitNativeAgentBackend(cli_tool=config.cli_tool)
        cmd = backend._generate_command("$PROMPT_FILE")

        assert "opencode -p" in cmd
        assert "claude" not in cmd
        assert "--dangerously-skip-permissions" not in cmd

    def test_missing_cli_tool_defaults_to_claude(self, tmp_path):
        """Config file without cli_tool defaults to claude command."""
        from swarm_orchestrator.backends.git_native import GitNativeAgentBackend

        config_file = tmp_path / "config.json"
        config_file.write_text('{"agent_backend": "git-native"}')

        config = load_config(config_file)
        backend = GitNativeAgentBackend(cli_tool=config.cli_tool)
        cmd = backend._generate_command("$PROMPT_FILE")

        assert config.cli_tool == "claude"
        assert "claude" in cmd
        assert "--dangerously-skip-permissions" in cmd


class TestCompressionConfig:
    """Tests for compression configuration fields."""

    def test_default_compression_values(self):
        """Default compression settings are sensible."""
        config = SwarmConfig()
        assert config.enable_diff_compression is True
        assert config.compression_min_tokens == 500
        assert config.compression_target_ratio == 0.3

    def test_custom_compression_values(self):
        """Compression settings can be customized."""
        config = SwarmConfig(
            enable_diff_compression=False,
            compression_min_tokens=1000,
            compression_target_ratio=0.5,
        )
        assert config.enable_diff_compression is False
        assert config.compression_min_tokens == 1000
        assert config.compression_target_ratio == 0.5

    def test_compression_ratio_too_low_raises_error(self):
        """compression_target_ratio below 0.1 raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            SwarmConfig(compression_target_ratio=0.05)
        assert "compression_target_ratio" in str(exc_info.value)

    def test_compression_ratio_too_high_raises_error(self):
        """compression_target_ratio above 1.0 raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            SwarmConfig(compression_target_ratio=1.5)
        assert "compression_target_ratio" in str(exc_info.value)

    def test_compression_ratio_at_bounds_is_valid(self):
        """compression_target_ratio at 0.1 and 1.0 are valid."""
        config_low = SwarmConfig(compression_target_ratio=0.1)
        assert config_low.compression_target_ratio == 0.1

        config_high = SwarmConfig(compression_target_ratio=1.0)
        assert config_high.compression_target_ratio == 1.0

    def test_from_dict_loads_compression_settings(self):
        """from_dict correctly loads compression settings."""
        data = {
            "enable_diff_compression": False,
            "compression_min_tokens": 200,
            "compression_target_ratio": 0.4,
        }
        config = SwarmConfig.from_dict(data)
        assert config.enable_diff_compression is False
        assert config.compression_min_tokens == 200
        assert config.compression_target_ratio == 0.4

    def test_from_dict_defaults_compression_settings(self):
        """from_dict uses defaults when compression settings not specified."""
        data = {}
        config = SwarmConfig.from_dict(data)
        assert config.enable_diff_compression is True
        assert config.compression_min_tokens == 500
        assert config.compression_target_ratio == 0.3

    def test_to_dict_includes_compression_settings(self):
        """to_dict includes all compression fields."""
        config = SwarmConfig(
            enable_diff_compression=False,
            compression_min_tokens=750,
            compression_target_ratio=0.6,
        )
        data = config.to_dict()
        assert data["enable_diff_compression"] is False
        assert data["compression_min_tokens"] == 750
        assert data["compression_target_ratio"] == 0.6

    def test_compression_settings_roundtrip(self, tmp_path):
        """Compression settings survive save/load roundtrip."""
        original = SwarmConfig(
            enable_diff_compression=False,
            compression_min_tokens=800,
            compression_target_ratio=0.7,
        )
        config_file = tmp_path / "config.json"

        save_config(original, config_file)
        loaded = load_config(config_file)

        assert loaded.enable_diff_compression == original.enable_diff_compression
        assert loaded.compression_min_tokens == original.compression_min_tokens
        assert loaded.compression_target_ratio == original.compression_target_ratio


class TestExplorationModelConfig:
    """Tests for exploration_model configuration field."""

    def test_default_exploration_model(self):
        """Default exploration_model is claude-haiku-3-5."""
        config = SwarmConfig()
        assert config.exploration_model == "claude-haiku-3-5"

    def test_custom_exploration_model(self):
        """exploration_model can be set to any model string."""
        config = SwarmConfig(exploration_model="claude-sonnet-4-20250514")
        assert config.exploration_model == "claude-sonnet-4-20250514"

    def test_from_dict_loads_exploration_model(self):
        """from_dict correctly loads exploration_model from dict."""
        data = {"exploration_model": "claude-opus-4-20250514"}
        config = SwarmConfig.from_dict(data)
        assert config.exploration_model == "claude-opus-4-20250514"

    def test_from_dict_defaults_exploration_model(self):
        """from_dict uses default when exploration_model not specified."""
        data = {}
        config = SwarmConfig.from_dict(data)
        assert config.exploration_model == "claude-haiku-3-5"

    def test_to_dict_includes_exploration_model(self):
        """to_dict includes exploration_model field."""
        config = SwarmConfig(exploration_model="claude-opus-4-20250514")
        data = config.to_dict()
        assert "exploration_model" in data
        assert data["exploration_model"] == "claude-opus-4-20250514"

    def test_exploration_model_roundtrip(self, tmp_path):
        """exploration_model survives save/load roundtrip."""
        original = SwarmConfig(exploration_model="custom-model-123")
        config_file = tmp_path / "config.json"

        save_config(original, config_file)
        loaded = load_config(config_file)

        assert loaded.exploration_model == "custom-model-123"
