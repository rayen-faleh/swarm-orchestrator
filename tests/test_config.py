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
        """Default config uses schaltwerk and claude-cli backends."""
        config = SwarmConfig()

        assert config.worktree_backend == "schaltwerk"
        assert config.agent_backend == "schaltwerk"
        assert config.llm_backend == "claude-cli"
        assert config.llm_model == "claude-sonnet-4-20250514"
        assert config.llm_timeout == 120

    def test_custom_values(self):
        """Config accepts custom values."""
        config = SwarmConfig(
            worktree_backend="schaltwerk",
            agent_backend="schaltwerk",
            llm_backend="anthropic-api",
            llm_model="claude-opus-4-20250514",
            llm_timeout=300,
        )

        assert config.llm_backend == "anthropic-api"
        assert config.llm_model == "claude-opus-4-20250514"
        assert config.llm_timeout == 300

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

    def test_invalid_llm_backend(self):
        """Invalid llm_backend raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            SwarmConfig(llm_backend="invalid")

        assert "llm_backend" in str(exc_info.value)

    def test_from_dict(self):
        """SwarmConfig.from_dict creates config from dictionary."""
        data = {
            "worktree_backend": "schaltwerk",
            "agent_backend": "schaltwerk",
            "llm_backend": "anthropic-api",
            "llm_model": "claude-opus-4-20250514",
            "llm_timeout": 180,
        }

        config = SwarmConfig.from_dict(data)

        assert config.llm_backend == "anthropic-api"
        assert config.llm_model == "claude-opus-4-20250514"
        assert config.llm_timeout == 180

    def test_from_dict_with_defaults(self):
        """SwarmConfig.from_dict uses defaults for missing keys."""
        data = {"llm_backend": "anthropic-api"}

        config = SwarmConfig.from_dict(data)

        assert config.worktree_backend == "schaltwerk"
        assert config.agent_backend == "schaltwerk"
        assert config.llm_backend == "anthropic-api"
        assert config.llm_model == "claude-sonnet-4-20250514"

    def test_to_dict(self):
        """SwarmConfig.to_dict returns dictionary representation."""
        config = SwarmConfig(llm_backend="anthropic-api", llm_timeout=200)

        data = config.to_dict()

        assert data["worktree_backend"] == "schaltwerk"
        assert data["agent_backend"] == "schaltwerk"
        assert data["llm_backend"] == "anthropic-api"
        assert data["llm_timeout"] == 200


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_from_default_path(self, tmp_path, monkeypatch):
        """load_config loads from .swarm/config.json by default."""
        monkeypatch.chdir(tmp_path)
        config_dir = tmp_path / ".swarm"
        config_dir.mkdir()
        config_file = config_dir / "config.json"
        config_file.write_text(json.dumps({"llm_backend": "anthropic-api"}))

        config = load_config()

        assert config.llm_backend == "anthropic-api"

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
        assert config.llm_backend == "claude-cli"

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
        config = SwarmConfig(llm_backend="anthropic-api")

        save_config(config)

        config_file = tmp_path / ".swarm" / "config.json"
        assert config_file.exists()
        data = json.loads(config_file.read_text())
        assert data["llm_backend"] == "anthropic-api"

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
            llm_backend="anthropic-api",
            llm_model="claude-opus-4-20250514",
            llm_timeout=250,
        )
        config_file = tmp_path / "config.json"

        save_config(original, config_file)
        loaded = load_config(config_file)

        assert loaded.worktree_backend == original.worktree_backend
        assert loaded.agent_backend == original.agent_backend
        assert loaded.llm_backend == original.llm_backend
        assert loaded.llm_model == original.llm_model
        assert loaded.llm_timeout == original.llm_timeout


class TestBackendRegistry:
    """Tests for the centralized backend registry."""

    def test_backends_has_all_types(self):
        """BACKENDS should have worktree, agent, and llm entries."""
        assert "worktree" in BACKENDS
        assert "agent" in BACKENDS
        assert "llm" in BACKENDS

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

    def test_get_backend_choices_llm(self):
        """get_backend_choices returns LLM options."""
        choices = get_backend_choices("llm")
        assert "claude-cli" in choices
        assert "anthropic-api" in choices

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
