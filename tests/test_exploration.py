"""
Tests for the exploration module.

TDD approach: These tests define the expected behavior of the ExplorationExecutor.
"""

import pytest
from unittest.mock import MagicMock, patch

from swarm_orchestrator.exploration import (
    ExplorationExecutor,
    needs_exploration,
)
from swarm_orchestrator.decomposer import (
    CodeInsight,
    WebResearchFinding,
    ExplorationResult,
)


class TestExplorationExecutor:
    """Tests for ExplorationExecutor class."""

    @pytest.fixture
    def mock_subprocess(self):
        """Create a mock subprocess.run."""
        with patch("swarm_orchestrator.exploration.subprocess.run") as mock:
            yield mock

    def test_explore_returns_exploration_result(self, mock_subprocess):
        """Should return a populated ExplorationResult."""
        mock_subprocess.return_value = MagicMock(
            returncode=0,
            stdout="""{
                "needs_exploration": true,
                "code_insights": [
                    {
                        "file_path": "src/auth.py",
                        "description": "Authentication module",
                        "patterns": ["jwt", "middleware"],
                        "dependencies": ["pyjwt"]
                    }
                ],
                "web_findings": [],
                "context_summary": "The codebase uses JWT for authentication"
            }""",
            stderr="",
        )

        executor = ExplorationExecutor()
        result = executor.explore("Add user authentication")

        assert isinstance(result, ExplorationResult)
        assert len(result.code_insights) == 1
        assert result.code_insights[0].file_path == "src/auth.py"
        assert result.context_summary == "The codebase uses JWT for authentication"

    def test_explore_with_web_research(self, mock_subprocess):
        """Should include web research findings when relevant."""
        mock_subprocess.return_value = MagicMock(
            returncode=0,
            stdout="""{
                "needs_exploration": true,
                "code_insights": [],
                "web_findings": [
                    {
                        "source": "https://docs.example.com/api",
                        "summary": "API documentation for OAuth2",
                        "relevance": "Directly applicable for auth implementation"
                    }
                ],
                "context_summary": "OAuth2 is recommended for third-party auth"
            }""",
            stderr="",
        )

        executor = ExplorationExecutor(enable_web_research=True)
        result = executor.explore("Implement OAuth2 login")

        assert len(result.web_findings) == 1
        assert "OAuth2" in result.web_findings[0].summary

    def test_explore_detects_simple_task(self, mock_subprocess):
        """Should detect when exploration is not needed."""
        mock_subprocess.return_value = MagicMock(
            returncode=0,
            stdout="""{
                "needs_exploration": false,
                "code_insights": [],
                "web_findings": [],
                "context_summary": "Simple task - no exploration needed"
            }""",
            stderr="",
        )

        executor = ExplorationExecutor()
        result = executor.explore("Fix typo in README")

        assert result.context_summary == "Simple task - no exploration needed"
        assert len(result.code_insights) == 0

    def test_explore_with_multiple_insights(self, mock_subprocess):
        """Should handle multiple code insights."""
        mock_subprocess.return_value = MagicMock(
            returncode=0,
            stdout="""{
                "needs_exploration": true,
                "code_insights": [
                    {"file_path": "src/models/user.py", "description": "User model", "patterns": [], "dependencies": []},
                    {"file_path": "src/api/routes.py", "description": "API routes", "patterns": ["REST"], "dependencies": ["fastapi"]},
                    {"file_path": "tests/test_api.py", "description": "API tests", "patterns": [], "dependencies": ["pytest"]}
                ],
                "web_findings": [],
                "context_summary": "REST API with user model and tests"
            }""",
            stderr="",
        )

        executor = ExplorationExecutor()
        result = executor.explore("Add user profile endpoint")

        assert len(result.code_insights) == 3
        assert result.code_insights[1].patterns == ["REST"]
        assert result.code_insights[1].dependencies == ["fastapi"]

    def test_explore_handles_cli_error(self, mock_subprocess):
        """Should handle CLI errors gracefully."""
        mock_subprocess.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="Error: something went wrong",
        )

        executor = ExplorationExecutor()
        result = executor.explore("Some task")

        # Should return empty result rather than crash
        assert isinstance(result, ExplorationResult)
        assert len(result.code_insights) == 0
        assert result.context_summary == ""

    def test_explore_handles_invalid_json(self, mock_subprocess):
        """Should handle invalid JSON responses gracefully."""
        mock_subprocess.return_value = MagicMock(
            returncode=0,
            stdout="Not valid JSON response",
            stderr="",
        )

        executor = ExplorationExecutor()
        result = executor.explore("Some task")

        # Should return empty result rather than crash
        assert isinstance(result, ExplorationResult)
        assert len(result.code_insights) == 0

    def test_explore_uses_custom_timeout(self, mock_subprocess):
        """Should respect custom timeout setting."""
        mock_subprocess.return_value = MagicMock(
            returncode=0,
            stdout='{"needs_exploration": false, "code_insights": [], "web_findings": [], "context_summary": ""}',
            stderr="",
        )

        executor = ExplorationExecutor(timeout=60)
        executor.explore("Quick task")

        # Verify subprocess was called with correct timeout
        call_args = mock_subprocess.call_args
        assert call_args.kwargs.get("timeout") == 60

    def test_explore_handles_timeout(self, mock_subprocess):
        """Should handle timeout gracefully."""
        import subprocess
        mock_subprocess.side_effect = subprocess.TimeoutExpired(cmd="claude", timeout=120)

        executor = ExplorationExecutor()
        result = executor.explore("Complex task")

        # Should return empty result rather than crash
        assert isinstance(result, ExplorationResult)
        assert len(result.code_insights) == 0

    def test_explore_handles_cli_not_found(self, mock_subprocess):
        """Should handle missing CLI gracefully."""
        mock_subprocess.side_effect = FileNotFoundError()

        executor = ExplorationExecutor()
        result = executor.explore("Some task")

        # Should return empty result rather than crash
        assert isinstance(result, ExplorationResult)

    def test_explore_with_working_directory(self, mock_subprocess):
        """Should pass working directory to CLI."""
        mock_subprocess.return_value = MagicMock(
            returncode=0,
            stdout='{"needs_exploration": false, "code_insights": [], "web_findings": [], "context_summary": ""}',
            stderr="",
        )

        executor = ExplorationExecutor(working_dir="/path/to/project")
        executor.explore("Task")

        call_args = mock_subprocess.call_args
        assert call_args.kwargs.get("cwd") == "/path/to/project"


class TestNeedsExploration:
    """Tests for needs_exploration detection function."""

    def test_simple_task_no_exploration(self):
        """Simple tasks should not need exploration."""
        assert needs_exploration("Fix typo in README") is False
        assert needs_exploration("Update version number") is False
        assert needs_exploration("Add comment to function") is False

    def test_complex_task_needs_exploration(self):
        """Complex tasks should need exploration."""
        assert needs_exploration("Implement user authentication with OAuth2") is True
        assert needs_exploration("Add real-time notifications using WebSockets") is True
        assert needs_exploration("Refactor database layer to use async queries") is True

    def test_api_integration_needs_exploration(self):
        """API integration tasks should need exploration."""
        assert needs_exploration("Integrate with Stripe payment API") is True
        assert needs_exploration("Add GraphQL endpoint for user data") is True

    def test_new_feature_needs_exploration(self):
        """New feature implementation should need exploration."""
        assert needs_exploration("Add caching layer for API responses") is True
        assert needs_exploration("Implement rate limiting") is True

    def test_bug_fix_mixed(self):
        """Bug fixes may or may not need exploration."""
        # Simple bug fix
        assert needs_exploration("Fix null pointer in login") is False
        # Complex bug that needs investigation
        assert needs_exploration("Debug intermittent authentication failures in distributed system") is True


class TestExplorationExecutorInit:
    """Tests for ExplorationExecutor initialization."""

    def test_default_initialization(self):
        """Should initialize with sensible defaults."""
        executor = ExplorationExecutor()

        assert executor.timeout == 120
        assert executor.enable_web_research is False
        assert executor.working_dir is None
        assert executor.model is None

    def test_custom_initialization(self):
        """Should accept custom configuration."""
        executor = ExplorationExecutor(
            timeout=60,
            enable_web_research=True,
            working_dir="/custom/path",
        )

        assert executor.timeout == 60
        assert executor.enable_web_research is True
        assert executor.working_dir == "/custom/path"

    def test_initialization_with_model(self):
        """Should accept model parameter."""
        executor = ExplorationExecutor(model="claude-3-haiku")

        assert executor.model == "claude-3-haiku"


class TestExplorationExecutorModel:
    """Tests for ExplorationExecutor model parameter."""

    @pytest.fixture
    def mock_subprocess(self):
        """Create a mock subprocess.run."""
        with patch("swarm_orchestrator.exploration.subprocess.run") as mock:
            yield mock

    def test_cli_command_includes_model_flag(self, mock_subprocess):
        """Should include --model flag when model is specified."""
        mock_subprocess.return_value = MagicMock(
            returncode=0,
            stdout='{"needs_exploration": false, "code_insights": [], "web_findings": [], "context_summary": ""}',
            stderr="",
        )

        executor = ExplorationExecutor(model="claude-3-haiku")
        executor.explore("Some task")

        call_args = mock_subprocess.call_args
        cmd = call_args[0][0]
        assert "--model" in cmd
        assert "claude-3-haiku" in cmd

    def test_cli_command_without_model_flag(self, mock_subprocess):
        """Should not include --model flag when model is None."""
        mock_subprocess.return_value = MagicMock(
            returncode=0,
            stdout='{"needs_exploration": false, "code_insights": [], "web_findings": [], "context_summary": ""}',
            stderr="",
        )

        executor = ExplorationExecutor()
        executor.explore("Some task")

        call_args = mock_subprocess.call_args
        cmd = call_args[0][0]
        assert "--model" not in cmd

    def test_model_flag_position_in_command(self, mock_subprocess):
        """Should add model flag after base command."""
        mock_subprocess.return_value = MagicMock(
            returncode=0,
            stdout='{"needs_exploration": false, "code_insights": [], "web_findings": [], "context_summary": ""}',
            stderr="",
        )

        executor = ExplorationExecutor(model="claude-3-sonnet")
        executor.explore("Task")

        call_args = mock_subprocess.call_args
        cmd = call_args[0][0]
        model_index = cmd.index("--model")
        assert cmd[model_index + 1] == "claude-3-sonnet"


class TestExplorationResultStructure:
    """Tests for ExplorationResult structure returned by executor."""

    @pytest.fixture
    def mock_subprocess(self):
        """Create a mock subprocess.run."""
        with patch("swarm_orchestrator.exploration.subprocess.run") as mock:
            yield mock

    def test_result_serializable(self, mock_subprocess):
        """Result should be serializable for agent prompts."""
        mock_subprocess.return_value = MagicMock(
            returncode=0,
            stdout="""{
                "needs_exploration": true,
                "code_insights": [
                    {"file_path": "src/main.py", "description": "Entry point", "patterns": ["singleton"], "dependencies": ["click"]}
                ],
                "web_findings": [
                    {"source": "https://click.palletsprojects.com", "summary": "CLI framework", "relevance": "High"}
                ],
                "context_summary": "CLI application using Click"
            }""",
            stderr="",
        )

        executor = ExplorationExecutor(enable_web_research=True)
        result = executor.explore("Add new CLI command")

        # Result should be convertible to dict
        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "code_insights" in result_dict
        assert "web_findings" in result_dict
        assert "context_summary" in result_dict
        assert result_dict["code_insights"][0]["file_path"] == "src/main.py"
