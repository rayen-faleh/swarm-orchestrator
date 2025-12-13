"""
Tests for the decomposer module.

TDD approach: These tests define the expected behavior of task decomposition.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from swarm_orchestrator.decomposer import (
    Decomposer,
    DecomposerError,
    Subtask,
    DecompositionResult,
    decompose_task,
    DECOMPOSE_PROMPT,
)


class TestSubtask:
    """Tests for Subtask dataclass."""

    def test_create_subtask(self):
        """Should create a subtask with all fields."""
        subtask = Subtask(
            id="add-auth",
            description="Add authentication",
            prompt="Implement JWT authentication for the API",
        )

        assert subtask.id == "add-auth"
        assert subtask.description == "Add authentication"
        assert "JWT" in subtask.prompt


class TestDecompositionResult:
    """Tests for DecompositionResult dataclass."""

    def test_atomic_result(self):
        """Should represent an atomic task."""
        result = DecompositionResult(
            is_atomic=True,
            subtasks=[Subtask(id="task-1", description="Do thing", prompt="Do the thing")],
            original_query="Do the thing",
        )

        assert result.is_atomic is True
        assert len(result.subtasks) == 1

    def test_complex_result(self):
        """Should represent a complex task with multiple subtasks."""
        result = DecompositionResult(
            is_atomic=False,
            subtasks=[
                Subtask(id="task-1", description="First", prompt="Do first"),
                Subtask(id="task-2", description="Second", prompt="Do second"),
            ],
            original_query="Do multiple things",
        )

        assert result.is_atomic is False
        assert len(result.subtasks) == 2


class TestDecomposer:
    """Tests for Decomposer class."""

    @pytest.fixture
    def mock_subprocess(self):
        """Create a mock subprocess.run."""
        with patch("swarm_orchestrator.decomposer.subprocess.run") as mock:
            yield mock

    @pytest.fixture
    def mock_anthropic(self):
        """Create a mock Anthropic client."""
        with patch("swarm_orchestrator.decomposer.Anthropic") as mock:
            yield mock

    def test_decompose_atomic_task(self, mock_subprocess):
        """Should recognize atomic tasks and return single subtask."""
        mock_subprocess.return_value = MagicMock(
            returncode=0,
            stdout='{"is_atomic": true, "subtasks": [{"id": "is-prime", "description": "Write is_prime function", "prompt": "Write a Python function called is_prime that checks if a number is prime"}]}',
            stderr="",
        )

        decomposer = Decomposer()
        result = decomposer.decompose("Write a function to check if a number is prime")

        assert result.is_atomic is True
        assert len(result.subtasks) == 1
        assert result.subtasks[0].id == "is-prime"

    def test_decompose_complex_task(self, mock_subprocess):
        """Should decompose complex tasks into multiple subtasks."""
        mock_subprocess.return_value = MagicMock(
            returncode=0,
            stdout="""
            {
                "is_atomic": false,
                "subtasks": [
                    {"id": "add-auth", "description": "Add authentication", "prompt": "Implement JWT auth"},
                    {"id": "add-crud", "description": "Add CRUD endpoints", "prompt": "Create REST endpoints"},
                    {"id": "add-tests", "description": "Add tests", "prompt": "Write unit tests"}
                ]
            }
            """,
            stderr="",
        )

        decomposer = Decomposer()
        result = decomposer.decompose("Build a REST API with auth, CRUD, and tests")

        assert result.is_atomic is False
        assert len(result.subtasks) == 3
        assert result.subtasks[0].id == "add-auth"
        assert result.subtasks[1].id == "add-crud"
        assert result.subtasks[2].id == "add-tests"

    def test_handles_markdown_wrapped_json(self, mock_subprocess):
        """Should extract JSON from markdown code blocks."""
        mock_subprocess.return_value = MagicMock(
            returncode=0,
            stdout="""Here's the decomposition:

```json
{"is_atomic": true, "subtasks": [{"id": "task-1", "description": "Do it", "prompt": "Do the thing"}]}
```

This is an atomic task.""",
            stderr="",
        )

        decomposer = Decomposer()
        result = decomposer.decompose("Simple task")

        assert result.is_atomic is True
        assert len(result.subtasks) == 1

    def test_raises_on_invalid_json(self, mock_subprocess):
        """Should raise DecomposerError on unparseable response."""
        mock_subprocess.return_value = MagicMock(
            returncode=0,
            stdout="This is not JSON at all",
            stderr="",
        )

        decomposer = Decomposer()

        with pytest.raises(DecomposerError, match="Could not parse JSON"):
            decomposer.decompose("Some task")

    def test_preserves_original_query(self, mock_subprocess):
        """Should preserve the original query in the result."""
        mock_subprocess.return_value = MagicMock(
            returncode=0,
            stdout='{"is_atomic": true, "subtasks": [{"id": "t", "description": "d", "prompt": "p"}]}',
            stderr="",
        )

        query = "My original query"
        decomposer = Decomposer()
        result = decomposer.decompose(query)

        assert result.original_query == query

    def test_uses_api_when_specified(self):
        """Should use the API when use_api=True."""
        with patch("anthropic.Anthropic") as mock_anthropic:
            mock_response = MagicMock()
            mock_response.content = [
                MagicMock(
                    text='{"is_atomic": true, "subtasks": [{"id": "t", "description": "d", "prompt": "p"}]}'
                )
            ]
            mock_anthropic.return_value.messages.create.return_value = mock_response

            decomposer = Decomposer(use_api=True, model="claude-opus-4-20250514")
            decomposer.decompose("Test")

            call_args = mock_anthropic.return_value.messages.create.call_args
            assert call_args.kwargs["model"] == "claude-opus-4-20250514"

    def test_raises_on_cli_failure(self, mock_subprocess):
        """Should raise DecomposerError on CLI failure."""
        mock_subprocess.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="Error: something went wrong",
        )

        decomposer = Decomposer()

        with pytest.raises(DecomposerError, match="Claude CLI failed"):
            decomposer.decompose("Some task")

    def test_raises_on_cli_not_found(self, mock_subprocess):
        """Should raise DecomposerError when CLI is not installed."""
        mock_subprocess.side_effect = FileNotFoundError()

        decomposer = Decomposer()

        with pytest.raises(DecomposerError, match="Claude CLI not found"):
            decomposer.decompose("Some task")


class TestDecomposeTaskFunction:
    """Tests for the convenience decompose_task function."""

    @patch("swarm_orchestrator.decomposer.Decomposer")
    def test_calls_decomposer(self, mock_decomposer_class):
        """Should create a Decomposer and call decompose."""
        mock_instance = MagicMock()
        mock_decomposer_class.return_value = mock_instance
        mock_instance.decompose.return_value = DecompositionResult(
            is_atomic=True,
            subtasks=[Subtask(id="t", description="d", prompt="p")],
            original_query="query",
        )

        result = decompose_task("My query")

        mock_instance.decompose.assert_called_once_with("My query")
        assert result.is_atomic is True


class TestDecomposePrompt:
    """Tests for the decomposition prompt template."""

    def test_prompt_includes_query_placeholder(self):
        """Prompt should have {query} placeholder."""
        assert "{query}" in DECOMPOSE_PROMPT

    def test_prompt_describes_atomic_tasks(self):
        """Prompt should explain atomic tasks."""
        assert "ATOMIC" in DECOMPOSE_PROMPT

    def test_prompt_describes_complex_tasks(self):
        """Prompt should explain complex tasks."""
        assert "COMPLEX" in DECOMPOSE_PROMPT

    def test_prompt_specifies_json_output(self):
        """Prompt should specify JSON output format."""
        assert "JSON" in DECOMPOSE_PROMPT
        assert "is_atomic" in DECOMPOSE_PROMPT
        assert "subtasks" in DECOMPOSE_PROMPT
