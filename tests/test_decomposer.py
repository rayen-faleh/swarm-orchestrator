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
    SubtaskScope,
    DecompositionResult,
    decompose_task,
    validate_decomposition,
    DECOMPOSE_PROMPT,
    SCOPE_LIMITS,
)


def make_subtask(
    id: str = "test-task",
    title: str = "Test Task",
    description: str = "Test description",
    implementation: str = "Implement the test",
    verification: str = "Verify it works",
    success_criteria: list[str] = None,
    depends_on: list[str] = None,
    files: list[str] = None,
    estimated_loc: int = 50,
    functions: list[str] = None,
) -> Subtask:
    """Helper to create test subtasks with defaults."""
    return Subtask(
        id=id,
        title=title,
        description=description,
        scope=SubtaskScope(
            files=files or ["src/test.py"],
            estimated_loc=estimated_loc,
            functions=functions or ["test_function"],
        ),
        implementation=implementation,
        verification=verification,
        success_criteria=success_criteria or ["Tests pass"],
        depends_on=depends_on or [],
    )


class TestSubtaskScope:
    """Tests for SubtaskScope dataclass."""

    def test_create_scope(self):
        """Should create a scope with all fields."""
        scope = SubtaskScope(
            files=["src/auth.py", "tests/test_auth.py"],
            estimated_loc=60,
            functions=["login", "logout"],
        )

        assert len(scope.files) == 2
        assert scope.estimated_loc == 60
        assert len(scope.functions) == 2

    def test_is_within_limits_valid(self):
        """Should return True for scope within limits."""
        scope = SubtaskScope(
            files=["src/auth.py", "tests/test_auth.py"],
            estimated_loc=80,
            functions=["login", "logout", "verify"],
        )

        assert scope.is_within_limits() is True

    def test_is_within_limits_exceeds_files(self):
        """Should return False when files exceed limit."""
        scope = SubtaskScope(
            files=["f1.py", "f2.py", "f3.py", "f4.py"],  # > 3 files
            estimated_loc=50,
            functions=["func"],
        )

        assert scope.is_within_limits() is False

    def test_is_within_limits_exceeds_loc(self):
        """Should return False when LOC exceeds limit."""
        scope = SubtaskScope(
            files=["f1.py"],
            estimated_loc=200,  # > 150 LOC
            functions=["func"],
        )

        assert scope.is_within_limits() is False

    def test_get_warnings_within_target(self):
        """Should return no warnings when within target."""
        scope = SubtaskScope(
            files=["src/auth.py"],
            estimated_loc=50,
            functions=["login"],
        )

        assert scope.get_warnings() == []

    def test_get_warnings_exceeds_soft_limits(self):
        """Should return warnings for soft limit violations."""
        scope = SubtaskScope(
            files=["f1.py", "f2.py", "f3.py"],  # > 2 target
            estimated_loc=100,  # > 80 target
            functions=["f1", "f2", "f3", "f4"],  # > 3 target
        )

        warnings = scope.get_warnings()
        assert len(warnings) == 3
        assert any("3 files" in w for w in warnings)
        assert any("100 LOC" in w for w in warnings)
        assert any("4 functions" in w for w in warnings)


class TestSubtask:
    """Tests for Subtask dataclass."""

    def test_create_subtask(self):
        """Should create a subtask with all fields."""
        subtask = make_subtask(
            id="add-auth",
            title="Add Authentication",
            description="Add JWT authentication to the API",
            implementation="Implement JWT auth with login/logout",
            verification="Test valid/invalid credentials",
            files=["src/auth.py", "tests/test_auth.py"],
            estimated_loc=60,
        )

        assert subtask.id == "add-auth"
        assert subtask.title == "Add Authentication"
        assert subtask.description == "Add JWT authentication to the API"
        assert "JWT" in subtask.implementation
        assert subtask.scope.estimated_loc == 60

    def test_prompt_property(self):
        """Should generate a formatted prompt from subtask fields."""
        subtask = make_subtask(
            title="Add Health Endpoint",
            description="Add /health endpoint for monitoring",
            implementation="Create GET /health returning status",
            verification="Test returns 200 with valid JSON",
            success_criteria=["Endpoint returns 200", "Tests pass"],
            files=["src/routes.py"],
        )

        prompt = subtask.prompt

        assert "Add Health Endpoint" in prompt
        assert "/health" in prompt
        assert "Endpoint returns 200" in prompt
        assert "Tests pass" in prompt
        assert "src/routes.py" in prompt


class TestDecompositionResult:
    """Tests for DecompositionResult dataclass."""

    def test_atomic_result(self):
        """Should represent an atomic task."""
        result = DecompositionResult(
            is_atomic=True,
            subtasks=[make_subtask(id="task-1", description="Do thing")],
            original_query="Do the thing",
        )

        assert result.is_atomic is True
        assert len(result.subtasks) == 1

    def test_complex_result(self):
        """Should represent a complex task with multiple subtasks."""
        result = DecompositionResult(
            is_atomic=False,
            subtasks=[
                make_subtask(id="task-1", description="First"),
                make_subtask(id="task-2", description="Second", depends_on=["task-1"]),
            ],
            original_query="Do multiple things",
        )

        assert result.is_atomic is False
        assert len(result.subtasks) == 2
        assert result.subtasks[1].depends_on == ["task-1"]

    def test_reasoning_field(self):
        """Should store decomposition reasoning."""
        result = DecompositionResult(
            is_atomic=False,
            subtasks=[make_subtask()],
            original_query="test",
            reasoning="Split by domain boundaries",
        )

        assert result.reasoning == "Split by domain boundaries"

    def test_total_estimated_loc(self):
        """Should sum LOC across all subtasks."""
        result = DecompositionResult(
            is_atomic=False,
            subtasks=[
                make_subtask(id="t1", estimated_loc=40),
                make_subtask(id="t2", estimated_loc=60),
                make_subtask(id="t3", estimated_loc=50),
            ],
            original_query="test",
        )

        assert result.total_estimated_loc() == 150

    def test_get_scope_warnings(self):
        """Should collect warnings from all subtasks."""
        result = DecompositionResult(
            is_atomic=False,
            subtasks=[
                make_subtask(id="t1", estimated_loc=50),  # No warnings
                make_subtask(id="t2", estimated_loc=100),  # Warning: > 80 LOC
            ],
            original_query="test",
        )

        warnings = result.get_scope_warnings()
        assert "t1" not in warnings
        assert "t2" in warnings
        assert any("100 LOC" in w for w in warnings["t2"])


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
            subtasks=[make_subtask(id="t", description="d")],
            original_query="query",
        )

        result = decompose_task("My query")

        mock_instance.decompose.assert_called_once_with("My query")
        assert result.is_atomic is True


class TestValidateDecomposition:
    """Tests for the validate_decomposition function."""

    def test_valid_decomposition(self):
        """Should return valid for well-scoped decomposition."""
        result = DecompositionResult(
            is_atomic=False,
            subtasks=[
                make_subtask(id="t1", files=["f1.py"], estimated_loc=50),
                make_subtask(id="t2", files=["f2.py"], estimated_loc=60, depends_on=["t1"]),
            ],
            original_query="test",
        )

        is_valid, warnings = validate_decomposition(result)

        assert is_valid is True
        # May have soft limit warnings but should still be valid
        assert not any("EXCEEDS LIMIT" in w for w in warnings)

    def test_invalid_too_many_files(self):
        """Should fail validation when files exceed hard limit."""
        result = DecompositionResult(
            is_atomic=True,
            subtasks=[
                make_subtask(
                    id="t1",
                    files=["f1.py", "f2.py", "f3.py", "f4.py"],  # > 3 max
                    estimated_loc=50,
                )
            ],
            original_query="test",
        )

        is_valid, warnings = validate_decomposition(result)

        assert is_valid is False
        assert any("EXCEEDS LIMIT" in w and "files" in w for w in warnings)

    def test_invalid_too_many_loc(self):
        """Should fail validation when LOC exceeds hard limit."""
        result = DecompositionResult(
            is_atomic=True,
            subtasks=[
                make_subtask(id="t1", estimated_loc=200)  # > 150 max
            ],
            original_query="test",
        )

        is_valid, warnings = validate_decomposition(result)

        assert is_valid is False
        assert any("EXCEEDS LIMIT" in w and "LOC" in w for w in warnings)

    def test_invalid_unknown_dependency(self):
        """Should fail validation for unknown dependencies."""
        result = DecompositionResult(
            is_atomic=False,
            subtasks=[
                make_subtask(id="t1"),
                make_subtask(id="t2", depends_on=["unknown-task"]),
            ],
            original_query="test",
        )

        is_valid, warnings = validate_decomposition(result)

        assert is_valid is False
        assert any("unknown subtask" in w for w in warnings)

    def test_warns_on_high_total_loc(self):
        """Should warn when total LOC is high."""
        result = DecompositionResult(
            is_atomic=False,
            subtasks=[
                make_subtask(id="t1", estimated_loc=140),
                make_subtask(id="t2", estimated_loc=140),
                make_subtask(id="t3", estimated_loc=140),
                make_subtask(id="t4", estimated_loc=140),  # Total: 560 > 500
            ],
            original_query="test",
        )

        is_valid, warnings = validate_decomposition(result)

        # Valid because individual tasks are within limits
        assert is_valid is True
        # But should warn about total
        assert any("Total estimated LOC" in w for w in warnings)


class TestScopeLimits:
    """Tests for scope limit constants."""

    def test_limits_exist(self):
        """Should have all expected limit keys."""
        assert "files" in SCOPE_LIMITS
        assert "estimated_loc" in SCOPE_LIMITS
        assert "functions" in SCOPE_LIMITS

    def test_limits_have_target_and_max(self):
        """Each limit should have target and max values."""
        for key, limits in SCOPE_LIMITS.items():
            assert "target" in limits, f"{key} missing target"
            assert "max" in limits, f"{key} missing max"
            assert limits["target"] <= limits["max"], f"{key} target > max"


class TestDecomposePrompt:
    """Tests for the decomposition prompt template."""

    def test_prompt_includes_query_placeholder(self):
        """Prompt should have {query} placeholder."""
        assert "{query}" in DECOMPOSE_PROMPT

    def test_prompt_describes_atomic_tasks(self):
        """Prompt should explain atomic tasks."""
        # New prompt uses different terminology
        assert "atomic" in DECOMPOSE_PROMPT.lower()

    def test_prompt_describes_scope_guidelines(self):
        """Prompt should include research-backed scope guidelines."""
        assert "files" in DECOMPOSE_PROMPT.lower()
        assert "loc" in DECOMPOSE_PROMPT.lower()
        assert "functions" in DECOMPOSE_PROMPT.lower()

    def test_prompt_specifies_json_output(self):
        """Prompt should specify JSON output format."""
        assert "JSON" in DECOMPOSE_PROMPT
        assert "is_atomic" in DECOMPOSE_PROMPT
        assert "subtasks" in DECOMPOSE_PROMPT

    def test_prompt_includes_scope_field(self):
        """Prompt should specify scope in subtask format."""
        assert "scope" in DECOMPOSE_PROMPT
        assert "estimated_loc" in DECOMPOSE_PROMPT

    def test_prompt_includes_verification(self):
        """Prompt should specify verification requirements."""
        assert "verification" in DECOMPOSE_PROMPT.lower()
        assert "success_criteria" in DECOMPOSE_PROMPT

    def test_prompt_references_research(self):
        """Prompt should reference research basis."""
        # Check for research paper references or concepts
        assert any(term in DECOMPOSE_PROMPT for term in [
            "SWE-bench", "Agentless", "MAKER", "research", "Research"
        ])
