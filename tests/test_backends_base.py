"""
Tests for abstract backend base classes.

Verifies interface contracts for WorktreeBackend, AgentBackend, and LLMBackend.
"""

import pytest
from abc import ABC

from swarm_orchestrator.backends.base import (
    WorktreeBackend,
    AgentBackend,
    LLMBackend,
    SessionInfo,
    DiffResult,
    AgentStatus,
    DecomposeResult,
)


class TestSessionInfo:
    """Tests for the SessionInfo data model."""

    def test_session_info_creation(self):
        """SessionInfo can be created with required fields."""
        info = SessionInfo(
            name="test-session",
            status="running",
            branch="feature/test",
        )
        assert info.name == "test-session"
        assert info.status == "running"
        assert info.branch == "feature/test"
        assert info.worktree_path is None
        assert info.ready_to_merge is False

    def test_session_info_with_all_fields(self):
        """SessionInfo can be created with all optional fields."""
        info = SessionInfo(
            name="test-session",
            status="reviewed",
            branch="feature/test",
            worktree_path="/path/to/worktree",
            ready_to_merge=True,
            created_at="2024-01-01T00:00:00Z",
            metadata={"key": "value"},
        )
        assert info.worktree_path == "/path/to/worktree"
        assert info.ready_to_merge is True
        assert info.created_at == "2024-01-01T00:00:00Z"
        assert info.metadata == {"key": "value"}


class TestDiffResult:
    """Tests for the DiffResult data model."""

    def test_diff_result_creation(self):
        """DiffResult can be created with required fields."""
        result = DiffResult(
            files=["src/main.py", "tests/test_main.py"],
            content="diff --git a/src/main.py...",
        )
        assert result.files == ["src/main.py", "tests/test_main.py"]
        assert result.content == "diff --git a/src/main.py..."
        assert result.stats is None

    def test_diff_result_with_stats(self):
        """DiffResult can include stats."""
        result = DiffResult(
            files=["src/main.py"],
            content="diff content",
            stats={"added": 10, "deleted": 5},
        )
        assert result.stats == {"added": 10, "deleted": 5}


class TestAgentStatus:
    """Tests for the AgentStatus data model."""

    def test_agent_status_creation(self):
        """AgentStatus can be created with required fields."""
        status = AgentStatus(
            agent_id="agent-1",
            is_finished=False,
        )
        assert status.agent_id == "agent-1"
        assert status.is_finished is False
        assert status.implementation is None

    def test_agent_status_with_implementation(self):
        """AgentStatus can include implementation."""
        status = AgentStatus(
            agent_id="agent-1",
            is_finished=True,
            implementation="diff content here",
        )
        assert status.is_finished is True
        assert status.implementation == "diff content here"


class TestDecomposeResult:
    """Tests for the DecomposeResult data model."""

    def test_decompose_result_creation(self):
        """DecomposeResult can be created with required fields."""
        result = DecomposeResult(
            is_atomic=True,
            raw_response={"key": "value"},
        )
        assert result.is_atomic is True
        assert result.raw_response == {"key": "value"}
        assert result.reasoning is None

    def test_decompose_result_with_reasoning(self):
        """DecomposeResult can include reasoning."""
        result = DecomposeResult(
            is_atomic=False,
            raw_response={"subtasks": []},
            reasoning="Task requires multiple steps",
        )
        assert result.reasoning == "Task requires multiple steps"


class TestWorktreeBackend:
    """Tests for the WorktreeBackend abstract base class."""

    def test_is_abstract_class(self):
        """WorktreeBackend is an abstract class."""
        assert issubclass(WorktreeBackend, ABC)

    def test_cannot_instantiate_directly(self):
        """Cannot instantiate WorktreeBackend directly."""
        with pytest.raises(TypeError) as exc_info:
            WorktreeBackend()
        assert "abstract" in str(exc_info.value).lower()

    def test_concrete_implementation_must_implement_all_methods(self):
        """Concrete implementation must implement all abstract methods."""
        class PartialImpl(WorktreeBackend):
            def create_session(self, name: str, content: str) -> SessionInfo:
                return SessionInfo(name=name, status="created", branch="test")

        with pytest.raises(TypeError):
            PartialImpl()

    def test_concrete_implementation_works(self):
        """Concrete implementation can be instantiated."""
        class CompleteImpl(WorktreeBackend):
            def create_session(self, name: str, content: str) -> SessionInfo:
                return SessionInfo(name=name, status="created", branch="test")

            def delete_session(self, name: str, force: bool = False) -> None:
                pass

            def get_session(self, name: str) -> SessionInfo | None:
                return None

            def list_sessions(self, filter_type: str = "all") -> list[SessionInfo]:
                return []

            def get_diff(self, session_name: str) -> DiffResult:
                return DiffResult(files=[], content="")

            def merge_session(self, name: str, commit_message: str) -> None:
                pass

        impl = CompleteImpl()
        assert isinstance(impl, WorktreeBackend)


class TestAgentBackend:
    """Tests for the AgentBackend abstract base class."""

    def test_is_abstract_class(self):
        """AgentBackend is an abstract class."""
        assert issubclass(AgentBackend, ABC)

    def test_cannot_instantiate_directly(self):
        """Cannot instantiate AgentBackend directly."""
        with pytest.raises(TypeError) as exc_info:
            AgentBackend()
        assert "abstract" in str(exc_info.value).lower()

    def test_concrete_implementation_works(self):
        """Concrete implementation can be instantiated."""
        class CompleteImpl(AgentBackend):
            def spawn_agent(self, session_name: str, prompt: str) -> str:
                return "agent-id"

            def wait_for_completion(
                self, agent_ids: list[str], timeout: int | None = None
            ) -> dict[str, AgentStatus]:
                return {}

            def send_message(self, agent_id: str, message: str) -> None:
                pass

            def get_status(self, agent_id: str) -> AgentStatus:
                return AgentStatus(agent_id=agent_id, is_finished=False)

            def stop_agent(self, session_name: str) -> bool:
                return True

        impl = CompleteImpl()
        assert isinstance(impl, AgentBackend)


class TestLLMBackend:
    """Tests for the LLMBackend abstract base class."""

    def test_is_abstract_class(self):
        """LLMBackend is an abstract class."""
        assert issubclass(LLMBackend, ABC)

    def test_cannot_instantiate_directly(self):
        """Cannot instantiate LLMBackend directly."""
        with pytest.raises(TypeError) as exc_info:
            LLMBackend()
        assert "abstract" in str(exc_info.value).lower()

    def test_concrete_implementation_works(self):
        """Concrete implementation can be instantiated."""
        class CompleteImpl(LLMBackend):
            def decompose(self, query: str, context: str | None = None) -> DecomposeResult:
                return DecomposeResult(is_atomic=True, raw_response={})

            def explore(self, query: str) -> str:
                return "exploration results"

        impl = CompleteImpl()
        assert isinstance(impl, LLMBackend)


class TestInterfaceDocstrings:
    """Tests to verify interfaces have proper documentation."""

    def test_worktree_backend_has_docstrings(self):
        """WorktreeBackend methods have docstrings."""
        assert WorktreeBackend.__doc__ is not None
        assert WorktreeBackend.create_session.__doc__ is not None
        assert WorktreeBackend.delete_session.__doc__ is not None
        assert WorktreeBackend.get_diff.__doc__ is not None
        assert WorktreeBackend.merge_session.__doc__ is not None

    def test_agent_backend_has_docstrings(self):
        """AgentBackend methods have docstrings."""
        assert AgentBackend.__doc__ is not None
        assert AgentBackend.spawn_agent.__doc__ is not None
        assert AgentBackend.wait_for_completion.__doc__ is not None
        assert AgentBackend.send_message.__doc__ is not None
        assert AgentBackend.stop_agent.__doc__ is not None

    def test_llm_backend_has_docstrings(self):
        """LLMBackend methods have docstrings."""
        assert LLMBackend.__doc__ is not None
        assert LLMBackend.decompose.__doc__ is not None
        assert LLMBackend.explore.__doc__ is not None
