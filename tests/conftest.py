"""
Shared pytest fixtures for swarm_orchestrator tests.
"""

import pytest
from unittest.mock import MagicMock

from swarm_orchestrator.decomposer import Subtask, SubtaskScope, DecompositionResult
from swarm_orchestrator.voting import VoteGroup, VoteResult


def make_test_subtask(
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


@pytest.fixture(autouse=True)
def reset_schaltwerk_singleton():
    """Reset the Schaltwerk client singleton before and after each test."""
    from swarm_orchestrator import schaltwerk
    schaltwerk.reset_client()
    yield
    schaltwerk.reset_client()


@pytest.fixture
def sample_subtask():
    """A sample subtask for testing."""
    return make_test_subtask(
        id="add-feature",
        title="Add Feature",
        description="Add a new feature",
        implementation="Implement the feature with proper error handling",
        verification="Test the feature works correctly",
        files=["src/feature.py", "tests/test_feature.py"],
        estimated_loc=60,
    )


@pytest.fixture
def sample_decomposition_atomic(sample_subtask):
    """An atomic decomposition result."""
    return DecompositionResult(
        is_atomic=True,
        subtasks=[sample_subtask],
        original_query="Add a new feature",
    )


@pytest.fixture
def sample_decomposition_complex():
    """A complex decomposition result with multiple subtasks."""
    return DecompositionResult(
        is_atomic=False,
        subtasks=[
            make_test_subtask(id="task-1", title="First Task", description="First task"),
            make_test_subtask(id="task-2", title="Second Task", description="Second task", depends_on=["task-1"]),
            make_test_subtask(id="task-3", title="Third Task", description="Third task", depends_on=["task-2"]),
        ],
        original_query="Do multiple things",
    )


@pytest.fixture
def sample_vote_result_consensus():
    """A vote result with consensus."""
    winner = VoteGroup(
        diff_hash="abc123",
        diff_content="+def hello(): pass",
        sessions=["agent-0", "agent-1", "agent-2"],
    )
    return VoteResult(
        groups=[
            winner,
            VoteGroup(
                diff_hash="def456",
                diff_content="+def other(): pass",
                sessions=["agent-3", "agent-4"],
            ),
        ],
        winner=winner,
        total_votes=5,
        consensus_reached=True,
        confidence=0.6,
    )


@pytest.fixture
def sample_vote_result_no_consensus():
    """A vote result without consensus."""
    return VoteResult(
        groups=[
            VoteGroup(diff_hash="a", diff_content="+A", sessions=["agent-0"]),
            VoteGroup(diff_hash="b", diff_content="+B", sessions=["agent-1"]),
            VoteGroup(diff_hash="c", diff_content="+C", sessions=["agent-2"]),
        ],
        winner=None,
        total_votes=3,
        consensus_reached=False,
        confidence=1 / 3,
    )


@pytest.fixture
def mock_schaltwerk_client():
    """A mock SchaltwerkClient for testing."""
    from swarm_orchestrator.schaltwerk import SessionStatus

    client = MagicMock()
    client.create_spec.return_value = {"created": True}
    client.start_agent.return_value = {"started": True}
    client.get_session_status.return_value = SessionStatus(
        name="test-session",
        status="completed",
        session_state="completed",
        ready_to_merge=True,
        branch="test-branch",
    )
    client.get_full_diff.return_value = "+def test(): pass"
    client.merge_session.return_value = {"merged": True}
    client.cancel_session.return_value = {"cancelled": True}
    client.get_session_list.return_value = []
    return client


@pytest.fixture
def mock_mcp_client():
    """A mock MCPClient for testing."""
    client = MagicMock()
    client.call_tool.return_value = {}
    client.start.return_value = None
    client.stop.return_value = None
    return client
