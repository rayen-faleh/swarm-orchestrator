"""
Tests for the orchestrator module.

TDD approach: These tests define the expected behavior of the orchestration workflow.
"""

import pytest
from unittest.mock import MagicMock, patch

from swarm_orchestrator.orchestrator import (
    Orchestrator,
    SubtaskResult,
    OrchestrationResult,
    run_swarm,
)
from swarm_orchestrator.decomposer import Subtask, DecompositionResult
from swarm_orchestrator.voting import VoteGroup, VoteResult


class TestOrchestrator:
    """Tests for Orchestrator class."""

    @pytest.fixture
    def orchestrator(self, mock_schaltwerk_client):
        """Create an orchestrator with mocked dependencies."""
        with patch("swarm_orchestrator.orchestrator.get_client") as mock_get_client:
            mock_get_client.return_value = mock_schaltwerk_client
            orch = Orchestrator(agent_count=3, timeout=60)
            yield orch

    @pytest.fixture
    def mock_decompose(self):
        """Mock the decompose_task function."""
        with patch("swarm_orchestrator.orchestrator.decompose_task") as mock:
            yield mock

    def test_initialization(self):
        """Should initialize with default values."""
        with patch("swarm_orchestrator.orchestrator.get_client"):
            orch = Orchestrator()
            assert orch.agent_count == 3
            assert orch.timeout == 600

    def test_initialization_custom_values(self):
        """Should accept custom agent count and timeout."""
        with patch("swarm_orchestrator.orchestrator.get_client"):
            orch = Orchestrator(agent_count=5, timeout=300)
            assert orch.agent_count == 5
            assert orch.timeout == 300

    def test_spawn_agents_creates_correct_number(
        self, orchestrator, sample_subtask, mock_schaltwerk_client
    ):
        """Should spawn the configured number of agents."""
        sessions = orchestrator._spawn_agents(sample_subtask)

        assert len(sessions) == 3
        assert mock_schaltwerk_client.create_spec.call_count == 3
        assert mock_schaltwerk_client.start_agent.call_count == 3

    def test_spawn_agents_naming_convention(
        self, orchestrator, sample_subtask, mock_schaltwerk_client
    ):
        """Sessions should be named {subtask_id}-agent-{n}."""
        sessions = orchestrator._spawn_agents(sample_subtask)

        assert sessions[0] == "add-feature-agent-0"
        assert sessions[1] == "add-feature-agent-1"
        assert sessions[2] == "add-feature-agent-2"

    def test_vote_on_outputs_with_consensus(
        self, orchestrator, mock_schaltwerk_client
    ):
        """Should correctly identify consensus from diffs."""
        # All agents return same diff
        mock_schaltwerk_client.get_full_diff.return_value = "+same content"

        sessions = ["agent-0", "agent-1", "agent-2"]
        result = orchestrator._vote_on_outputs(sessions)

        assert result.consensus_reached is True
        assert result.confidence == 1.0

    def test_vote_on_outputs_no_consensus(
        self, orchestrator, mock_schaltwerk_client
    ):
        """Should correctly identify when there's no consensus."""
        # Each agent returns different diff
        diffs = ["+content A", "+content B", "+content C"]
        mock_schaltwerk_client.get_full_diff.side_effect = diffs

        sessions = ["agent-0", "agent-1", "agent-2"]
        result = orchestrator._vote_on_outputs(sessions)

        assert result.consensus_reached is False

    def test_merge_winner_success(
        self, orchestrator, sample_subtask, mock_schaltwerk_client
    ):
        """Should merge the winning session."""
        mock_schaltwerk_client.merge_session.return_value = {"merged": True}

        success = orchestrator._merge_winner("agent-0", sample_subtask)

        assert success is True
        mock_schaltwerk_client.merge_session.assert_called_once()

    def test_merge_winner_failure(
        self, orchestrator, sample_subtask, mock_schaltwerk_client
    ):
        """Should handle merge failures gracefully."""
        mock_schaltwerk_client.merge_session.side_effect = Exception("Merge failed")

        success = orchestrator._merge_winner("agent-0", sample_subtask)

        assert success is False

    def test_cleanup_sessions(self, orchestrator, mock_schaltwerk_client):
        """Should cancel all provided sessions."""
        sessions = ["agent-1", "agent-2"]
        orchestrator._cleanup_sessions(sessions)

        assert mock_schaltwerk_client.cancel_session.call_count == 2

    def test_cleanup_empty_list(self, orchestrator, mock_schaltwerk_client):
        """Should handle empty session list."""
        orchestrator._cleanup_sessions([])

        mock_schaltwerk_client.cancel_session.assert_not_called()


class TestSubtaskResult:
    """Tests for SubtaskResult dataclass."""

    def test_create_successful_result(self, sample_subtask, sample_vote_result_consensus):
        """Should create a result for successful consensus."""
        result = SubtaskResult(
            subtask=sample_subtask,
            sessions=["agent-0", "agent-1", "agent-2"],
            vote_result=sample_vote_result_consensus,
            winner_session="agent-0",
            merged=True,
        )

        assert result.merged is True
        assert result.winner_session == "agent-0"

    def test_create_failed_result(self, sample_subtask, sample_vote_result_no_consensus):
        """Should create a result for failed consensus."""
        result = SubtaskResult(
            subtask=sample_subtask,
            sessions=["agent-0", "agent-1", "agent-2"],
            vote_result=sample_vote_result_no_consensus,
            winner_session=None,
            merged=False,
        )

        assert result.merged is False
        assert result.winner_session is None


class TestOrchestrationResult:
    """Tests for OrchestrationResult dataclass."""

    def test_overall_success_all_merged(self, sample_subtask, sample_vote_result_consensus):
        """Overall success when all subtasks merged."""
        subtask_results = [
            SubtaskResult(
                subtask=sample_subtask,
                sessions=["a"],
                vote_result=sample_vote_result_consensus,
                winner_session="a",
                merged=True,
            )
        ]

        result = OrchestrationResult(
            query="Test query",
            decomposition=DecompositionResult(
                is_atomic=True,
                subtasks=[sample_subtask],
                original_query="Test query",
            ),
            subtask_results=subtask_results,
            overall_success=True,
        )

        assert result.overall_success is True

    def test_overall_failure_some_not_merged(
        self, sample_subtask, sample_vote_result_consensus, sample_vote_result_no_consensus
    ):
        """Overall failure when any subtask not merged."""
        subtask_results = [
            SubtaskResult(
                subtask=sample_subtask,
                sessions=["a"],
                vote_result=sample_vote_result_consensus,
                winner_session="a",
                merged=True,
            ),
            SubtaskResult(
                subtask=Subtask(id="b", description="b", prompt="b"),
                sessions=["b"],
                vote_result=sample_vote_result_no_consensus,
                winner_session=None,
                merged=False,
            ),
        ]

        result = OrchestrationResult(
            query="Test query",
            decomposition=DecompositionResult(
                is_atomic=False,
                subtasks=[sample_subtask],
                original_query="Test query",
            ),
            subtask_results=subtask_results,
            overall_success=False,
        )

        assert result.overall_success is False


class TestRunSwarmFunction:
    """Tests for the convenience run_swarm function."""

    @patch("swarm_orchestrator.orchestrator.Orchestrator")
    def test_creates_orchestrator_with_defaults(self, mock_orchestrator_class):
        """Should create orchestrator with default values."""
        mock_instance = MagicMock()
        mock_orchestrator_class.return_value = mock_instance
        mock_instance.run.return_value = MagicMock()

        run_swarm("Test query")

        mock_orchestrator_class.assert_called_once_with(
            agent_count=3, timeout=600
        )

    @patch("swarm_orchestrator.orchestrator.Orchestrator")
    def test_creates_orchestrator_with_custom_values(self, mock_orchestrator_class):
        """Should pass custom values to orchestrator."""
        mock_instance = MagicMock()
        mock_orchestrator_class.return_value = mock_instance
        mock_instance.run.return_value = MagicMock()

        run_swarm("Test query", agents=5, timeout=300)

        mock_orchestrator_class.assert_called_once_with(
            agent_count=5, timeout=300
        )

    @patch("swarm_orchestrator.orchestrator.Orchestrator")
    def test_calls_run_with_query(self, mock_orchestrator_class):
        """Should call run with the provided query."""
        mock_instance = MagicMock()
        mock_orchestrator_class.return_value = mock_instance
        mock_instance.run.return_value = MagicMock()

        run_swarm("My query")

        mock_instance.run.assert_called_once_with("My query")
