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

    def test_timeout_propagated_to_client(self):
        """Should pass timeout to the Schaltwerk client."""
        with patch("swarm_orchestrator.orchestrator.get_client") as mock_get_client:
            Orchestrator(timeout=120)
            mock_get_client.assert_called_once_with(timeout=120)

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
        """Should cancel all provided sessions by finding actual names from Schaltwerk."""
        from swarm_orchestrator.schaltwerk import SessionStatus

        # Mock get_session_list to return sessions with suffixes (like real Schaltwerk)
        mock_schaltwerk_client.get_session_list.return_value = [
            SessionStatus(name="agent-1-ab", status="running", session_state="running", ready_to_merge=False, branch="b1"),
            SessionStatus(name="agent-2-cd", status="running", session_state="running", ready_to_merge=False, branch="b2"),
        ]

        sessions = ["agent-1", "agent-2"]
        orchestrator._cleanup_sessions(sessions)

        assert mock_schaltwerk_client.cancel_session.call_count == 2
        # Verify it used the actual names with suffixes
        mock_schaltwerk_client.cancel_session.assert_any_call("agent-1-ab", force=True)
        mock_schaltwerk_client.cancel_session.assert_any_call("agent-2-cd", force=True)

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


# =============================================================================
# Tests for MCP-based orchestrator flow
# =============================================================================

class TestMCPBasedOrchestrator:
    """Tests for the new MCP-based agent coordination flow."""

    @pytest.fixture
    def mcp_orchestrator(self, mock_schaltwerk_client):
        """Create an orchestrator with mocked dependencies."""
        with patch("swarm_orchestrator.orchestrator.get_client") as mock_get_client:
            mock_get_client.return_value = mock_schaltwerk_client
            orch = Orchestrator(agent_count=3, timeout=60)
            yield orch

    def test_generates_agent_prompt_with_mcp_instructions(
        self, mcp_orchestrator, sample_subtask
    ):
        """Agent prompts should include MCP tool usage instructions."""
        prompt = mcp_orchestrator._generate_agent_prompt(
            subtask=sample_subtask,
            task_id="task-1",
            agent_id="task-1-agent-0",
        )

        # Should include original task prompt
        assert sample_subtask.prompt in prompt

        # Should include task and agent IDs
        assert "task-1" in prompt
        assert "task-1-agent-0" in prompt

        # Should include MCP tool instructions
        assert "finished_work" in prompt
        assert "get_all_implementations" in prompt
        assert "cast_vote" in prompt

    def test_generates_unique_task_id(self, mcp_orchestrator, sample_subtask):
        """Should generate unique task IDs for each subtask."""
        task_id_1 = mcp_orchestrator._generate_task_id(sample_subtask, 0)
        task_id_2 = mcp_orchestrator._generate_task_id(sample_subtask, 1)

        assert task_id_1 != task_id_2
        assert sample_subtask.id in task_id_1

    def test_creates_swarm_task_on_spawn(
        self, mcp_orchestrator, sample_subtask
    ):
        """Should register task with SwarmMCPServer when spawning agents."""
        with patch.object(mcp_orchestrator, 'swarm_server') as mock_server:
            mock_server.create_task.return_value = MagicMock()

            mcp_orchestrator._spawn_agents_with_mcp(sample_subtask, "task-1")

            mock_server.create_task.assert_called_once()
            call_args = mock_server.create_task.call_args
            assert call_args.kwargs["task_id"] == "task-1"
            assert call_args.kwargs["agent_count"] == 3

    def test_wait_for_mcp_completion_polls_state(self, mcp_orchestrator):
        """Should poll SwarmState until all agents have finished."""
        mock_state = MagicMock()

        # Simulate agents finishing over time
        mock_state.get_task.return_value.all_agents_finished.side_effect = [
            False, False, True
        ]

        with patch.object(mcp_orchestrator, 'swarm_server') as mock_server:
            mock_server.state = mock_state

            with patch("time.sleep"):
                mcp_orchestrator._wait_for_mcp_completion("task-1")

        assert mock_state.get_task.call_count >= 1

    def test_wait_for_mcp_votes_polls_state(self, mcp_orchestrator):
        """Should poll SwarmState until all agents have voted."""
        mock_state = MagicMock()

        # Simulate agents voting over time
        mock_state.get_task.return_value.all_agents_voted.side_effect = [
            False, False, True
        ]
        mock_state.get_vote_results.return_value = {
            "success": True,
            "all_voted": True,
            "winner": "task-1-agent-2",
            "vote_counts": {"task-1-agent-2": 2, "task-1-agent-0": 1},
        }

        with patch.object(mcp_orchestrator, 'swarm_server') as mock_server:
            mock_server.state = mock_state

            with patch("time.sleep"):
                result = mcp_orchestrator._wait_for_mcp_votes("task-1")

        assert result["winner"] == "task-1-agent-2"

    def test_get_winner_session_from_agent_id(self, mcp_orchestrator):
        """Should map winning agent ID to Schaltwerk session name."""
        mock_task = MagicMock()
        mock_task.session_names = {
            "task-1-agent-0": "task-1-agent-0",
            "task-1-agent-1": "task-1-agent-1",
            "task-1-agent-2": "task-1-agent-2",
        }

        with patch.object(mcp_orchestrator, 'swarm_server') as mock_server:
            mock_server.state.get_task.return_value = mock_task

            session = mcp_orchestrator._get_winner_session("task-1", "task-1-agent-2")

        assert session == "task-1-agent-2"


class TestAgentPromptTemplate:
    """Tests for the agent prompt template."""

    def test_prompt_template_exists(self):
        """Should have an AGENT_PROMPT_TEMPLATE constant."""
        from swarm_orchestrator.orchestrator import AGENT_PROMPT_TEMPLATE

        assert AGENT_PROMPT_TEMPLATE is not None
        assert len(AGENT_PROMPT_TEMPLATE) > 100

    def test_prompt_template_has_placeholders(self):
        """Template should have required placeholders."""
        from swarm_orchestrator.orchestrator import AGENT_PROMPT_TEMPLATE

        assert "{task_prompt}" in AGENT_PROMPT_TEMPLATE
        assert "{task_id}" in AGENT_PROMPT_TEMPLATE
        assert "{agent_id}" in AGENT_PROMPT_TEMPLATE
        assert "{agent_count}" in AGENT_PROMPT_TEMPLATE

    def test_prompt_template_describes_workflow(self):
        """Template should describe the expected workflow with embedded instructions."""
        from swarm_orchestrator.orchestrator import AGENT_PROMPT_TEMPLATE

        # Instructions are now embedded directly (not referencing external file)
        # Should contain key workflow phases
        assert "Phase 1: Implementation" in AGENT_PROMPT_TEMPLATE
        assert "Phase 2: Signal Completion" in AGENT_PROMPT_TEMPLATE
        assert "Phase 3: Review and Vote" in AGENT_PROMPT_TEMPLATE

        # Should mention key MCP tools
        assert "finished_work" in AGENT_PROMPT_TEMPLATE
        assert "cast_vote" in AGENT_PROMPT_TEMPLATE
        assert "get_all_implementations" in AGENT_PROMPT_TEMPLATE
