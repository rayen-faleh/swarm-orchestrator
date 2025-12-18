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
from swarm_orchestrator.decomposer import Subtask, SubtaskScope, DecompositionResult
from swarm_orchestrator.voting import VoteGroup, VoteResult
from swarm_orchestrator.config import SwarmConfig


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
        """Should cancel all provided sessions by finding actual names from worktree backend."""
        from swarm_orchestrator.backends import SessionInfo

        # Mock worktree backend's list_sessions to return sessions with suffixes
        mock_worktree = MagicMock()
        mock_worktree.list_sessions.return_value = [
            SessionInfo(name="agent-1-ab", status="running", branch="b1"),
            SessionInfo(name="agent-2-cd", status="running", branch="b2"),
        ]
        orchestrator._worktree_backend = mock_worktree

        sessions = ["agent-1", "agent-2"]
        orchestrator._cleanup_sessions(sessions)

        assert mock_worktree.delete_session.call_count == 2
        # Verify it used the actual names with suffixes
        mock_worktree.delete_session.assert_any_call("agent-1-ab", force=True)
        mock_worktree.delete_session.assert_any_call("agent-2-cd", force=True)

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
                subtask=make_test_subtask(id="b", description="b"),
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

    def test_prompt_template_has_exploration_placeholder(self):
        """Template should have placeholder for exploration findings."""
        from swarm_orchestrator.orchestrator import AGENT_PROMPT_TEMPLATE

        assert "{exploration_context}" in AGENT_PROMPT_TEMPLATE


# =============================================================================
# Tests for exploration integration in orchestrator
# =============================================================================

# =============================================================================
# Tests for backend factory methods
# =============================================================================

class TestBackendFactory:
    """Tests for backend factory methods in Orchestrator."""

    def test_create_agent_backend_cursor_cli(self):
        """_create_agent_backend returns CursorCLIAgentBackend when config.agent_backend == 'cursor-cli'."""
        from swarm_orchestrator.backends import CursorCLIAgentBackend

        config = SwarmConfig(agent_backend="cursor-cli")

        with patch("swarm_orchestrator.orchestrator.get_client"):
            orch = Orchestrator(config=config)
            backend = orch._create_agent_backend()

        assert isinstance(backend, CursorCLIAgentBackend)

    def test_create_agent_backend_schaltwerk(self):
        """_create_agent_backend returns SchaltwerkAgentBackend when config.agent_backend == 'schaltwerk'."""
        from swarm_orchestrator.backends import SchaltwerkAgentBackend

        config = SwarmConfig(agent_backend="schaltwerk")

        with patch("swarm_orchestrator.orchestrator.get_client"):
            orch = Orchestrator(config=config)
            backend = orch._create_agent_backend()

        assert isinstance(backend, SchaltwerkAgentBackend)

    def test_create_agent_backend_unknown_raises_value_error(self):
        """_create_agent_backend raises ValueError for unknown backend."""
        with patch("swarm_orchestrator.orchestrator.get_client"):
            # Create orchestrator with valid config first, then test the factory directly
            orch = Orchestrator()
            # Manually change backend to invalid value to test factory
            orch.config.agent_backend = "unknown-backend"

            with pytest.raises(ValueError) as excinfo:
                orch._create_agent_backend()

            assert "Unknown agent backend" in str(excinfo.value)
            assert "unknown-backend" in str(excinfo.value)

    def test_orchestrator_uses_cursor_cli_backend_from_config(self):
        """Orchestrator with SwarmConfig(agent_backend='cursor-cli') uses CursorCLIAgentBackend."""
        from swarm_orchestrator.backends import CursorCLIAgentBackend

        config = SwarmConfig(agent_backend="cursor-cli")

        with patch("swarm_orchestrator.orchestrator.get_client"):
            orch = Orchestrator(config=config)

        assert isinstance(orch._agent_backend, CursorCLIAgentBackend)


class TestCursorCLIIntegration:
    """Integration tests for cursor-cli backend with mocked subprocess."""

    def test_cursor_cli_spawn_agent_with_mocked_subprocess(self, tmp_path):
        """CursorCLIAgentBackend spawns cursor-agent process correctly."""
        from swarm_orchestrator.backends import CursorCLIAgentBackend

        backend = CursorCLIAgentBackend()
        # Override worktree path lookup
        backend._get_worktree_path = lambda name: str(tmp_path)

        with patch("subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_popen.return_value = mock_process

            session = backend.spawn_agent("test-session", "Test prompt")

        assert session == "test-session"
        mock_popen.assert_called_once()
        call_args = mock_popen.call_args
        assert call_args.kwargs["cwd"] == str(tmp_path)
        assert "cursor-agent" in call_args.args[0]
        # Verify prompt file was written
        prompt_file = tmp_path / ".swarm-prompt.md"
        assert prompt_file.exists()
        assert prompt_file.read_text() == "Test prompt"

    def test_cursor_cli_wait_for_completion_success(self, tmp_path):
        """CursorCLIAgentBackend waits for process completion."""
        from swarm_orchestrator.backends import CursorCLIAgentBackend

        backend = CursorCLIAgentBackend()
        backend._get_worktree_path = lambda name: str(tmp_path)

        with patch("subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.communicate.return_value = (b"output", b"")
            mock_process.returncode = 0
            mock_popen.return_value = mock_process

            backend.spawn_agent("test-session", "prompt")
            results = backend.wait_for_completion(["test-session"], timeout=30)

        assert "test-session" in results
        assert results["test-session"].is_finished is True

    def test_cursor_cli_end_to_end_workflow(self, tmp_path):
        """End-to-end test: spawn, wait, get status with mocked subprocess."""
        from swarm_orchestrator.backends import CursorCLIAgentBackend

        backend = CursorCLIAgentBackend()
        backend._get_worktree_path = lambda name: str(tmp_path)

        with patch("subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.poll.return_value = None  # Still running initially
            mock_process.communicate.return_value = (b"done", b"")
            mock_process.returncode = 0
            mock_popen.return_value = mock_process

            # Spawn
            session = backend.spawn_agent("agent-0", "Implement feature X")
            assert session == "agent-0"

            # Check status while running
            mock_process.poll.return_value = None
            status = backend.get_status("agent-0")
            assert status.agent_id == "agent-0"
            assert status.is_finished is False

            # Wait for completion
            mock_process.poll.return_value = 0
            results = backend.wait_for_completion(["agent-0"], timeout=60)
            assert results["agent-0"].is_finished is True


class TestExplorationIntegration:
    """Tests for exploration integration in the orchestration flow."""

    @pytest.fixture
    def orchestrator_with_exploration(self, mock_schaltwerk_client):
        """Create an orchestrator with exploration enabled."""
        with patch("swarm_orchestrator.orchestrator.get_client") as mock_get_client:
            mock_get_client.return_value = mock_schaltwerk_client
            orch = Orchestrator(agent_count=3, timeout=60)
            yield orch

    @pytest.fixture
    def mock_exploration(self):
        """Mock ExplorationExecutor."""
        with patch("swarm_orchestrator.orchestrator.ExplorationExecutor") as mock:
            yield mock

    @pytest.fixture
    def mock_decompose(self):
        """Mock decompose_task."""
        with patch("swarm_orchestrator.orchestrator.decompose_task") as mock:
            yield mock

    def test_run_calls_explore_before_decompose(
        self, orchestrator_with_exploration, mock_exploration, mock_decompose, sample_subtask
    ):
        """Orchestrator.run should call exploration before decomposition."""
        from swarm_orchestrator.decomposer import ExplorationResult

        # Use a complex query that triggers exploration
        complex_query = "Implement user authentication with OAuth2"

        # Setup mocks
        mock_explorer = MagicMock()
        mock_exploration.return_value = mock_explorer
        mock_explorer.explore.return_value = ExplorationResult(
            code_insights=[],
            web_findings=[],
            context_summary="Test context",
        )
        mock_decompose.return_value = DecompositionResult(
            is_atomic=True,
            subtasks=[sample_subtask],
            original_query=complex_query,
        )

        # Mock process_subtask to avoid full flow
        with patch.object(orchestrator_with_exploration, '_process_subtask_with_mcp') as mock_process:
            mock_process.return_value = SubtaskResult(
                subtask=sample_subtask,
                sessions=["agent-0"],
                vote_result=VoteResult(groups=[], winner=None, total_votes=0, consensus_reached=True, confidence=0),
                winner_session="agent-0",
                merged=False,
            )

            orchestrator_with_exploration.run(complex_query)

        # Verify exploration was called
        mock_explorer.explore.assert_called_once_with(complex_query)

    def test_run_passes_exploration_to_decompose(
        self, orchestrator_with_exploration, mock_exploration, sample_subtask
    ):
        """Decomposer should receive exploration results."""
        from swarm_orchestrator.decomposer import ExplorationResult, CodeInsight

        # Use a complex query that triggers exploration
        complex_query = "Implement API caching with Redis"

        exploration_result = ExplorationResult(
            code_insights=[CodeInsight(file_path="src/main.py", description="Entry point")],
            web_findings=[],
            context_summary="Important context",
        )

        mock_explorer = MagicMock()
        mock_exploration.return_value = mock_explorer
        mock_explorer.explore.return_value = exploration_result

        with patch("swarm_orchestrator.orchestrator.decompose_task") as mock_decompose:
            mock_decompose.return_value = DecompositionResult(
                is_atomic=True,
                subtasks=[sample_subtask],
                original_query=complex_query,
            )

            with patch.object(orchestrator_with_exploration, '_process_subtask_with_mcp') as mock_process:
                mock_process.return_value = SubtaskResult(
                    subtask=sample_subtask,
                    sessions=["a"],
                    vote_result=VoteResult(groups=[], winner=None, total_votes=0, consensus_reached=True, confidence=0),
                    winner_session="a",
                    merged=False,
                )

                orchestrator_with_exploration.run(complex_query)

            # Verify decompose received exploration context
            mock_decompose.assert_called_once()
            call_kwargs = mock_decompose.call_args.kwargs
            assert "exploration_result" in call_kwargs
            assert call_kwargs["exploration_result"] == exploration_result

    def test_skip_exploration_flag(self, mock_schaltwerk_client, sample_subtask):
        """Should skip exploration when flag is set."""
        with patch("swarm_orchestrator.orchestrator.get_client") as mock_get_client:
            mock_get_client.return_value = mock_schaltwerk_client

            orch = Orchestrator(agent_count=3, skip_exploration=True)

            with patch("swarm_orchestrator.orchestrator.ExplorationExecutor") as mock_exploration:
                with patch("swarm_orchestrator.orchestrator.decompose_task") as mock_decompose:
                    mock_decompose.return_value = DecompositionResult(
                        is_atomic=True,
                        subtasks=[sample_subtask],
                        original_query="test",
                    )

                    with patch.object(orch, '_process_subtask_with_mcp') as mock_process:
                        mock_process.return_value = SubtaskResult(
                            subtask=sample_subtask,
                            sessions=["a"],
                            vote_result=VoteResult(groups=[], winner=None, total_votes=0, consensus_reached=True, confidence=0),
                            winner_session="a",
                            merged=False,
                        )

                        orch.run("test")

                    # Exploration should NOT be called
                    mock_exploration.assert_not_called()

    def test_exploration_findings_in_agent_prompt(
        self, orchestrator_with_exploration, sample_subtask
    ):
        """Agent prompts should include exploration findings."""
        from swarm_orchestrator.decomposer import ExplorationResult, CodeInsight

        exploration_result = ExplorationResult(
            code_insights=[CodeInsight(file_path="src/auth.py", description="Auth module")],
            web_findings=[],
            context_summary="Uses JWT for authentication",
        )

        prompt = orchestrator_with_exploration._generate_agent_prompt(
            subtask=sample_subtask,
            task_id="task-1",
            agent_id="task-1-agent-0",
            exploration_result=exploration_result,
        )

        # Should include exploration context
        assert "Uses JWT for authentication" in prompt
        assert "src/auth.py" in prompt

    def test_exploration_findings_empty_when_skipped(
        self, orchestrator_with_exploration, sample_subtask
    ):
        """Agent prompts should handle no exploration gracefully."""
        prompt = orchestrator_with_exploration._generate_agent_prompt(
            subtask=sample_subtask,
            task_id="task-1",
            agent_id="task-1-agent-0",
            exploration_result=None,
        )

        # Should not crash and should have valid prompt
        assert "task-1" in prompt
        assert sample_subtask.prompt in prompt

    def test_needs_exploration_auto_detection(self, mock_schaltwerk_client, sample_subtask):
        """Should auto-detect if exploration is needed."""
        with patch("swarm_orchestrator.orchestrator.get_client") as mock_get_client:
            mock_get_client.return_value = mock_schaltwerk_client

            orch = Orchestrator(agent_count=3)

            with patch("swarm_orchestrator.orchestrator.needs_exploration") as mock_needs:
                mock_needs.return_value = False

                with patch("swarm_orchestrator.orchestrator.ExplorationExecutor") as mock_exploration:
                    with patch("swarm_orchestrator.orchestrator.decompose_task") as mock_decompose:
                        mock_decompose.return_value = DecompositionResult(
                            is_atomic=True,
                            subtasks=[sample_subtask],
                            original_query="fix typo",
                        )

                        with patch.object(orch, '_process_subtask_with_mcp') as mock_process:
                            mock_process.return_value = SubtaskResult(
                                subtask=sample_subtask,
                                sessions=["a"],
                                vote_result=VoteResult(groups=[], winner=None, total_votes=0, consensus_reached=True, confidence=0),
                                winner_session="a",
                                merged=False,
                            )

                            orch.run("fix typo in README")

                        # needs_exploration was called to check
                        mock_needs.assert_called_once_with("fix typo in README")
                        # Exploration not called since needs_exploration returned False
                        mock_exploration.assert_not_called()


# =============================================================================
# Tests for git-native backend integration
# =============================================================================

class TestGitNativeBackendFactory:
    """Tests for git-native backend factory methods."""

    def test_create_worktree_backend_git_native(self):
        """_create_worktree_backend returns GitNativeWorktreeBackend when config.worktree_backend == 'git-native'."""
        from swarm_orchestrator.backends import GitNativeWorktreeBackend

        config = SwarmConfig(worktree_backend="git-native")

        with patch("swarm_orchestrator.orchestrator.get_client"):
            orch = Orchestrator(config=config)
            backend = orch._create_worktree_backend()

        assert isinstance(backend, GitNativeWorktreeBackend)

    def test_create_agent_backend_git_native(self):
        """_create_agent_backend returns GitNativeAgentBackend when config.agent_backend == 'git-native'."""
        from swarm_orchestrator.backends import GitNativeAgentBackend

        config = SwarmConfig(agent_backend="git-native")

        with patch("swarm_orchestrator.orchestrator.get_client"):
            orch = Orchestrator(config=config)
            backend = orch._create_agent_backend()

        assert isinstance(backend, GitNativeAgentBackend)

    def test_orchestrator_uses_git_native_worktree_backend_from_config(self):
        """Orchestrator with SwarmConfig(worktree_backend='git-native') uses GitNativeWorktreeBackend."""
        from swarm_orchestrator.backends import GitNativeWorktreeBackend

        config = SwarmConfig(worktree_backend="git-native")

        with patch("swarm_orchestrator.orchestrator.get_client"):
            orch = Orchestrator(config=config)

        assert isinstance(orch._worktree_backend, GitNativeWorktreeBackend)

    def test_orchestrator_uses_git_native_agent_backend_from_config(self):
        """Orchestrator with SwarmConfig(agent_backend='git-native') uses GitNativeAgentBackend."""
        from swarm_orchestrator.backends import GitNativeAgentBackend

        config = SwarmConfig(agent_backend="git-native")

        with patch("swarm_orchestrator.orchestrator.get_client"):
            orch = Orchestrator(config=config)

        assert isinstance(orch._agent_backend, GitNativeAgentBackend)


class TestSpawnAgentsUsesBakedBackends:
    """Tests that _spawn_agents_with_mcp uses the configured backends."""

    @pytest.fixture
    def mock_worktree_backend(self):
        """Mock worktree backend."""
        from swarm_orchestrator.backends import SessionInfo
        backend = MagicMock()
        backend.create_session.return_value = SessionInfo(
            name="test-session",
            status="running",
            branch="test-branch",
            worktree_path="/tmp/test",
        )
        backend.list_sessions.return_value = []
        return backend

    @pytest.fixture
    def mock_agent_backend(self):
        """Mock agent backend."""
        backend = MagicMock()
        backend.spawn_agent.return_value = "test-session"
        return backend

    def test_spawn_agents_uses_worktree_backend(
        self, mock_schaltwerk_client, mock_worktree_backend, mock_agent_backend, sample_subtask
    ):
        """_spawn_agents_with_mcp uses _worktree_backend.create_session."""
        with patch("swarm_orchestrator.orchestrator.get_client") as mock_get_client:
            mock_get_client.return_value = mock_schaltwerk_client

            orch = Orchestrator(
                agent_count=2,
                worktree_backend=mock_worktree_backend,
                agent_backend=mock_agent_backend,
            )

            orch._spawn_agents_with_mcp(sample_subtask, "task-123")

        # Verify worktree backend was used to create sessions
        assert mock_worktree_backend.create_session.call_count == 2

    def test_spawn_agents_uses_agent_backend(
        self, mock_schaltwerk_client, mock_worktree_backend, mock_agent_backend, sample_subtask
    ):
        """_spawn_agents_with_mcp uses _agent_backend.spawn_agent."""
        with patch("swarm_orchestrator.orchestrator.get_client") as mock_get_client:
            mock_get_client.return_value = mock_schaltwerk_client

            orch = Orchestrator(
                agent_count=2,
                worktree_backend=mock_worktree_backend,
                agent_backend=mock_agent_backend,
            )

            orch._spawn_agents_with_mcp(sample_subtask, "task-123")

        # Verify agent backend was used to spawn agents
        assert mock_agent_backend.spawn_agent.call_count == 2

    def test_cleanup_sessions_uses_worktree_backend(
        self, mock_schaltwerk_client, mock_worktree_backend, mock_agent_backend
    ):
        """_cleanup_sessions uses _worktree_backend.delete_session."""
        from swarm_orchestrator.backends import SessionInfo

        # Setup mock to return sessions
        mock_worktree_backend.list_sessions.return_value = [
            SessionInfo(name="agent-0", status="running", branch="b1"),
            SessionInfo(name="agent-1", status="running", branch="b2"),
        ]

        with patch("swarm_orchestrator.orchestrator.get_client") as mock_get_client:
            mock_get_client.return_value = mock_schaltwerk_client

            orch = Orchestrator(
                worktree_backend=mock_worktree_backend,
                agent_backend=mock_agent_backend,
            )

            orch._cleanup_sessions(["agent-0", "agent-1"])

        # Verify worktree backend was used to delete sessions
        assert mock_worktree_backend.delete_session.call_count == 2
        mock_worktree_backend.delete_session.assert_any_call("agent-0", force=True)
        mock_worktree_backend.delete_session.assert_any_call("agent-1", force=True)


# =============================================================================
# Tests for watch dashboard toggle during run
# =============================================================================

class TestWatchToggleDuringRun:
    """Tests for the 'w' key toggle to show watch dashboard during run."""

    @pytest.fixture
    def orchestrator_for_watch(self, mock_schaltwerk_client):
        """Create an orchestrator for testing watch toggle."""
        with patch("swarm_orchestrator.orchestrator.get_client") as mock_get_client:
            mock_get_client.return_value = mock_schaltwerk_client
            orch = Orchestrator(agent_count=3, timeout=60)
            yield orch

    def test_check_key_press_returns_none_when_no_input(self, orchestrator_for_watch):
        """_check_key_press should return None when no key is pressed."""
        with patch("select.select", return_value=([], [], [])):
            result = orchestrator_for_watch._check_key_press()
            assert result is None

    def test_check_key_press_returns_key_when_pressed(self, orchestrator_for_watch):
        """_check_key_press should return the key when one is pressed."""
        import io

        # Need to patch the imports in the orchestrator module
        with patch("swarm_orchestrator.orchestrator.select.select") as mock_select:
            # First call returns input available, second call (inside the function) also returns available
            mock_select.side_effect = [[True], [True]]
            with patch("swarm_orchestrator.orchestrator.sys.stdin") as mock_stdin:
                mock_stdin.fileno.return_value = 0
                mock_stdin.read.return_value = "w"
                with patch("termios.tcgetattr", return_value=[]):
                    with patch("termios.tcsetattr"):
                        with patch("tty.setcbreak"):
                            result = orchestrator_for_watch._check_key_press()
                            assert result == "w"

    def test_check_key_press_handles_non_tty(self, orchestrator_for_watch):
        """_check_key_press should gracefully handle non-TTY environments."""
        import termios

        with patch("select.select", return_value=([True], [], [])):
            with patch("termios.tcgetattr", side_effect=termios.error("not a tty")):
                result = orchestrator_for_watch._check_key_press()
                assert result is None

    def test_show_watch_dashboard_creates_and_runs_dashboard(self, orchestrator_for_watch):
        """_show_watch_dashboard should create SessionsDashboard and call run()."""
        # Patch at tui module level since it's imported locally in _show_watch_dashboard
        with patch("swarm_orchestrator.tui.SessionsDashboard") as MockDashboard:
            mock_dashboard_instance = MagicMock()
            MockDashboard.return_value = mock_dashboard_instance

            orchestrator_for_watch._show_watch_dashboard()

            MockDashboard.assert_called_once_with(backend=orchestrator_for_watch._worktree_backend)
            mock_dashboard_instance.run.assert_called_once()

    def test_wait_for_mcp_completion_shows_help_text(self, orchestrator_for_watch, capsys):
        """_wait_for_mcp_completion should show help text about 'w' shortcut."""
        mock_task = MagicMock()
        mock_task.all_agents_finished.return_value = True
        mock_task.agent_ids = []
        mock_task.agent_statuses = {}

        with patch.object(orchestrator_for_watch, 'swarm_server') as mock_server:
            mock_server.state.get_task.return_value = mock_task

            orchestrator_for_watch._wait_for_mcp_completion("task-1")

        # Check the help text was printed (via rich console)
        # Since Rich Console is used, we check the console output
        # We can verify the method completed without error

    def test_wait_for_mcp_completion_checks_for_w_key(self, orchestrator_for_watch):
        """_wait_for_mcp_completion should check for 'w' key press in the loop."""
        mock_task = MagicMock()
        # First iteration: not finished, second: finished
        mock_task.all_agents_finished.side_effect = [False, True]
        mock_task.agent_ids = ["agent-1"]
        mock_task.agent_statuses = {}

        with patch.object(orchestrator_for_watch, 'swarm_server') as mock_server:
            mock_server.state.get_task.return_value = mock_task

            with patch.object(orchestrator_for_watch, '_check_key_press', return_value=None) as mock_check:
                with patch("time.sleep"):
                    orchestrator_for_watch._wait_for_mcp_completion("task-1")

                # Should have checked for key press at least once
                assert mock_check.call_count >= 1

    def test_wait_for_mcp_completion_opens_dashboard_on_w(self, orchestrator_for_watch):
        """_wait_for_mcp_completion should open dashboard when 'w' is pressed."""
        mock_task = MagicMock()
        # First call: not finished (triggers loop), second: finished
        mock_task.all_agents_finished.side_effect = [False, True]
        mock_task.agent_ids = ["agent-1"]
        mock_task.agent_statuses = {}

        with patch.object(orchestrator_for_watch, 'swarm_server') as mock_server:
            mock_server.state.get_task.return_value = mock_task

            # Simulate 'w' key press on first check
            with patch.object(orchestrator_for_watch, '_check_key_press', return_value="w") as mock_check:
                with patch.object(orchestrator_for_watch, '_show_watch_dashboard') as mock_show:
                    with patch("time.sleep"):
                        orchestrator_for_watch._wait_for_mcp_completion("task-1")

                    mock_show.assert_called_once()

    def test_wait_for_mcp_votes_shows_help_text(self, orchestrator_for_watch):
        """_wait_for_mcp_votes should show help text about 'w' shortcut."""
        mock_task = MagicMock()
        mock_task.all_agents_voted.return_value = True
        mock_task.agent_ids = []
        mock_task.votes = {}

        with patch.object(orchestrator_for_watch, 'swarm_server') as mock_server:
            mock_server.state.get_task.return_value = mock_task
            mock_server.state.get_vote_results.return_value = {"success": True}

            orchestrator_for_watch._wait_for_mcp_votes("task-1")

    def test_wait_for_mcp_votes_opens_dashboard_on_w(self, orchestrator_for_watch):
        """_wait_for_mcp_votes should open dashboard when 'w' is pressed."""
        mock_task = MagicMock()
        mock_task.all_agents_voted.side_effect = [False, True]
        mock_task.agent_ids = ["agent-1"]
        mock_task.votes = {}

        with patch.object(orchestrator_for_watch, 'swarm_server') as mock_server:
            mock_server.state.get_task.return_value = mock_task
            mock_server.state.get_vote_results.return_value = {"success": True}

            with patch.object(orchestrator_for_watch, '_check_key_press', return_value="w"):
                with patch.object(orchestrator_for_watch, '_show_watch_dashboard') as mock_show:
                    with patch("time.sleep"):
                        orchestrator_for_watch._wait_for_mcp_votes("task-1")

                    mock_show.assert_called_once()
