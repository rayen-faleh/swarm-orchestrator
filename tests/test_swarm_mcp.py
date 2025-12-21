"""
Tests for the Swarm MCP server components.

TDD approach: These tests define the expected behavior of:
- SwarmState: State management for tracking agent completion and votes
- SwarmMCPServer: MCP server with tools for agent coordination
"""

import pytest
from unittest.mock import MagicMock, patch
import json
import tempfile
from pathlib import Path

from swarm_orchestrator.swarm_mcp.state import (
    SwarmState,
    AgentStatus,
    TaskState,
    VoteRecord,
)
from swarm_orchestrator.swarm_mcp.server import SwarmMCPServer
from swarm_orchestrator.swarm_mcp.tools import (
    FinishedWorkTool,
    GetAllImplementationsTool,
    CastVoteTool,
    GetVoteResultsTool,
)


# =============================================================================
# SwarmState Tests
# =============================================================================

class TestAgentStatus:
    """Tests for AgentStatus enum."""

    def test_status_values(self):
        """Should have expected status values."""
        assert AgentStatus.WORKING.value == "working"
        assert AgentStatus.FINISHED.value == "finished"
        assert AgentStatus.VOTED.value == "voted"


class TestTaskState:
    """Tests for TaskState dataclass."""

    def test_create_task_state(self):
        """Should create task state with required fields."""
        state = TaskState(
            task_id="task-1",
            agent_ids=["agent-0", "agent-1", "agent-2"],
            session_names={"agent-0": "session-0", "agent-1": "session-1", "agent-2": "session-2"},
        )

        assert state.task_id == "task-1"
        assert len(state.agent_ids) == 3
        assert state.session_names["agent-0"] == "session-0"
        assert state.agent_statuses == {}
        assert state.votes == {}
        assert state.implementations == {}

    def test_all_agents_finished(self):
        """Should detect when all agents have finished."""
        state = TaskState(
            task_id="task-1",
            agent_ids=["agent-0", "agent-1"],
            session_names={"agent-0": "s0", "agent-1": "s1"},
        )

        assert state.all_agents_finished() is False

        state.agent_statuses["agent-0"] = AgentStatus.FINISHED
        assert state.all_agents_finished() is False

        state.agent_statuses["agent-1"] = AgentStatus.FINISHED
        assert state.all_agents_finished() is True

    def test_all_agents_voted(self):
        """Should detect when all agents have voted."""
        state = TaskState(
            task_id="task-1",
            agent_ids=["agent-0", "agent-1"],
            session_names={"agent-0": "s0", "agent-1": "s1"},
        )

        assert state.all_agents_voted() is False

        state.agent_statuses["agent-0"] = AgentStatus.VOTED
        assert state.all_agents_voted() is False

        state.agent_statuses["agent-1"] = AgentStatus.VOTED
        assert state.all_agents_voted() is True

    def test_get_vote_counts(self):
        """Should count votes correctly."""
        state = TaskState(
            task_id="task-1",
            agent_ids=["agent-0", "agent-1", "agent-2"],
            session_names={},
        )

        state.votes = {
            "agent-0": VoteRecord(voter="agent-0", voted_for="agent-1", reason="Better approach"),
            "agent-1": VoteRecord(voter="agent-1", voted_for="agent-1", reason="Self vote"),
            "agent-2": VoteRecord(voter="agent-2", voted_for="agent-0", reason="Cleaner code"),
        }

        counts = state.get_vote_counts()

        assert counts["agent-1"] == 2
        assert counts["agent-0"] == 1
        assert counts.get("agent-2", 0) == 0

    def test_get_winner(self):
        """Should return winner with most votes."""
        state = TaskState(
            task_id="task-1",
            agent_ids=["agent-0", "agent-1", "agent-2"],
            session_names={},
        )

        # No votes yet
        assert state.get_winner() is None

        state.votes = {
            "agent-0": VoteRecord(voter="agent-0", voted_for="agent-2", reason=""),
            "agent-1": VoteRecord(voter="agent-1", voted_for="agent-2", reason=""),
            "agent-2": VoteRecord(voter="agent-2", voted_for="agent-0", reason=""),
        }

        winner = state.get_winner()
        assert winner == "agent-2"


class TestSwarmState:
    """Tests for SwarmState manager."""

    def test_create_task(self):
        """Should create a new task with agents."""
        state = SwarmState()

        task = state.create_task(
            task_id="task-1",
            agent_ids=["agent-0", "agent-1", "agent-2", "agent-3", "agent-4"],
            session_names={
                "agent-0": "task-1-agent-0",
                "agent-1": "task-1-agent-1",
                "agent-2": "task-1-agent-2",
                "agent-3": "task-1-agent-3",
                "agent-4": "task-1-agent-4",
            },
        )

        assert task.task_id == "task-1"
        assert len(task.agent_ids) == 5
        assert state.get_task("task-1") is not None

    def test_get_nonexistent_task(self):
        """Should return None for unknown task."""
        state = SwarmState()
        assert state.get_task("unknown") is None

    def test_mark_agent_finished(self):
        """Should mark agent as finished and store implementation."""
        state = SwarmState()
        state.create_task(
            task_id="task-1",
            agent_ids=["agent-0"],
            session_names={"agent-0": "session-0"},
        )

        result = state.mark_agent_finished(
            task_id="task-1",
            agent_id="agent-0",
            implementation="def hello(): pass",
        )

        assert result["success"] is True
        assert result["agents_remaining"] == 0

        task = state.get_task("task-1")
        assert task.agent_statuses["agent-0"] == AgentStatus.FINISHED
        # Implementation is now an ImplementationSummary object
        impl = task.implementations["agent-0"]
        assert impl.full_diff == "def hello(): pass"
        assert impl.agent_id == "agent-0"

    def test_mark_agent_finished_counts_remaining(self):
        """Should correctly count remaining agents."""
        state = SwarmState()
        state.create_task(
            task_id="task-1",
            agent_ids=["agent-0", "agent-1", "agent-2"],
            session_names={},
        )

        result = state.mark_agent_finished("task-1", "agent-0", "impl-0")
        assert result["agents_remaining"] == 2

        result = state.mark_agent_finished("task-1", "agent-1", "impl-1")
        assert result["agents_remaining"] == 1

        result = state.mark_agent_finished("task-1", "agent-2", "impl-2")
        assert result["agents_remaining"] == 0
        assert result["all_finished"] is True

    def test_mark_agent_finished_invalid_task(self):
        """Should return error for invalid task."""
        state = SwarmState()

        result = state.mark_agent_finished("unknown", "agent-0", "impl")

        assert result["success"] is False
        assert "not found" in result["error"]

    def test_mark_agent_finished_invalid_agent(self):
        """Should return error for invalid agent."""
        state = SwarmState()
        state.create_task("task-1", ["agent-0"], {})

        result = state.mark_agent_finished("task-1", "agent-99", "impl")

        assert result["success"] is False
        assert "not found" in result["error"]

    def test_get_all_implementations(self):
        """Should return all implementations with structured summaries."""
        state = SwarmState()
        state.create_task(
            task_id="task-1",
            agent_ids=["agent-0", "agent-1"],
            session_names={"agent-0": "session-0", "agent-1": "session-1"},
        )

        # Use realistic diff format
        diff_0 = "diff --git a/foo.py b/foo.py\n+def foo(): return 1"
        diff_1 = "diff --git a/foo.py b/foo.py\n+def foo(): return 2"
        state.mark_agent_finished("task-1", "agent-0", diff_0)
        state.mark_agent_finished("task-1", "agent-1", diff_1)

        result = state.get_all_implementations("task-1")

        assert result["success"] is True
        assert len(result["implementations"]) == 2

        impl_0 = next(i for i in result["implementations"] if i["agent_id"] == "agent-0")
        # New format has summary and condensed_diff
        assert impl_0["session_name"] == "session-0"
        assert "summary" in impl_0
        assert "condensed_diff" in impl_0
        assert "foo.py" in impl_0["summary"]["files_changed"]
        assert "+def foo(): return 1" in impl_0["condensed_diff"]

    def test_get_implementations_not_all_finished(self):
        """Should return error if not all agents finished."""
        state = SwarmState()
        state.create_task("task-1", ["agent-0", "agent-1"], {})
        state.mark_agent_finished("task-1", "agent-0", "impl-0")

        result = state.get_all_implementations("task-1")

        assert result["success"] is False
        assert "finished" in result["error"].lower()

    def test_cast_vote(self):
        """Should record a vote."""
        state = SwarmState()
        state.create_task("task-1", ["agent-0", "agent-1", "agent-2"], {})

        # All must finish before voting
        state.mark_agent_finished("task-1", "agent-0", "impl-0")
        state.mark_agent_finished("task-1", "agent-1", "impl-1")
        state.mark_agent_finished("task-1", "agent-2", "impl-2")

        result = state.cast_vote(
            task_id="task-1",
            agent_id="agent-0",
            voted_for="agent-1",
            reason="Better implementation",
        )

        assert result["success"] is True
        assert result["votes_remaining"] == 2

        task = state.get_task("task-1")
        assert task.votes["agent-0"].voted_for == "agent-1"

    def test_cast_vote_before_all_finished(self):
        """Should reject vote if not all agents finished."""
        state = SwarmState()
        state.create_task("task-1", ["agent-0", "agent-1"], {})
        state.mark_agent_finished("task-1", "agent-0", "impl-0")

        result = state.cast_vote("task-1", "agent-0", "agent-1", "reason")

        assert result["success"] is False
        assert "finished" in result["error"].lower()

    def test_cast_vote_invalid_target(self):
        """Should reject vote for non-existent agent."""
        state = SwarmState()
        state.create_task("task-1", ["agent-0", "agent-1"], {})
        state.mark_agent_finished("task-1", "agent-0", "impl-0")
        state.mark_agent_finished("task-1", "agent-1", "impl-1")

        result = state.cast_vote("task-1", "agent-0", "agent-99", "reason")

        assert result["success"] is False
        assert "invalid" in result["error"].lower()

    def test_cast_vote_self_vote_rejected(self):
        """Should reject self-votes to ensure fairness."""
        state = SwarmState()
        state.create_task("task-1", ["agent-0", "agent-1"], {})
        state.mark_agent_finished("task-1", "agent-0", "impl-0")
        state.mark_agent_finished("task-1", "agent-1", "impl-1")

        result = state.cast_vote("task-1", "agent-0", "agent-0", "I'm the best!")

        assert result["success"] is False
        assert "cannot vote for yourself" in result["error"].lower()

    def test_cast_vote_duplicate(self):
        """Should reject duplicate votes."""
        state = SwarmState()
        state.create_task("task-1", ["agent-0", "agent-1"], {})
        state.mark_agent_finished("task-1", "agent-0", "impl-0")
        state.mark_agent_finished("task-1", "agent-1", "impl-1")

        state.cast_vote("task-1", "agent-0", "agent-1", "first vote")
        result = state.cast_vote("task-1", "agent-0", "agent-1", "change vote")

        assert result["success"] is False
        assert "already voted" in result["error"]

    def test_get_vote_results(self):
        """Should return vote results with winner."""
        state = SwarmState()
        state.create_task("task-1", ["agent-0", "agent-1", "agent-2"], {})

        # All finish
        for i in range(3):
            state.mark_agent_finished("task-1", f"agent-{i}", f"impl-{i}")

        # All vote
        state.cast_vote("task-1", "agent-0", "agent-2", "")
        state.cast_vote("task-1", "agent-1", "agent-2", "")
        state.cast_vote("task-1", "agent-2", "agent-0", "")

        result = state.get_vote_results("task-1")

        assert result["success"] is True
        assert result["all_voted"] is True
        assert result["winner"] == "agent-2"
        assert result["vote_counts"]["agent-2"] == 2
        assert result["vote_counts"]["agent-0"] == 1

    def test_get_vote_results_partial(self):
        """Should return partial results if not all voted."""
        state = SwarmState()
        state.create_task("task-1", ["agent-0", "agent-1", "agent-2"], {})

        for i in range(3):
            state.mark_agent_finished("task-1", f"agent-{i}", f"impl-{i}")

        state.cast_vote("task-1", "agent-0", "agent-1", "")

        result = state.get_vote_results("task-1")

        assert result["success"] is True
        assert result["all_voted"] is False
        assert result["votes_cast"] == 1
        assert result["votes_remaining"] == 2

    def test_persistence_save_load(self):
        """Should persist and restore state with structured implementations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"

            # Create and populate state
            state1 = SwarmState(persistence_path=str(state_file))
            state1.create_task("task-1", ["agent-0", "agent-1"], {"agent-0": "s0", "agent-1": "s1"})
            state1.mark_agent_finished("task-1", "agent-0", "diff --git a/file.py b/file.py\n+impl-0")
            state1.save()

            # Load in new instance
            state2 = SwarmState(persistence_path=str(state_file))
            state2.load()

            task = state2.get_task("task-1")
            assert task is not None
            # Implementation is now an ImplementationSummary object
            impl = task.implementations["agent-0"]
            assert impl.full_diff == "diff --git a/file.py b/file.py\n+impl-0"
            assert impl.agent_id == "agent-0"
            assert "file.py" in impl.stats.files_changed
            assert task.agent_statuses["agent-0"] == AgentStatus.FINISHED


# =============================================================================
# MCP Tools Tests
# =============================================================================

class TestFinishedWorkTool:
    """Tests for finished_work MCP tool."""

    @pytest.fixture
    def state_with_task(self):
        """Create state with a task."""
        state = SwarmState()
        state.create_task(
            "task-1",
            ["agent-0", "agent-1"],
            {"agent-0": "session-0", "agent-1": "session-1"},
        )
        return state

    def test_execute_marks_finished(self, state_with_task):
        """Should mark agent as finished."""
        tool = FinishedWorkTool(state_with_task)

        result = tool.execute({
            "task_id": "task-1",
            "agent_id": "agent-0",
            "implementation": "def solve(): pass",
        })

        assert result["success"] is True
        task = state_with_task.get_task("task-1")
        assert task.agent_statuses["agent-0"] == AgentStatus.FINISHED

    def test_execute_missing_params(self, state_with_task):
        """Should return error for missing parameters."""
        tool = FinishedWorkTool(state_with_task)

        result = tool.execute({"task_id": "task-1"})

        assert result["success"] is False
        assert "required" in result["error"].lower()


class TestGetAllImplementationsTool:
    """Tests for get_all_implementations MCP tool."""

    @pytest.fixture
    def state_all_finished(self):
        """Create state with all agents finished."""
        state = SwarmState()
        state.create_task(
            "task-1",
            ["agent-0", "agent-1"],
            {"agent-0": "session-0", "agent-1": "session-1"},
        )
        state.mark_agent_finished("task-1", "agent-0", "impl-0")
        state.mark_agent_finished("task-1", "agent-1", "impl-1")
        return state

    def test_execute_returns_implementations(self, state_all_finished):
        """Should return all implementations."""
        tool = GetAllImplementationsTool(state_all_finished)

        result = tool.execute({"task_id": "task-1"})

        assert result["success"] is True
        assert len(result["implementations"]) == 2


class TestCastVoteTool:
    """Tests for cast_vote MCP tool."""

    @pytest.fixture
    def state_ready_to_vote(self):
        """Create state ready for voting."""
        state = SwarmState()
        state.create_task(
            "task-1",
            ["agent-0", "agent-1", "agent-2"],
            {},
        )
        for i in range(3):
            state.mark_agent_finished("task-1", f"agent-{i}", f"impl-{i}")
        return state

    def test_execute_records_vote(self, state_ready_to_vote):
        """Should record the vote."""
        tool = CastVoteTool(state_ready_to_vote)

        result = tool.execute({
            "task_id": "task-1",
            "agent_id": "agent-0",
            "voted_for": "agent-2",
            "reason": "Best solution",
        })

        assert result["success"] is True
        task = state_ready_to_vote.get_task("task-1")
        assert task.votes["agent-0"].voted_for == "agent-2"


class TestGetVoteResultsTool:
    """Tests for get_vote_results MCP tool."""

    def test_execute_returns_results(self):
        """Should return vote results."""
        state = SwarmState()
        # Need 3 agents since agents can't vote for themselves
        state.create_task("task-1", ["agent-0", "agent-1", "agent-2"], {})
        state.mark_agent_finished("task-1", "agent-0", "impl-0")
        state.mark_agent_finished("task-1", "agent-1", "impl-1")
        state.mark_agent_finished("task-1", "agent-2", "impl-2")
        # All vote for agent-1 (no self-votes allowed)
        state.cast_vote("task-1", "agent-0", "agent-1", "")
        state.cast_vote("task-1", "agent-1", "agent-0", "")  # agent-1 can't vote for itself
        state.cast_vote("task-1", "agent-2", "agent-1", "")

        tool = GetVoteResultsTool(state)
        result = tool.execute({"task_id": "task-1"})

        assert result["success"] is True
        assert result["winner"] == "agent-1"  # 2 votes vs 1 vote


# =============================================================================
# MCP Server Tests
# =============================================================================

class TestSwarmMCPServer:
    """Tests for SwarmMCPServer."""

    def test_list_tools(self):
        """Should list all available tools."""
        server = SwarmMCPServer()

        tools = server.list_tools()

        tool_names = [t["name"] for t in tools]
        assert "finished_work" in tool_names
        assert "get_all_implementations" in tool_names
        assert "cast_vote" in tool_names
        assert "get_vote_results" in tool_names

    def test_call_tool_finished_work(self):
        """Should handle finished_work tool call."""
        server = SwarmMCPServer()
        server.state.create_task("task-1", ["agent-0"], {"agent-0": "s0"})

        result = server.call_tool("finished_work", {
            "task_id": "task-1",
            "agent_id": "agent-0",
            "implementation": "def solve(): pass",
        })

        assert result["success"] is True

    def test_call_tool_unknown(self):
        """Should return error for unknown tool."""
        server = SwarmMCPServer()

        result = server.call_tool("unknown_tool", {})

        assert result["success"] is False
        assert "unknown" in result["error"].lower()

    def test_create_task_for_agents(self):
        """Should create task with proper agent setup."""
        server = SwarmMCPServer()

        task = server.create_task(
            task_id="task-1",
            agent_count=5,
            session_prefix="task-1",
        )

        assert len(task.agent_ids) == 5
        assert task.agent_ids[0] == "task-1-agent-0"
        assert task.session_names["task-1-agent-0"] == "task-1-agent-0"

    def test_handle_jsonrpc_request(self):
        """Should handle JSON-RPC 2.0 requests."""
        server = SwarmMCPServer()
        server.state.create_task("task-1", ["agent-0"], {"agent-0": "s0"})

        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "finished_work",
                "arguments": {
                    "task_id": "task-1",
                    "agent_id": "agent-0",
                    "implementation": "impl",
                },
            },
        }

        response = server.handle_request(request)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert "result" in response

    def test_handle_tools_list_request(self):
        """Should handle tools/list request."""
        server = SwarmMCPServer()

        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {},
        }

        response = server.handle_request(request)

        assert "result" in response
        assert "tools" in response["result"]


# =============================================================================
# Integration Tests
# =============================================================================

class TestSwarmMCPIntegration:
    """Integration tests for the full workflow."""

    def test_full_workflow(self):
        """Test complete workflow: create -> finish -> get impls -> vote -> results."""
        server = SwarmMCPServer()

        # 1. Create task with 3 agents
        task = server.create_task("task-1", agent_count=3, session_prefix="task-1")

        # 2. All agents finish their work
        for i in range(3):
            result = server.call_tool("finished_work", {
                "task_id": "task-1",
                "agent_id": f"task-1-agent-{i}",
                "implementation": f"def solve(): return {i}",
            })
            assert result["success"] is True

        # 3. Agent 0 gets all implementations
        result = server.call_tool("get_all_implementations", {"task_id": "task-1"})
        assert result["success"] is True
        assert len(result["implementations"]) == 3

        # 4. All agents vote (agent-2 gets majority)
        server.call_tool("cast_vote", {
            "task_id": "task-1",
            "agent_id": "task-1-agent-0",
            "voted_for": "task-1-agent-2",
            "reason": "Most elegant solution",
        })
        server.call_tool("cast_vote", {
            "task_id": "task-1",
            "agent_id": "task-1-agent-1",
            "voted_for": "task-1-agent-2",
            "reason": "Best approach",
        })
        server.call_tool("cast_vote", {
            "task_id": "task-1",
            "agent_id": "task-1-agent-2",
            "voted_for": "task-1-agent-0",
            "reason": "Self-vote for second best",
        })

        # 5. Get results
        result = server.call_tool("get_vote_results", {"task_id": "task-1"})

        assert result["success"] is True
        assert result["all_voted"] is True
        assert result["winner"] == "task-1-agent-2"
        assert result["vote_counts"]["task-1-agent-2"] == 2

    def test_workflow_with_persistence(self):
        """Test workflow with state persistence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"

            # Create server and task
            server1 = SwarmMCPServer(persistence_path=str(state_file))
            server1.create_task("task-1", agent_count=2, session_prefix="task-1")

            # Agent 0 finishes
            server1.call_tool("finished_work", {
                "task_id": "task-1",
                "agent_id": "task-1-agent-0",
                "implementation": "impl-0",
            })

            # Simulate restart
            server2 = SwarmMCPServer(persistence_path=str(state_file))

            # Agent 1 finishes in new server instance
            result = server2.call_tool("finished_work", {
                "task_id": "task-1",
                "agent_id": "task-1-agent-1",
                "implementation": "impl-1",
            })

            assert result["success"] is True
            assert result["all_finished"] is True


# =============================================================================
# Compression Integration Tests
# =============================================================================

class TestCompressionInGetAllImplementations:
    """Tests for compression in get_all_implementations."""

    def test_compression_enabled_compresses_large_diffs(self):
        """Large diffs should be compressed when enabled."""
        from swarm_orchestrator.config import SwarmConfig

        config = SwarmConfig(
            enable_diff_compression=True,
            compression_min_tokens=50,  # Low threshold for testing
            compression_target_ratio=0.5,
        )
        state = SwarmState(compression_config=config)
        state.create_task("task-1", ["agent-0"], {"agent-0": "session-0"})

        # Create a large diff that exceeds threshold
        large_diff = "diff --git a/foo.py b/foo.py\n" + "\n".join([
            f"+# line {i} with extra verbose content words here" for i in range(100)
        ])
        state.mark_agent_finished("task-1", "agent-0", large_diff)

        result = state.get_all_implementations("task-1")

        assert result["success"] is True
        impl = result["implementations"][0]
        assert "condensed_diff" in impl
        # Compression stats should be present
        assert "compression" in impl
        assert "original_tokens" in impl["compression"]
        assert "compressed_tokens" in impl["compression"]

    def test_compression_disabled_returns_original(self):
        """Diffs should not be compressed when disabled."""
        from swarm_orchestrator.config import SwarmConfig

        config = SwarmConfig(enable_diff_compression=False)
        state = SwarmState(compression_config=config)
        state.create_task("task-1", ["agent-0"], {"agent-0": "session-0"})

        diff = "diff --git a/foo.py b/foo.py\n+def foo(): return 1"
        state.mark_agent_finished("task-1", "agent-0", diff)

        result = state.get_all_implementations("task-1")

        assert result["success"] is True
        impl = result["implementations"][0]
        # Original condensed diff should be returned
        assert "+def foo(): return 1" in impl["condensed_diff"]
        # No compression stats when disabled
        assert impl.get("compression") is None

    def test_small_diffs_bypass_compression(self):
        """Diffs below threshold should bypass compression."""
        from swarm_orchestrator.config import SwarmConfig

        config = SwarmConfig(
            enable_diff_compression=True,
            compression_min_tokens=500,  # Default high threshold
        )
        state = SwarmState(compression_config=config)
        state.create_task("task-1", ["agent-0"], {"agent-0": "session-0"})

        small_diff = "diff --git a/foo.py b/foo.py\n+x=1"
        state.mark_agent_finished("task-1", "agent-0", small_diff)

        result = state.get_all_implementations("task-1")

        assert result["success"] is True
        impl = result["implementations"][0]
        # Original diff preserved
        assert "+x=1" in impl["condensed_diff"]
        # No compression applied (below threshold)
        assert impl.get("compression") is None

    def test_state_without_compression_config_works(self):
        """SwarmState without compression config uses defaults."""
        state = SwarmState()  # No config
        state.create_task("task-1", ["agent-0"], {"agent-0": "session-0"})

        diff = "diff --git a/foo.py b/foo.py\n+def foo(): pass"
        state.mark_agent_finished("task-1", "agent-0", diff)

        result = state.get_all_implementations("task-1")

        assert result["success"] is True
        assert len(result["implementations"]) == 1


# =============================================================================
# SwarmMCPServer Compression Config Tests
# =============================================================================

class TestSwarmMCPServerCompressionConfig:
    """Tests for SwarmMCPServer accepting compression_config."""

    def test_server_accepts_compression_config(self):
        """SwarmMCPServer should accept compression_config parameter."""
        from swarm_orchestrator.config import SwarmConfig

        config = SwarmConfig(
            enable_diff_compression=True,
            compression_min_tokens=100,
            compression_target_ratio=0.5,
        )
        server = SwarmMCPServer(compression_config=config)

        # Verify config is passed to state
        assert server.state._compression_config is config

    def test_server_works_without_compression_config(self):
        """SwarmMCPServer should work without compression_config."""
        server = SwarmMCPServer()

        # Should work and state should have no compression config
        assert server.state._compression_config is None

    def test_server_with_persistence_and_compression(self):
        """SwarmMCPServer should accept both persistence_path and compression_config."""
        from swarm_orchestrator.config import SwarmConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            config = SwarmConfig(enable_diff_compression=True)

            server = SwarmMCPServer(
                persistence_path=str(state_file),
                compression_config=config,
            )

            assert server.state.persistence_path == str(state_file)
            assert server.state._compression_config is config

    def test_server_compression_integration(self):
        """SwarmMCPServer should use compression when configured."""
        from swarm_orchestrator.config import SwarmConfig

        config = SwarmConfig(
            enable_diff_compression=True,
            compression_min_tokens=10,  # Very low threshold
            compression_target_ratio=0.5,
        )
        server = SwarmMCPServer(compression_config=config)
        server.create_task("task-1", agent_count=1, session_prefix="task-1")

        # Create a diff that exceeds threshold
        large_diff = "diff --git a/foo.py b/foo.py\n" + "\n".join([
            f"+# verbose line number {i} with extra content here" for i in range(50)
        ])
        server.call_tool("finished_work", {
            "task_id": "task-1",
            "agent_id": "task-1-agent-0",
            "implementation": large_diff,
        })

        result = server.call_tool("get_all_implementations", {"task_id": "task-1"})

        assert result["success"] is True
        impl = result["implementations"][0]
        # Compression stats should be present for large diff
        assert "compression" in impl
