# Multi-Agent Consensus System - Implementation Plan

## Vision

A proof-of-concept system that demonstrates the core principles from SwarmAgentic and MAKER research papers applied to code generation tasks. The system decomposes user queries into subtasks, executes them via multiple independent agents, and uses consensus voting to select the most reliable solution.

---

## Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Framework | LangGraph | Best for stateful multi-agent workflows with graph-based orchestration |
| LLM | Qwen3-Coder (local via Ollama) | Cost-effective, strong coding performance, privacy |
| Agent Count | 5 | Odd number guarantees majority consensus (no ties) |
| Consensus Method | Exact string match | Simplest for PoC; code outputs should be deterministic |
| Task Type | Python code generation | Matches model strengths, easy to verify via execution |
| Execution | Sequential | Local model friendly, simpler implementation |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER QUERY                                │
│                "Write a function to check if number is prime"    │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DECOMPOSER NODE                               │
│         Breaks query into atomic subtasks (if needed)            │
│         Output: List[Subtask]                                    │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                 MULTI-AGENT EXECUTOR                             │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │ Agent 1 │ │ Agent 2 │ │ Agent 3 │ │ Agent 4 │ │ Agent 5 │   │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘   │
│       │           │           │           │           │         │
│       └───────────┴─────┬─────┴───────────┴───────────┘         │
│                         │                                        │
│                         ▼                                        │
│              5 Independent Solutions                             │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CONSENSUS NODE                                │
│         - Group identical solutions                              │
│         - Count votes per group                                  │
│         - Select majority (>= 3 votes)                           │
│         - If no majority: return all candidates + warning        │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FINAL OUTPUT                                │
│         - Consensus solution                                     │
│         - Vote distribution                                      │
│         - Confidence score                                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Epic 1: Core Infrastructure

### US-1.1: Project Setup and Dependencies

**As a** developer
**I want** a properly configured Python project with all dependencies
**So that** I can begin implementing the multi-agent system

**Acceptance Criteria:**
- [ ] Python 3.11+ project with `pyproject.toml`
- [ ] LangGraph and LangChain dependencies installed
- [ ] Ollama Python client configured
- [ ] pytest and pytest-asyncio for testing
- [ ] Project runs `pytest` successfully with no errors
- [ ] `.gitignore` configured for Python projects

**TDD Approach:**
```python
# test_setup.py
def test_langgraph_import():
    """Verify LangGraph is installed and importable."""
    from langgraph.graph import StateGraph
    assert StateGraph is not None

def test_ollama_client_import():
    """Verify Ollama client is installed."""
    import ollama
    assert ollama is not None

def test_project_structure():
    """Verify expected project structure exists."""
    from pathlib import Path
    assert Path("src/swarm_agents").is_dir()
    assert Path("tests").is_dir()
```

---

### US-1.2: Ollama Client Wrapper

**As a** developer
**I want** a wrapper around the Ollama client for Qwen3-Coder
**So that** I can easily swap models and handle errors consistently

**Acceptance Criteria:**
- [ ] `OllamaClient` class that wraps ollama library
- [ ] Configurable model name (default: `qwen3-coder`)
- [ ] `generate(prompt: str) -> str` method
- [ ] Proper error handling for connection failures
- [ ] Retry logic with configurable attempts
- [ ] Timeout configuration

**TDD Approach:**
```python
# test_ollama_client.py
import pytest
from unittest.mock import Mock, patch

def test_client_initialization_with_default_model():
    """Client should default to qwen3-coder model."""
    from swarm_agents.llm import OllamaClient
    client = OllamaClient()
    assert client.model == "qwen3-coder"

def test_client_initialization_with_custom_model():
    """Client should accept custom model name."""
    from swarm_agents.llm import OllamaClient
    client = OllamaClient(model="codellama")
    assert client.model == "codellama"

def test_generate_returns_string():
    """Generate should return string response."""
    from swarm_agents.llm import OllamaClient
    with patch('ollama.generate') as mock_gen:
        mock_gen.return_value = {'response': 'def hello(): pass'}
        client = OllamaClient()
        result = client.generate("Write hello function")
        assert isinstance(result, str)
        assert "def hello" in result

def test_generate_handles_connection_error():
    """Should raise clear error on connection failure."""
    from swarm_agents.llm import OllamaClient, OllamaConnectionError
    with patch('ollama.generate') as mock_gen:
        mock_gen.side_effect = ConnectionError("Connection refused")
        client = OllamaClient()
        with pytest.raises(OllamaConnectionError):
            client.generate("test prompt")

def test_generate_retries_on_failure():
    """Should retry specified number of times before failing."""
    from swarm_agents.llm import OllamaClient
    with patch('ollama.generate') as mock_gen:
        mock_gen.side_effect = [TimeoutError(), TimeoutError(), {'response': 'success'}]
        client = OllamaClient(retries=3)
        result = client.generate("test")
        assert result == "success"
        assert mock_gen.call_count == 3
```

---

## Epic 2: Task Decomposition

### US-2.1: Subtask Data Model

**As a** developer
**I want** a clear data model for subtasks
**So that** the system has a consistent structure for passing work between nodes

**Acceptance Criteria:**
- [ ] `Subtask` dataclass with: `id`, `description`, `context`, `dependencies`
- [ ] `DecompositionResult` dataclass with: `original_query`, `subtasks`, `is_atomic`
- [ ] JSON serialization/deserialization support
- [ ] Validation for required fields

**TDD Approach:**
```python
# test_models.py
import pytest
from swarm_agents.models import Subtask, DecompositionResult

def test_subtask_creation():
    """Subtask should be created with required fields."""
    subtask = Subtask(
        id="st-1",
        description="Write is_prime function",
        context="Should return bool"
    )
    assert subtask.id == "st-1"
    assert subtask.dependencies == []

def test_subtask_to_json():
    """Subtask should serialize to JSON."""
    subtask = Subtask(id="st-1", description="test")
    json_str = subtask.to_json()
    assert '"id": "st-1"' in json_str

def test_decomposition_result_with_atomic_task():
    """Atomic tasks should have single subtask matching original."""
    result = DecompositionResult(
        original_query="Write hello function",
        subtasks=[Subtask(id="st-1", description="Write hello function")],
        is_atomic=True
    )
    assert result.is_atomic
    assert len(result.subtasks) == 1

def test_decomposition_result_validation():
    """Should raise error if subtasks list is empty."""
    with pytest.raises(ValueError):
        DecompositionResult(original_query="test", subtasks=[], is_atomic=False)
```

---

### US-2.2: Decomposer Node

**As a** developer
**I want** a decomposer that breaks complex queries into atomic subtasks
**So that** each subtask can be independently solved by agents

**Acceptance Criteria:**
- [ ] `Decomposer` class that takes an `OllamaClient`
- [ ] `decompose(query: str) -> DecompositionResult` method
- [ ] Recognizes atomic tasks (single function) and returns as-is
- [ ] Breaks complex tasks into 2-5 subtasks
- [ ] Each subtask has clear, actionable description
- [ ] Prompt engineering to ensure consistent output format

**TDD Approach:**
```python
# test_decomposer.py
import pytest
from unittest.mock import Mock

def test_atomic_task_not_decomposed():
    """Simple single-function requests should not be split."""
    from swarm_agents.decomposer import Decomposer
    mock_client = Mock()
    mock_client.generate.return_value = '''
    {"is_atomic": true, "subtasks": [{"id": "1", "description": "Write is_prime function"}]}
    '''
    decomposer = Decomposer(mock_client)
    result = decomposer.decompose("Write a function to check if a number is prime")
    assert result.is_atomic
    assert len(result.subtasks) == 1

def test_complex_task_decomposed():
    """Complex tasks should be broken into multiple subtasks."""
    from swarm_agents.decomposer import Decomposer
    mock_client = Mock()
    mock_client.generate.return_value = '''
    {"is_atomic": false, "subtasks": [
        {"id": "1", "description": "Write function to read CSV file"},
        {"id": "2", "description": "Write function to filter rows"},
        {"id": "3", "description": "Write function to compute average"}
    ]}
    '''
    decomposer = Decomposer(mock_client)
    result = decomposer.decompose("Read a CSV, filter rows where age > 18, compute average salary")
    assert not result.is_atomic
    assert len(result.subtasks) == 3

def test_decomposer_handles_malformed_response():
    """Should handle and retry on malformed LLM output."""
    from swarm_agents.decomposer import Decomposer, DecompositionError
    mock_client = Mock()
    mock_client.generate.return_value = "This is not JSON"
    decomposer = Decomposer(mock_client)
    with pytest.raises(DecompositionError):
        decomposer.decompose("test query")

def test_subtasks_have_unique_ids():
    """Each subtask should have a unique identifier."""
    from swarm_agents.decomposer import Decomposer
    mock_client = Mock()
    mock_client.generate.return_value = '''
    {"is_atomic": false, "subtasks": [
        {"id": "1", "description": "Task 1"},
        {"id": "2", "description": "Task 2"}
    ]}
    '''
    decomposer = Decomposer(mock_client)
    result = decomposer.decompose("Multi-step task")
    ids = [s.id for s in result.subtasks]
    assert len(ids) == len(set(ids))  # All unique
```

---

## Epic 3: Multi-Agent Execution

### US-3.1: Single Agent Executor

**As a** developer
**I want** an agent that can execute a single subtask
**So that** I can run multiple instances for consensus

**Acceptance Criteria:**
- [ ] `Agent` class with unique `agent_id`
- [ ] `execute(subtask: Subtask) -> AgentResult` method
- [ ] `AgentResult` contains: `agent_id`, `subtask_id`, `code`, `reasoning`, `timestamp`
- [ ] Agent uses system prompt optimized for code generation
- [ ] Code extraction from LLM response (handles markdown code blocks)

**TDD Approach:**
```python
# test_agent.py
import pytest
from unittest.mock import Mock

def test_agent_has_unique_id():
    """Each agent should have a unique identifier."""
    from swarm_agents.agent import Agent
    agent1 = Agent(Mock())
    agent2 = Agent(Mock())
    assert agent1.agent_id != agent2.agent_id

def test_agent_execute_returns_result():
    """Execute should return AgentResult with code."""
    from swarm_agents.agent import Agent
    from swarm_agents.models import Subtask
    mock_client = Mock()
    mock_client.generate.return_value = "```python\ndef is_prime(n):\n    return n > 1\n```"
    agent = Agent(mock_client)
    subtask = Subtask(id="1", description="Write is_prime")
    result = agent.execute(subtask)
    assert result.code == "def is_prime(n):\n    return n > 1"
    assert result.agent_id == agent.agent_id

def test_agent_extracts_code_from_markdown():
    """Should extract code from markdown code blocks."""
    from swarm_agents.agent import Agent
    mock_client = Mock()
    mock_client.generate.return_value = '''
    Here's the solution:
    ```python
    def hello():
        return "world"
    ```
    This function returns "world".
    '''
    agent = Agent(mock_client)
    subtask = Subtask(id="1", description="test")
    result = agent.execute(subtask)
    assert result.code.strip() == 'def hello():\n        return "world"'

def test_agent_result_includes_timestamp():
    """Result should include execution timestamp."""
    from swarm_agents.agent import Agent
    from datetime import datetime
    mock_client = Mock()
    mock_client.generate.return_value = "```python\npass\n```"
    agent = Agent(mock_client)
    result = agent.execute(Subtask(id="1", description="test"))
    assert isinstance(result.timestamp, datetime)
```

---

### US-3.2: Multi-Agent Pool

**As a** developer
**I want** a pool of 5 agents that execute the same subtask independently
**So that** I can collect multiple solutions for consensus voting

**Acceptance Criteria:**
- [ ] `AgentPool` class with configurable agent count (default: 5)
- [ ] `execute_all(subtask: Subtask) -> List[AgentResult]` method
- [ ] Sequential execution (one agent at a time)
- [ ] All agents receive identical input
- [ ] Results collected in order of execution

**TDD Approach:**
```python
# test_agent_pool.py
import pytest
from unittest.mock import Mock, patch

def test_pool_creates_correct_number_of_agents():
    """Pool should create specified number of agents."""
    from swarm_agents.agent import AgentPool
    pool = AgentPool(Mock(), agent_count=5)
    assert len(pool.agents) == 5

def test_pool_executes_all_agents():
    """All agents should execute the subtask."""
    from swarm_agents.agent import AgentPool
    from swarm_agents.models import Subtask
    mock_client = Mock()
    mock_client.generate.return_value = "```python\ndef test(): pass\n```"
    pool = AgentPool(mock_client, agent_count=5)
    subtask = Subtask(id="1", description="test")
    results = pool.execute_all(subtask)
    assert len(results) == 5
    assert mock_client.generate.call_count == 5

def test_pool_results_have_different_agent_ids():
    """Each result should come from a different agent."""
    from swarm_agents.agent import AgentPool
    from swarm_agents.models import Subtask
    mock_client = Mock()
    mock_client.generate.return_value = "```python\npass\n```"
    pool = AgentPool(mock_client, agent_count=5)
    results = pool.execute_all(Subtask(id="1", description="test"))
    agent_ids = [r.agent_id for r in results]
    assert len(set(agent_ids)) == 5  # All unique

def test_pool_executes_sequentially():
    """Agents should execute one at a time (not parallel)."""
    from swarm_agents.agent import AgentPool
    from swarm_agents.models import Subtask
    execution_order = []

    def track_execution(*args, **kwargs):
        execution_order.append(len(execution_order))
        return "```python\npass\n```"

    mock_client = Mock()
    mock_client.generate.side_effect = track_execution
    pool = AgentPool(mock_client, agent_count=3)
    pool.execute_all(Subtask(id="1", description="test"))
    assert execution_order == [0, 1, 2]  # Sequential
```

---

## Epic 4: Consensus Mechanism

### US-4.1: Vote Counter

**As a** developer
**I want** a vote counter that groups identical solutions
**So that** I can determine which solution has majority support

**Acceptance Criteria:**
- [ ] `VoteCounter` class
- [ ] `count_votes(results: List[AgentResult]) -> VoteResult` method
- [ ] Groups results by exact code match (after normalization)
- [ ] `VoteResult` contains: `groups`, `winner`, `confidence`, `is_consensus`
- [ ] Code normalization: strip whitespace, normalize newlines

**TDD Approach:**
```python
# test_voting.py
import pytest

def test_unanimous_vote():
    """All agents agree - should return high confidence."""
    from swarm_agents.voting import VoteCounter
    from swarm_agents.models import AgentResult
    results = [
        AgentResult(agent_id=f"a{i}", subtask_id="1", code="def f(): return 1")
        for i in range(5)
    ]
    counter = VoteCounter()
    vote = counter.count_votes(results)
    assert vote.is_consensus
    assert vote.confidence == 1.0
    assert vote.winner.code == "def f(): return 1"

def test_majority_vote():
    """3 out of 5 agree - should return majority solution."""
    from swarm_agents.voting import VoteCounter
    from swarm_agents.models import AgentResult
    results = [
        AgentResult(agent_id="a1", subtask_id="1", code="def f(): return 1"),
        AgentResult(agent_id="a2", subtask_id="1", code="def f(): return 1"),
        AgentResult(agent_id="a3", subtask_id="1", code="def f(): return 1"),
        AgentResult(agent_id="a4", subtask_id="1", code="def f(): return 2"),
        AgentResult(agent_id="a5", subtask_id="1", code="def f(): return 3"),
    ]
    counter = VoteCounter()
    vote = counter.count_votes(results)
    assert vote.is_consensus
    assert vote.confidence == 0.6  # 3/5
    assert vote.winner.code == "def f(): return 1"

def test_no_consensus():
    """No majority - should indicate no consensus."""
    from swarm_agents.voting import VoteCounter
    from swarm_agents.models import AgentResult
    results = [
        AgentResult(agent_id="a1", subtask_id="1", code="def f(): return 1"),
        AgentResult(agent_id="a2", subtask_id="1", code="def f(): return 2"),
        AgentResult(agent_id="a3", subtask_id="1", code="def f(): return 3"),
        AgentResult(agent_id="a4", subtask_id="1", code="def f(): return 4"),
        AgentResult(agent_id="a5", subtask_id="1", code="def f(): return 5"),
    ]
    counter = VoteCounter()
    vote = counter.count_votes(results)
    assert not vote.is_consensus
    assert vote.winner is None

def test_whitespace_normalization():
    """Should treat code with different whitespace as identical."""
    from swarm_agents.voting import VoteCounter
    from swarm_agents.models import AgentResult
    results = [
        AgentResult(agent_id="a1", subtask_id="1", code="def f():\n    return 1"),
        AgentResult(agent_id="a2", subtask_id="1", code="def f():\n    return 1\n"),
        AgentResult(agent_id="a3", subtask_id="1", code="def f():\n    return 1  "),
        AgentResult(agent_id="a4", subtask_id="1", code="def f(): return 2"),
        AgentResult(agent_id="a5", subtask_id="1", code="def f(): return 2"),
    ]
    counter = VoteCounter()
    vote = counter.count_votes(results)
    assert vote.is_consensus
    assert vote.confidence == 0.6  # 3/5 (first three match after normalization)

def test_vote_result_includes_all_groups():
    """VoteResult should include all solution groups with counts."""
    from swarm_agents.voting import VoteCounter
    from swarm_agents.models import AgentResult
    results = [
        AgentResult(agent_id="a1", subtask_id="1", code="A"),
        AgentResult(agent_id="a2", subtask_id="1", code="A"),
        AgentResult(agent_id="a3", subtask_id="1", code="A"),
        AgentResult(agent_id="a4", subtask_id="1", code="B"),
        AgentResult(agent_id="a5", subtask_id="1", code="B"),
    ]
    counter = VoteCounter()
    vote = counter.count_votes(results)
    assert len(vote.groups) == 2
    assert vote.groups["A"] == 3
    assert vote.groups["B"] == 2
```

---

### US-4.2: Consensus Node

**As a** developer
**I want** a consensus node that integrates into the LangGraph workflow
**So that** voting is part of the processing pipeline

**Acceptance Criteria:**
- [ ] `ConsensusNode` class compatible with LangGraph
- [ ] Takes `List[AgentResult]` from state
- [ ] Returns `ConsensusResult` with winning solution or all candidates
- [ ] Logs vote distribution for debugging
- [ ] Handles edge case of empty results list

**TDD Approach:**
```python
# test_consensus_node.py
import pytest

def test_consensus_node_with_majority():
    """Node should return winning solution when majority exists."""
    from swarm_agents.nodes import ConsensusNode
    from swarm_agents.models import AgentResult, GraphState

    state = GraphState(
        agent_results=[
            AgentResult(agent_id=f"a{i}", subtask_id="1", code="winner")
            for i in range(3)
        ] + [
            AgentResult(agent_id="a4", subtask_id="1", code="loser"),
            AgentResult(agent_id="a5", subtask_id="1", code="other"),
        ]
    )

    node = ConsensusNode()
    result = node.process(state)
    assert result["consensus"].code == "winner"
    assert result["consensus"].confidence == 0.6

def test_consensus_node_no_majority():
    """Node should return all candidates when no majority."""
    from swarm_agents.nodes import ConsensusNode
    from swarm_agents.models import AgentResult, GraphState

    state = GraphState(
        agent_results=[
            AgentResult(agent_id=f"a{i}", subtask_id="1", code=f"solution{i}")
            for i in range(5)
        ]
    )

    node = ConsensusNode()
    result = node.process(state)
    assert result["consensus"].code is None
    assert len(result["consensus"].candidates) == 5

def test_consensus_node_empty_results():
    """Should handle empty results gracefully."""
    from swarm_agents.nodes import ConsensusNode
    from swarm_agents.models import GraphState

    node = ConsensusNode()
    state = GraphState(agent_results=[])
    with pytest.raises(ValueError, match="No agent results"):
        node.process(state)
```

---

## Epic 5: LangGraph Workflow

### US-5.1: Graph State Definition

**As a** developer
**I want** a well-defined state schema for the LangGraph workflow
**So that** data flows correctly between nodes

**Acceptance Criteria:**
- [ ] `GraphState` TypedDict with all required fields
- [ ] Fields: `query`, `subtasks`, `current_subtask`, `agent_results`, `consensus_results`, `final_output`
- [ ] State transitions are type-safe
- [ ] State is serializable for debugging

**TDD Approach:**
```python
# test_graph_state.py
import pytest

def test_graph_state_creation():
    """State should be creatable with initial query."""
    from swarm_agents.state import GraphState
    state = GraphState(query="Write is_prime function")
    assert state["query"] == "Write is_prime function"
    assert state["subtasks"] == []

def test_graph_state_serialization():
    """State should be JSON serializable."""
    import json
    from swarm_agents.state import GraphState, state_to_dict
    state = GraphState(query="test")
    json_str = json.dumps(state_to_dict(state))
    assert "test" in json_str

def test_graph_state_with_results():
    """State should hold agent results."""
    from swarm_agents.state import GraphState
    from swarm_agents.models import AgentResult
    state = GraphState(
        query="test",
        agent_results=[
            AgentResult(agent_id="a1", subtask_id="1", code="pass")
        ]
    )
    assert len(state["agent_results"]) == 1
```

---

### US-5.2: Workflow Graph Construction

**As a** developer
**I want** a LangGraph workflow connecting all nodes
**So that** the full pipeline executes end-to-end

**Acceptance Criteria:**
- [ ] `build_workflow() -> CompiledGraph` function
- [ ] Nodes: `decompose` -> `execute_agents` -> `consensus` -> `aggregate`
- [ ] Conditional edge: loop back for multiple subtasks
- [ ] Entry point is `decompose` node
- [ ] End point is `aggregate` node

**TDD Approach:**
```python
# test_workflow.py
import pytest
from unittest.mock import Mock, patch

def test_workflow_has_required_nodes():
    """Workflow should contain all required nodes."""
    from swarm_agents.workflow import build_workflow
    graph = build_workflow(Mock())
    node_names = list(graph.nodes.keys())
    assert "decompose" in node_names
    assert "execute_agents" in node_names
    assert "consensus" in node_names

def test_workflow_entry_point():
    """Workflow should start at decompose node."""
    from swarm_agents.workflow import build_workflow
    graph = build_workflow(Mock())
    # Check entry edge points to decompose
    assert graph.entry_point == "decompose"

def test_workflow_end_to_end_atomic_task():
    """Simple task should flow through all nodes."""
    from swarm_agents.workflow import build_workflow
    mock_client = Mock()
    # Mock decomposer returning atomic task
    mock_client.generate.side_effect = [
        '{"is_atomic": true, "subtasks": [{"id": "1", "description": "test"}]}',
        # 5 agent responses
        "```python\ndef f(): return 1\n```",
        "```python\ndef f(): return 1\n```",
        "```python\ndef f(): return 1\n```",
        "```python\ndef f(): return 1\n```",
        "```python\ndef f(): return 1\n```",
    ]

    graph = build_workflow(mock_client)
    result = graph.invoke({"query": "Write function f"})

    assert result["consensus_results"] is not None
    assert result["consensus_results"][0].confidence == 1.0
```

---

### US-5.3: Multi-Subtask Iteration

**As a** developer
**I want** the workflow to iterate over multiple subtasks
**So that** complex decomposed tasks are fully processed

**Acceptance Criteria:**
- [ ] Workflow loops through all subtasks
- [ ] Each subtask goes through agent pool + consensus
- [ ] Results aggregated per subtask
- [ ] Final output combines all subtask solutions

**TDD Approach:**
```python
# test_workflow_iteration.py
import pytest
from unittest.mock import Mock

def test_workflow_processes_multiple_subtasks():
    """Workflow should process each subtask independently."""
    from swarm_agents.workflow import build_workflow
    mock_client = Mock()

    # Decomposer returns 2 subtasks
    decompose_response = '''
    {"is_atomic": false, "subtasks": [
        {"id": "1", "description": "Task 1"},
        {"id": "2", "description": "Task 2"}
    ]}
    '''

    responses = [decompose_response]
    # 5 agents x 2 subtasks = 10 code generations
    for i in range(10):
        responses.append(f"```python\ndef task(): return {i % 2}\n```")

    mock_client.generate.side_effect = responses

    graph = build_workflow(mock_client)
    result = graph.invoke({"query": "Complex task"})

    assert len(result["consensus_results"]) == 2

def test_workflow_aggregates_subtask_solutions():
    """Final output should combine all subtask solutions."""
    from swarm_agents.workflow import build_workflow
    mock_client = Mock()

    # Setup mocks for 2 subtasks
    # ... (similar to above)

    graph = build_workflow(mock_client)
    result = graph.invoke({"query": "Multi-step task"})

    final_code = result["final_output"]
    assert "Task 1" in final_code or len(result["consensus_results"]) == 2
```

---

## Epic 6: CLI Interface

### US-6.1: Main Entry Point

**As a** user
**I want** a simple CLI to run queries through the system
**So that** I can test the multi-agent consensus workflow

**Acceptance Criteria:**
- [ ] `main.py` with CLI argument parsing
- [ ] `--query` argument for the task description
- [ ] `--model` argument to override default model
- [ ] `--agents` argument to set agent count
- [ ] Pretty-printed output showing consensus results
- [ ] Exit code 0 on success, 1 on failure

**TDD Approach:**
```python
# test_cli.py
import pytest
from click.testing import CliRunner

def test_cli_requires_query():
    """CLI should require --query argument."""
    from swarm_agents.main import cli
    runner = CliRunner()
    result = runner.invoke(cli, [])
    assert result.exit_code != 0
    assert "query" in result.output.lower()

def test_cli_accepts_query():
    """CLI should accept and process query."""
    from swarm_agents.main import cli
    runner = CliRunner()
    with patch('swarm_agents.main.build_workflow') as mock_wf:
        mock_wf.return_value.invoke.return_value = {
            "final_output": "def f(): pass",
            "consensus_results": [Mock(confidence=1.0)]
        }
        result = runner.invoke(cli, ["--query", "Write function f"])
        assert result.exit_code == 0

def test_cli_shows_confidence():
    """Output should display confidence score."""
    from swarm_agents.main import cli
    runner = CliRunner()
    # ... mock setup
    result = runner.invoke(cli, ["--query", "test"])
    assert "confidence" in result.output.lower()
```

---

## Epic 7: Integration Testing

### US-7.1: End-to-End Integration Test

**As a** developer
**I want** integration tests that verify the full system
**So that** I can ensure all components work together correctly

**Acceptance Criteria:**
- [ ] Integration test with real Ollama (skippable if not available)
- [ ] Test atomic task (single function)
- [ ] Test decomposed task (multiple functions)
- [ ] Verify consensus is reached
- [ ] Verify output code is valid Python (syntax check)

**TDD Approach:**
```python
# test_integration.py
import pytest
import ast

@pytest.mark.integration
@pytest.mark.skipif(not ollama_available(), reason="Ollama not running")
def test_end_to_end_atomic_task():
    """Full system test with atomic task."""
    from swarm_agents.workflow import build_workflow
    from swarm_agents.llm import OllamaClient

    client = OllamaClient(model="qwen3-coder")
    graph = build_workflow(client)

    result = graph.invoke({
        "query": "Write a Python function called is_even that returns True if a number is even"
    })

    # Should reach consensus
    assert result["consensus_results"][0].is_consensus

    # Output should be valid Python
    code = result["final_output"]
    ast.parse(code)  # Raises SyntaxError if invalid

    # Should contain expected function
    assert "def is_even" in code

@pytest.mark.integration
@pytest.mark.skipif(not ollama_available(), reason="Ollama not running")
def test_end_to_end_complex_task():
    """Full system test with decomposable task."""
    from swarm_agents.workflow import build_workflow
    from swarm_agents.llm import OllamaClient

    client = OllamaClient(model="qwen3-coder")
    graph = build_workflow(client)

    result = graph.invoke({
        "query": "Write two functions: one to check if prime, one to get all primes up to n"
    })

    # Should have processed multiple subtasks
    assert len(result["consensus_results"]) >= 1

    # Final output should be valid Python
    code = result["final_output"]
    ast.parse(code)
```

---

## Project Structure

```
swarm/
├── pyproject.toml
├── README.md
├── PLAN.md
├── src/
│   └── swarm_agents/
│       ├── __init__.py
│       ├── main.py           # CLI entry point
│       ├── llm.py            # OllamaClient
│       ├── models.py         # Data classes
│       ├── decomposer.py     # Task decomposition
│       ├── agent.py          # Agent + AgentPool
│       ├── voting.py         # VoteCounter
│       ├── nodes.py          # LangGraph nodes
│       ├── state.py          # GraphState
│       └── workflow.py       # Graph construction
└── tests/
    ├── __init__.py
    ├── conftest.py           # Fixtures
    ├── test_setup.py
    ├── test_ollama_client.py
    ├── test_models.py
    ├── test_decomposer.py
    ├── test_agent.py
    ├── test_agent_pool.py
    ├── test_voting.py
    ├── test_consensus_node.py
    ├── test_graph_state.py
    ├── test_workflow.py
    ├── test_cli.py
    └── test_integration.py
```

---

## Implementation Order

| Phase | User Stories | Dependencies |
|-------|-------------|--------------|
| 1 | US-1.1, US-1.2 | None |
| 2 | US-2.1, US-2.2 | Phase 1 |
| 3 | US-3.1, US-3.2 | Phase 1 |
| 4 | US-4.1, US-4.2 | Phase 3 |
| 5 | US-5.1, US-5.2, US-5.3 | Phases 2, 3, 4 |
| 6 | US-6.1 | Phase 5 |
| 7 | US-7.1 | Phase 6 |

---

## Definition of Done

A user story is complete when:

1. All acceptance criteria are met
2. All TDD tests pass
3. Code is documented with docstrings
4. No linting errors (ruff/flake8)
5. Type hints added for all public functions
6. Integration with existing code verified

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Ollama not installed | Clear setup instructions + skip markers for integration tests |
| Model produces inconsistent output | Stronger prompts with structured output format (JSON) |
| No consensus reached | Return all candidates with warning; user decides |
| Performance too slow | Profile and optimize; consider caching |

---

## Progress Tracker

```
Phase 1: [░░░░░░░░░░] 0%  - Core Infrastructure
Phase 2: [░░░░░░░░░░] 0%  - Task Decomposition
Phase 3: [░░░░░░░░░░] 0%  - Multi-Agent Execution
Phase 4: [░░░░░░░░░░] 0%  - Consensus Mechanism
Phase 5: [░░░░░░░░░░] 0%  - LangGraph Workflow
Phase 6: [░░░░░░░░░░] 0%  - CLI Interface
Phase 7: [░░░░░░░░░░] 0%  - Integration Testing

Overall: [░░░░░░░░░░] 0%
```
