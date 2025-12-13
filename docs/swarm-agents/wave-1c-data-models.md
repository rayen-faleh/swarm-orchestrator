# Wave 1C: Core Data Models

## Overview

Wave 1C establishes the foundational data models and state management structures for the multi-agent consensus system. This wave implements the core dataclasses and type definitions that will be used throughout the system for representing subtasks, agent outputs, voting results, and the overall graph state.

These models serve as the contract between different components of the system, ensuring type safety and consistent data structures across decomposition, agent execution, voting, and consensus phases.

## Dependencies

- **Requires**: Wave 1A (Project Bootstrap)
- **Parallel with**: Wave 1B (LLM Integration), Wave 1D (Configuration Management)
- **Enables**: Wave 2A (Task Decomposition), Wave 2B (Agent Orchestration), Wave 2C (Voting & Consensus)

## User Stories

### US-2.1: Subtask Data Model

**As a** system architect
**I want** structured data models for subtasks and decomposition results
**So that** task decomposition outputs can be consistently represented and validated across the system

#### Implementation Details

Create `src/swarm_agents/models.py` with the following dataclasses:

```python
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime
import json
from uuid import uuid4


@dataclass
class Subtask:
    """Represents a single subtask in a decomposed query."""

    id: str = field(default_factory=lambda: str(uuid4()))
    description: str = ""
    context: str = ""
    dependencies: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate required fields."""
        if not self.description:
            raise ValueError("Subtask description is required")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Subtask':
        """Create Subtask from dictionary."""
        return cls(**data)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'Subtask':
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class DecompositionResult:
    """Result of task decomposition operation."""

    original_query: str
    subtasks: List[Subtask] = field(default_factory=list)
    is_atomic: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def __post_init__(self):
        """Validate decomposition result."""
        if not self.original_query:
            raise ValueError("Original query is required")

        if not self.is_atomic and not self.subtasks:
            raise ValueError("Non-atomic queries must have subtasks")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'original_query': self.original_query,
            'subtasks': [st.to_dict() for st in self.subtasks],
            'is_atomic': self.is_atomic,
            'metadata': self.metadata,
            'timestamp': self.timestamp
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DecompositionResult':
        """Create DecompositionResult from dictionary."""
        subtasks = [Subtask.from_dict(st) for st in data.get('subtasks', [])]
        return cls(
            original_query=data['original_query'],
            subtasks=subtasks,
            is_atomic=data.get('is_atomic', False),
            metadata=data.get('metadata', {}),
            timestamp=data.get('timestamp', datetime.utcnow().isoformat())
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'DecompositionResult':
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))
```

#### Validation Rules

1. **Subtask**:
   - `description` must not be empty
   - `id` auto-generated if not provided
   - `dependencies` must be a list of valid subtask IDs
   - `context` can be empty but must be a string

2. **DecompositionResult**:
   - `original_query` must not be empty
   - If `is_atomic` is False, `subtasks` must contain at least one subtask
   - If `is_atomic` is True, `subtasks` should be empty
   - `timestamp` auto-generated in ISO format

### US-2.2: Agent Result Model

**As a** agent orchestrator
**I want** a standardized format for agent execution results
**So that** agent outputs can be collected and compared consistently

#### Implementation Details

Add to `src/swarm_agents/models.py`:

```python
@dataclass
class AgentResult:
    """Result from a single agent's execution."""

    agent_id: str
    subtask_id: str
    code: str
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    execution_time_ms: Optional[int] = None

    def __post_init__(self):
        """Validate agent result."""
        if not self.agent_id:
            raise ValueError("Agent ID is required")
        if not self.subtask_id:
            raise ValueError("Subtask ID is required")
        if not self.code:
            raise ValueError("Code output is required")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentResult':
        """Create AgentResult from dictionary."""
        return cls(**data)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'AgentResult':
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))
```

### US-2.3: Voting and Consensus Models

**As a** consensus mechanism
**I want** structured models for voting results and consensus outcomes
**So that** voting processes can be tracked and validated

#### Implementation Details

Add to `src/swarm_agents/models.py`:

```python
@dataclass
class VoteResult:
    """Result of a voting round."""

    groups: List[List[str]]  # Groups of similar solutions (agent IDs)
    winner: str  # Agent ID of winning solution
    confidence: float  # 0.0 to 1.0
    is_consensus: bool
    vote_distribution: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def __post_init__(self):
        """Validate vote result."""
        if not self.groups:
            raise ValueError("Groups cannot be empty")
        if not self.winner:
            raise ValueError("Winner is required")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")

        # Verify winner is in one of the groups
        all_agents = [agent for group in self.groups for agent in group]
        if self.winner not in all_agents:
            raise ValueError(f"Winner {self.winner} not found in groups")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VoteResult':
        """Create VoteResult from dictionary."""
        return cls(**data)


@dataclass
class ConsensusResult:
    """Final consensus result for a subtask."""

    subtask_id: str
    code: str
    confidence: float  # 0.0 to 1.0
    candidates: List[str]  # Agent IDs that contributed
    vote_distribution: Dict[str, int] = field(default_factory=dict)
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def __post_init__(self):
        """Validate consensus result."""
        if not self.subtask_id:
            raise ValueError("Subtask ID is required")
        if not self.code:
            raise ValueError("Code is required")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if not self.candidates:
            raise ValueError("Candidates list cannot be empty")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConsensusResult':
        """Create ConsensusResult from dictionary."""
        return cls(**data)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'ConsensusResult':
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))
```

### US-5.1: Graph State Definition

**As a** LangGraph orchestrator
**I want** a typed state definition for the graph workflow
**So that** state transitions are type-safe and predictable

#### Implementation Details

Create `src/swarm_agents/state.py`:

```python
from typing import TypedDict, List, Optional, Dict, Any
from swarm_agents.models import (
    Subtask,
    DecompositionResult,
    AgentResult,
    ConsensusResult
)


class GraphState(TypedDict, total=False):
    """State definition for the LangGraph workflow.

    This TypedDict defines all possible fields in the graph state.
    Using total=False allows fields to be optional.
    """

    # Input
    query: str

    # Decomposition phase
    decomposition_result: Optional[DecompositionResult]
    subtasks: List[Subtask]
    is_atomic: bool

    # Execution phase
    current_subtask_index: int
    current_subtask: Optional[Subtask]
    agent_results: Dict[str, List[AgentResult]]  # subtask_id -> [results]

    # Voting & Consensus phase
    vote_results: Dict[str, Any]  # subtask_id -> VoteResult dict
    consensus_results: Dict[str, ConsensusResult]  # subtask_id -> ConsensusResult

    # Final output
    final_output: str
    final_code: str

    # Metadata
    metadata: Dict[str, Any]
    error: Optional[str]


def create_initial_state(query: str) -> GraphState:
    """Create initial graph state from a query.

    Args:
        query: The user's input query

    Returns:
        GraphState with initialized fields
    """
    return GraphState(
        query=query,
        decomposition_result=None,
        subtasks=[],
        is_atomic=False,
        current_subtask_index=0,
        current_subtask=None,
        agent_results={},
        vote_results={},
        consensus_results={},
        final_output="",
        final_code="",
        metadata={},
        error=None
    )


def state_to_dict(state: GraphState) -> Dict[str, Any]:
    """Convert GraphState to a serializable dictionary.

    Handles conversion of complex objects to dictionaries.

    Args:
        state: The graph state to serialize

    Returns:
        Dictionary representation of the state
    """
    result = {}

    for key, value in state.items():
        if value is None:
            result[key] = None
        elif key == 'decomposition_result' and value:
            result[key] = value.to_dict()
        elif key == 'subtasks':
            result[key] = [st.to_dict() for st in value]
        elif key == 'current_subtask' and value:
            result[key] = value.to_dict()
        elif key == 'agent_results':
            result[key] = {
                subtask_id: [ar.to_dict() for ar in results]
                for subtask_id, results in value.items()
            }
        elif key == 'consensus_results':
            result[key] = {
                subtask_id: cr.to_dict()
                for subtask_id, cr in value.items()
            }
        else:
            result[key] = value

    return result


def dict_to_state(data: Dict[str, Any]) -> GraphState:
    """Convert dictionary to GraphState.

    Handles conversion of dictionaries back to typed objects.

    Args:
        data: Dictionary representation of state

    Returns:
        GraphState with typed objects
    """
    from swarm_agents.models import DecompositionResult, Subtask, AgentResult, ConsensusResult

    state = GraphState(**data)

    # Convert decomposition_result
    if data.get('decomposition_result'):
        state['decomposition_result'] = DecompositionResult.from_dict(
            data['decomposition_result']
        )

    # Convert subtasks
    if data.get('subtasks'):
        state['subtasks'] = [
            Subtask.from_dict(st) for st in data['subtasks']
        ]

    # Convert current_subtask
    if data.get('current_subtask'):
        state['current_subtask'] = Subtask.from_dict(data['current_subtask'])

    # Convert agent_results
    if data.get('agent_results'):
        state['agent_results'] = {
            subtask_id: [AgentResult.from_dict(ar) for ar in results]
            for subtask_id, results in data['agent_results'].items()
        }

    # Convert consensus_results
    if data.get('consensus_results'):
        state['consensus_results'] = {
            subtask_id: ConsensusResult.from_dict(cr)
            for subtask_id, cr in data['consensus_results'].items()
        }

    return state
```

## Acceptance Criteria

### US-2.1: Subtask Data Model
- [ ] Subtask dataclass with all required fields
- [ ] DecompositionResult dataclass with validation
- [ ] JSON serialization/deserialization methods
- [ ] Validation enforces required fields
- [ ] Auto-generation of IDs and timestamps
- [ ] Unit tests with 100% coverage

### US-2.2: Agent Result Model
- [ ] AgentResult dataclass with all required fields
- [ ] Validation for required fields
- [ ] JSON serialization/deserialization
- [ ] Timestamp and execution time tracking
- [ ] Unit tests with 100% coverage

### US-2.3: Voting and Consensus Models
- [ ] VoteResult dataclass with validation
- [ ] ConsensusResult dataclass with validation
- [ ] Confidence bounds checking (0.0-1.0)
- [ ] Winner validation in groups
- [ ] JSON serialization/deserialization
- [ ] Unit tests with 100% coverage

### US-5.1: Graph State Definition
- [ ] GraphState TypedDict with all workflow fields
- [ ] create_initial_state helper function
- [ ] state_to_dict serialization helper
- [ ] dict_to_state deserialization helper
- [ ] Type hints for all fields
- [ ] Unit tests with 100% coverage

## Technical Implementation

### File Structure

```
src/swarm_agents/
├── __init__.py
├── models.py      (~250 LOC)
│   ├── Subtask
│   ├── DecompositionResult
│   ├── AgentResult
│   ├── VoteResult
│   └── ConsensusResult
└── state.py       (~100 LOC)
    ├── GraphState (TypedDict)
    ├── create_initial_state()
    ├── state_to_dict()
    └── dict_to_state()

tests/
├── test_models.py (~300 LOC)
│   ├── test_subtask_creation
│   ├── test_subtask_validation
│   ├── test_subtask_serialization
│   ├── test_decomposition_result
│   ├── test_agent_result
│   ├── test_vote_result
│   └── test_consensus_result
└── test_graph_state.py (~200 LOC)
    ├── test_initial_state
    ├── test_state_serialization
    ├── test_state_deserialization
    └── test_complex_state_roundtrip
```

### Testing Strategy

#### Unit Tests for Models

```python
# tests/test_models.py
import pytest
from swarm_agents.models import (
    Subtask,
    DecompositionResult,
    AgentResult,
    VoteResult,
    ConsensusResult
)


class TestSubtask:
    def test_create_subtask_with_defaults(self):
        """Test subtask creation with auto-generated ID."""
        subtask = Subtask(description="Test task")
        assert subtask.id is not None
        assert subtask.description == "Test task"
        assert subtask.context == ""
        assert subtask.dependencies == []

    def test_subtask_requires_description(self):
        """Test that empty description raises ValueError."""
        with pytest.raises(ValueError, match="description is required"):
            Subtask(description="")

    def test_subtask_serialization(self):
        """Test JSON serialization roundtrip."""
        original = Subtask(
            description="Test",
            context="Context",
            dependencies=["dep1"]
        )
        json_str = original.to_json()
        restored = Subtask.from_json(json_str)

        assert restored.description == original.description
        assert restored.context == original.context
        assert restored.dependencies == original.dependencies


class TestDecompositionResult:
    def test_atomic_query(self):
        """Test atomic query decomposition."""
        result = DecompositionResult(
            original_query="Simple task",
            is_atomic=True
        )
        assert result.is_atomic
        assert result.subtasks == []

    def test_non_atomic_requires_subtasks(self):
        """Test that non-atomic queries need subtasks."""
        with pytest.raises(ValueError, match="must have subtasks"):
            DecompositionResult(
                original_query="Complex task",
                is_atomic=False,
                subtasks=[]
            )


class TestAgentResult:
    def test_create_agent_result(self):
        """Test agent result creation."""
        result = AgentResult(
            agent_id="agent-1",
            subtask_id="subtask-1",
            code="print('hello')",
            reasoning="Simple output"
        )
        assert result.agent_id == "agent-1"
        assert result.timestamp is not None

    def test_agent_result_requires_code(self):
        """Test that code is required."""
        with pytest.raises(ValueError, match="Code output is required"):
            AgentResult(
                agent_id="agent-1",
                subtask_id="subtask-1",
                code=""
            )


class TestVoteResult:
    def test_vote_result_validation(self):
        """Test vote result validation."""
        result = VoteResult(
            groups=[["agent-1", "agent-2"], ["agent-3"]],
            winner="agent-1",
            confidence=0.8,
            is_consensus=True
        )
        assert result.confidence == 0.8

    def test_winner_must_be_in_groups(self):
        """Test that winner must exist in groups."""
        with pytest.raises(ValueError, match="not found in groups"):
            VoteResult(
                groups=[["agent-1", "agent-2"]],
                winner="agent-3",
                confidence=0.5,
                is_consensus=False
            )

    def test_confidence_bounds(self):
        """Test confidence must be between 0 and 1."""
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            VoteResult(
                groups=[["agent-1"]],
                winner="agent-1",
                confidence=1.5,
                is_consensus=True
            )


class TestConsensusResult:
    def test_consensus_result(self):
        """Test consensus result creation."""
        result = ConsensusResult(
            subtask_id="task-1",
            code="final_code",
            confidence=0.9,
            candidates=["agent-1", "agent-2"]
        )
        assert result.confidence == 0.9
        assert len(result.candidates) == 2
```

#### Unit Tests for Graph State

```python
# tests/test_graph_state.py
import pytest
from swarm_agents.state import (
    GraphState,
    create_initial_state,
    state_to_dict,
    dict_to_state
)
from swarm_agents.models import Subtask, AgentResult


class TestGraphState:
    def test_create_initial_state(self):
        """Test initial state creation."""
        state = create_initial_state("test query")

        assert state['query'] == "test query"
        assert state['subtasks'] == []
        assert state['current_subtask_index'] == 0
        assert state['agent_results'] == {}
        assert state['error'] is None

    def test_state_serialization_simple(self):
        """Test serialization of simple state."""
        state = create_initial_state("test")
        state_dict = state_to_dict(state)

        assert state_dict['query'] == "test"
        assert isinstance(state_dict, dict)

    def test_state_serialization_with_subtasks(self):
        """Test serialization with subtasks."""
        state = create_initial_state("test")
        state['subtasks'] = [
            Subtask(description="Task 1"),
            Subtask(description="Task 2")
        ]

        state_dict = state_to_dict(state)
        assert len(state_dict['subtasks']) == 2
        assert state_dict['subtasks'][0]['description'] == "Task 1"

    def test_state_roundtrip(self):
        """Test serialization/deserialization roundtrip."""
        original = create_initial_state("test query")
        original['subtasks'] = [Subtask(description="Test")]
        original['agent_results'] = {
            "task-1": [
                AgentResult(
                    agent_id="agent-1",
                    subtask_id="task-1",
                    code="print('test')"
                )
            ]
        }

        # Serialize
        state_dict = state_to_dict(original)

        # Deserialize
        restored = dict_to_state(state_dict)

        assert restored['query'] == original['query']
        assert len(restored['subtasks']) == 1
        assert restored['subtasks'][0].description == "Test"
        assert "task-1" in restored['agent_results']
```

### Dependencies

Add to `pyproject.toml`:

```toml
[project]
dependencies = [
    "typing-extensions>=4.0.0",  # For TypedDict backport
]
```

### Import Structure

Update `src/swarm_agents/__init__.py`:

```python
"""Swarm Agents - Multi-agent consensus system."""

from swarm_agents.models import (
    Subtask,
    DecompositionResult,
    AgentResult,
    VoteResult,
    ConsensusResult,
)
from swarm_agents.state import (
    GraphState,
    create_initial_state,
    state_to_dict,
    dict_to_state,
)

__all__ = [
    "Subtask",
    "DecompositionResult",
    "AgentResult",
    "VoteResult",
    "ConsensusResult",
    "GraphState",
    "create_initial_state",
    "state_to_dict",
    "dict_to_state",
]
```

## Success Criteria

### Code Quality
- [ ] All models use Python 3.10+ dataclasses
- [ ] Type hints on all functions and methods
- [ ] Docstrings for all public classes and methods
- [ ] No circular imports
- [ ] Passes mypy type checking with strict mode
- [ ] Passes ruff linting with no errors

### Testing
- [ ] Unit test coverage >= 100% for models.py
- [ ] Unit test coverage >= 100% for state.py
- [ ] All validation edge cases tested
- [ ] Serialization roundtrip tests pass
- [ ] Tests run in < 1 second

### Documentation
- [ ] All dataclasses documented with field descriptions
- [ ] Usage examples in docstrings
- [ ] Type hints serve as inline documentation
- [ ] Clear error messages for validation failures

### Integration
- [ ] Models can be imported from swarm_agents package
- [ ] No runtime dependencies beyond standard library + typing-extensions
- [ ] Compatible with LangGraph state management
- [ ] JSON serialization produces valid, readable output

## Estimated Effort

**Total: 2-3 hours**

Breakdown:
- Models implementation: 1 hour
- State implementation: 30 minutes
- Unit tests: 1 hour
- Documentation & validation: 30 minutes

## Notes

### Design Decisions

1. **Dataclasses over Pydantic**: Using standard library dataclasses for simplicity and minimal dependencies. If validation becomes more complex, consider migrating to Pydantic in a future wave.

2. **UUID for IDs**: Auto-generating UUIDs for subtask IDs ensures uniqueness across distributed systems.

3. **ISO Timestamps**: Using ISO 8601 format for timestamps ensures consistency and parseability.

4. **TypedDict for GraphState**: Using TypedDict instead of dataclass allows LangGraph to use dictionary operations while maintaining type safety.

5. **Validation in __post_init__**: All validation happens in `__post_init__` to ensure invalid objects cannot be created.

### Future Enhancements

1. **Schema Versioning**: Add schema version fields to support future migrations
2. **Custom Serializers**: Support for custom serialization formats (YAML, MessagePack)
3. **Immutability**: Consider frozen dataclasses for immutable state
4. **Rich Comparison**: Add comparison methods for sorting and deduplication

### Related Waves

- **Wave 2A**: Will use `Subtask` and `DecompositionResult`
- **Wave 2B**: Will use `AgentResult` and consume `GraphState`
- **Wave 2C**: Will use `VoteResult` and `ConsensusResult`
- **Wave 3A**: Will use all models for integration testing
