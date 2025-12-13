# Wave 2C: Consensus Node

## Overview
**Wave Number:** 2C
**Wave Type:** ENHANCEMENT
**Status:** Not Started
**Dependencies:** Wave 1C (Data Models), Wave 1D (Voting Logic)
**Parallel Execution:** Can run in parallel with Wave 2A, Wave 2B

## User Story
**US-4.2: Consensus Node**

As a developer integrating the swarm system with LangGraph, I need a consensus node that processes agent results and determines the winning solution, so that the graph can make decisions based on collective agent output.

### Acceptance Criteria
- ConsensusNode class compatible with LangGraph node signature
- Takes List[AgentResult] from graph state
- Uses VoteCounter to determine consensus
- Returns ConsensusResult with winning solution or all candidates
- Logs vote distribution for debugging
- Handles edge case of empty results list
- Raises appropriate errors for invalid input

## Technical Specification

### File Structure
```
src/swarm_agents/
└── nodes.py  (~100 LOC)
tests/
└── test_consensus_node.py (~150 LOC)
```

### Key Components

#### 1. ConsensusNode Class

**Location:** `src/swarm_agents/nodes.py`

**Imports:**
```python
import logging
from typing import Dict, Any
from swarm_agents.voting import VoteCounter
from swarm_agents.models import AgentResult, ConsensusResult
from swarm_agents.state import GraphState
```

**Class Definition:**
```python
class ConsensusNode:
    """
    LangGraph-compatible node that processes agent results and determines consensus.

    This node takes a list of AgentResult objects from the graph state, uses the
    VoteCounter to determine if there's a consensus, and returns a ConsensusResult
    that either contains the winning solution or all candidate solutions.

    Attributes:
        vote_counter (VoteCounter): Instance for counting votes and determining consensus
        logger (logging.Logger): Logger for debugging vote distributions
    """

    def __init__(self):
        """Initialize the consensus node with a vote counter and logger."""
        self.vote_counter = VoteCounter()
        self.logger = logging.getLogger(__name__)

    def process(self, state: GraphState) -> Dict[str, Any]:
        """
        Process agent results and return consensus.

        This method follows LangGraph's node signature pattern:
        - Takes state dict as input
        - Returns dict with updates to merge into state

        Args:
            state (GraphState): The current graph state containing agent_results

        Returns:
            Dict[str, Any]: Dictionary containing the consensus result to merge into state

        Raises:
            ValueError: If agent_results is empty or not present in state

        Example:
            >>> node = ConsensusNode()
            >>> state = {"agent_results": [result1, result2, result3]}
            >>> updated_state = node.process(state)
            >>> print(updated_state["consensus"].code)
        """
        # Validate input
        agent_results = state.get("agent_results", [])
        if not agent_results:
            self.logger.error("No agent results found in state")
            raise ValueError("No agent results to process")

        # Count votes and determine consensus
        vote_result = self.vote_counter.count_votes(agent_results)

        # Log vote distribution for debugging
        self.logger.info(f"Vote distribution: {vote_result.groups}")
        self.logger.info(
            f"Consensus: {vote_result.is_consensus}, "
            f"Confidence: {vote_result.confidence:.2%}"
        )

        # Build consensus result based on vote outcome
        if vote_result.is_consensus:
            # We have a winning solution
            consensus = ConsensusResult(
                code=vote_result.winner.code,
                confidence=vote_result.confidence,
                candidates=[],
                vote_distribution=vote_result.groups
            )
            self.logger.info(f"Consensus reached with {len(vote_result.winner_group)} votes")
        else:
            # No consensus - return all candidates
            consensus = ConsensusResult(
                code=None,
                confidence=vote_result.confidence,
                candidates=[r.code for r in agent_results],
                vote_distribution=vote_result.groups
            )
            self.logger.warning(
                f"No consensus reached. Returning {len(agent_results)} candidates"
            )

        # Return dict to merge into state
        return {"consensus": consensus}
```

### Integration with LangGraph

The ConsensusNode is designed to integrate seamlessly with LangGraph:

```python
from langgraph.graph import StateGraph
from swarm_agents.nodes import ConsensusNode
from swarm_agents.state import GraphState

# Create graph
graph = StateGraph(GraphState)

# Initialize consensus node
consensus_node = ConsensusNode()

# Add node to graph
graph.add_node("consensus", consensus_node.process)

# Connect to other nodes
graph.add_edge("agents", "consensus")
graph.add_edge("consensus", "output")
```

**Node Signature:**
- Input: `state: GraphState` (dict-like object)
- Output: `Dict[str, Any]` (updates to merge into state)
- State Keys Used:
  - Input: `agent_results` (List[AgentResult])
  - Output: `consensus` (ConsensusResult)

## Test Cases

**Location:** `tests/test_consensus_node.py`

### Test Suite

```python
import pytest
import logging
from swarm_agents.nodes import ConsensusNode
from swarm_agents.models import AgentResult, ConsensusResult
from swarm_agents.state import GraphState


class TestConsensusNode:
    """Test suite for ConsensusNode class."""

    @pytest.fixture
    def consensus_node(self):
        """Create a ConsensusNode instance for testing."""
        return ConsensusNode()

    @pytest.fixture
    def sample_results_majority(self):
        """Create sample results with a clear majority (3/5)."""
        return [
            AgentResult(code="solution_a", confidence=0.9, metadata={}),
            AgentResult(code="solution_a", confidence=0.85, metadata={}),
            AgentResult(code="solution_a", confidence=0.8, metadata={}),
            AgentResult(code="solution_b", confidence=0.75, metadata={}),
            AgentResult(code="solution_c", confidence=0.7, metadata={}),
        ]

    @pytest.fixture
    def sample_results_no_majority(self):
        """Create sample results with no majority (all different)."""
        return [
            AgentResult(code="solution_a", confidence=0.9, metadata={}),
            AgentResult(code="solution_b", confidence=0.85, metadata={}),
            AgentResult(code="solution_c", confidence=0.8, metadata={}),
            AgentResult(code="solution_d", confidence=0.75, metadata={}),
            AgentResult(code="solution_e", confidence=0.7, metadata={}),
        ]

    @pytest.fixture
    def sample_results_unanimous(self):
        """Create sample results with unanimous agreement (5/5)."""
        return [
            AgentResult(code="solution_a", confidence=0.9, metadata={}),
            AgentResult(code="solution_a", confidence=0.88, metadata={}),
            AgentResult(code="solution_a", confidence=0.87, metadata={}),
            AgentResult(code="solution_a", confidence=0.85, metadata={}),
            AgentResult(code="solution_a", confidence=0.82, metadata={}),
        ]

    def test_consensus_node_with_majority(self, consensus_node, sample_results_majority):
        """Test consensus node correctly identifies majority winner (3/5 agree)."""
        state = {"agent_results": sample_results_majority}
        result = consensus_node.process(state)

        assert "consensus" in result
        consensus = result["consensus"]

        assert isinstance(consensus, ConsensusResult)
        assert consensus.code == "solution_a"
        assert consensus.confidence > 0.5
        assert len(consensus.candidates) == 0  # No candidates when consensus reached
        assert len(consensus.vote_distribution) == 3  # 3 unique solutions

    def test_consensus_node_no_majority(self, consensus_node, sample_results_no_majority):
        """Test consensus node handles no majority case (all different)."""
        state = {"agent_results": sample_results_no_majority}
        result = consensus_node.process(state)

        assert "consensus" in result
        consensus = result["consensus"]

        assert isinstance(consensus, ConsensusResult)
        assert consensus.code is None  # No winner
        assert len(consensus.candidates) == 5  # All candidates returned
        assert consensus.confidence < 0.5  # Low confidence
        assert len(consensus.vote_distribution) == 5  # 5 unique solutions

    def test_consensus_node_empty_results(self, consensus_node):
        """Test consensus node raises ValueError for empty results list."""
        state = {"agent_results": []}

        with pytest.raises(ValueError, match="No agent results to process"):
            consensus_node.process(state)

    def test_consensus_node_missing_results_key(self, consensus_node):
        """Test consensus node raises ValueError when agent_results key is missing."""
        state = {}

        with pytest.raises(ValueError, match="No agent results to process"):
            consensus_node.process(state)

    def test_consensus_node_logs_distribution(self, consensus_node, sample_results_majority, caplog):
        """Test consensus node logs vote distribution for debugging."""
        state = {"agent_results": sample_results_majority}

        with caplog.at_level(logging.INFO):
            consensus_node.process(state)

        # Check that vote distribution was logged
        assert "Vote distribution:" in caplog.text
        assert "Consensus:" in caplog.text
        assert "Confidence:" in caplog.text
        assert "Consensus reached with" in caplog.text

    def test_consensus_node_unanimous(self, consensus_node, sample_results_unanimous):
        """Test consensus node handles unanimous agreement (5/5)."""
        state = {"agent_results": sample_results_unanimous}
        result = consensus_node.process(state)

        assert "consensus" in result
        consensus = result["consensus"]

        assert isinstance(consensus, ConsensusResult)
        assert consensus.code == "solution_a"
        assert consensus.confidence == 1.0  # Perfect consensus
        assert len(consensus.candidates) == 0
        assert len(consensus.vote_distribution) == 1  # Only one group

    def test_consensus_node_logs_no_consensus_warning(
        self, consensus_node, sample_results_no_majority, caplog
    ):
        """Test consensus node logs warning when no consensus is reached."""
        state = {"agent_results": sample_results_no_majority}

        with caplog.at_level(logging.WARNING):
            consensus_node.process(state)

        assert "No consensus reached" in caplog.text
        assert "Returning 5 candidates" in caplog.text

    def test_consensus_node_return_type(self, consensus_node, sample_results_majority):
        """Test consensus node returns correct dict structure for LangGraph."""
        state = {"agent_results": sample_results_majority}
        result = consensus_node.process(state)

        assert isinstance(result, dict)
        assert "consensus" in result
        assert isinstance(result["consensus"], ConsensusResult)

    def test_consensus_node_state_preservation(self, consensus_node, sample_results_majority):
        """Test consensus node doesn't modify other state keys."""
        state = {
            "agent_results": sample_results_majority,
            "other_key": "other_value",
            "another_key": 123
        }
        result = consensus_node.process(state)

        # Result should only contain consensus key
        assert len(result) == 1
        assert "consensus" in result
        # Original state should not be modified
        assert "other_key" in state
        assert "another_key" in state
```

### Test Coverage Requirements
- Line coverage: >= 95%
- Branch coverage: >= 90%
- All edge cases covered:
  - Empty results list
  - Missing state keys
  - Majority consensus
  - No consensus
  - Unanimous consensus
  - Logging behavior

## Implementation Checklist

### Phase 1: Core Implementation
- [ ] Create `src/swarm_agents/nodes.py` file
- [ ] Implement ConsensusNode class
- [ ] Add proper type hints and docstrings
- [ ] Set up logging for vote distribution
- [ ] Handle empty results edge case
- [ ] Return correct dict structure for LangGraph

### Phase 2: Testing
- [ ] Create `tests/test_consensus_node.py` file
- [ ] Implement test fixtures
- [ ] Test majority consensus case
- [ ] Test no majority case
- [ ] Test empty results error handling
- [ ] Test logging output
- [ ] Test unanimous consensus
- [ ] Verify LangGraph compatibility

### Phase 3: Integration
- [ ] Verify integration with VoteCounter from Wave 1D
- [ ] Verify integration with models from Wave 1C
- [ ] Test with actual GraphState object
- [ ] Add integration test with LangGraph
- [ ] Update documentation with usage examples

### Phase 4: Quality Assurance
- [ ] Run all tests and verify coverage >= 95%
- [ ] Run type checker (mypy)
- [ ] Run linter (ruff/pylint)
- [ ] Verify logging output is useful for debugging
- [ ] Performance test with large result sets (100+ agents)
- [ ] Code review

## Dependencies

### Required Waves
- **Wave 1C (Data Models):** Provides `AgentResult`, `ConsensusResult`, `GraphState`
- **Wave 1D (Voting Logic):** Provides `VoteCounter` and `VoteResult`

### External Dependencies
```toml
[tool.poetry.dependencies]
python = "^3.11"
langgraph = "^0.2.0"  # For LangGraph integration
```

### Development Dependencies
```toml
[tool.poetry.group.dev.dependencies]
pytest = "^8.0.0"
pytest-cov = "^4.1.0"
mypy = "^1.8.0"
```

## Success Criteria

### Functional Requirements
- [ ] ConsensusNode processes agent results correctly
- [ ] Returns ConsensusResult with winning solution when consensus exists
- [ ] Returns all candidates when no consensus exists
- [ ] Handles empty results gracefully with clear error message
- [ ] Logs vote distribution for debugging
- [ ] Compatible with LangGraph node signature

### Non-Functional Requirements
- [ ] Test coverage >= 95%
- [ ] All type hints pass mypy strict mode
- [ ] No linter warnings
- [ ] Clear, comprehensive docstrings
- [ ] Processing time < 100ms for 100 agents

## Notes

### LangGraph Integration Pattern
The ConsensusNode follows LangGraph's functional node pattern:
- Stateless operation (uses instance variables only for configuration)
- Pure function behavior (no side effects except logging)
- Clear input/output contract via state dict
- Error handling that doesn't break the graph

### Logging Strategy
Logging is crucial for debugging consensus issues:
- INFO level: Normal vote distribution and consensus status
- WARNING level: No consensus reached
- ERROR level: Invalid input or processing errors

### Performance Considerations
- VoteCounter operations should be O(n) where n is number of agents
- Grouping operations should use efficient data structures (dict/Counter)
- Large vote distributions should be logged at DEBUG level only

### Future Enhancements (Not in Scope)
- Tie-breaking strategies for equal vote counts
- Weighted voting based on agent confidence
- Consensus threshold configuration (currently using VoteCounter default)
- Vote distribution visualization
- Historical consensus tracking

## Timeline Estimate
- Core implementation: 2 hours
- Testing: 2 hours
- Integration and QA: 1 hour
- **Total: 5 hours**

## Related Waves
- **Wave 2A (Agent Node):** Produces the AgentResult objects consumed by this node
- **Wave 2B (Reflection Node):** May consume ConsensusResult for validation
- **Wave 3A (Graph Builder):** Composes this node into the full LangGraph workflow
