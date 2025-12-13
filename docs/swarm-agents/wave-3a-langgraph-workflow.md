# Wave 3A: LangGraph Workflow

**Wave Type:** ORCHESTRATION (Sequential - Not Parallelizable)

**Status:** Pending

**Dependencies:**
- Wave 2A: Decomposer (US-3.1, US-3.2)
- Wave 2B: Agent Execution (US-4.1, US-4.2, US-4.3)
- Wave 2C: Consensus Node (US-6.1, US-6.2, US-6.3)

**Enables:**
- Wave 4A: CLI and Testing

---

## Overview

Wave 3A implements the core LangGraph workflow that orchestrates the entire multi-agent consensus system. This workflow coordinates the decomposer, agent pool, and consensus mechanisms in a stateful graph structure that can handle both atomic tasks and complex multi-subtask problems.

The workflow provides a declarative graph-based approach to managing the consensus pipeline, with support for conditional branching and iterative subtask processing.

---

## User Stories

### US-5.2: Workflow Graph Construction

**Description:** Build the LangGraph workflow with all required nodes and edges for the consensus pipeline.

**Acceptance Criteria:**
- `build_workflow(client: OllamaClient) -> CompiledGraph` function exists
- Graph contains nodes: decompose, execute_agents, consensus, aggregate
- Conditional edge logic for looping through multiple subtasks
- Entry point set to decompose node
- End point set to aggregate node
- All nodes properly connected with edges
- Graph compiles without errors

**Implementation Details:**

```python
from langgraph.graph import StateGraph, END
from swarm_agents.state import GraphState
from swarm_agents.decomposer import Decomposer
from swarm_agents.agent import AgentPool
from swarm_agents.nodes import ConsensusNode

def build_workflow(client: OllamaClient) -> CompiledGraph:
    """
    Build the LangGraph workflow for multi-agent consensus.

    Args:
        client: OllamaClient instance for LLM interactions

    Returns:
        CompiledGraph ready for execution
    """
    # Initialize components
    decomposer = Decomposer(client)
    agent_pool = AgentPool(client, agent_count=5)
    consensus_node = ConsensusNode()

    # Create graph with GraphState
    graph = StateGraph(GraphState)

    # Add nodes
    graph.add_node("decompose", decompose_node)
    graph.add_node("execute_agents", execute_agents_node)
    graph.add_node("consensus", consensus_node.process)
    graph.add_node("aggregate", aggregate_node)

    # Add edges
    graph.set_entry_point("decompose")
    graph.add_edge("decompose", "execute_agents")
    graph.add_edge("execute_agents", "consensus")

    # Conditional edge for iteration
    graph.add_conditional_edges(
        "consensus",
        should_continue,  # Check if more subtasks remain
        {
            "continue": "execute_agents",
            "end": "aggregate"
        }
    )

    graph.add_edge("aggregate", END)

    return graph.compile()
```

**Node Function Signatures:**

```python
def decompose_node(state: GraphState) -> GraphState:
    """
    Decompose the main task into subtasks.

    Args:
        state: Current graph state with original_task

    Returns:
        Updated state with subtasks list and current_subtask_index=0
    """
    pass

def execute_agents_node(state: GraphState) -> GraphState:
    """
    Execute agent pool on current subtask.

    Args:
        state: Graph state with current subtask

    Returns:
        Updated state with agent_responses for current subtask
    """
    pass

def should_continue(state: GraphState) -> str:
    """
    Determine if more subtasks need processing.

    Args:
        state: Graph state with subtask progress

    Returns:
        "continue" if more subtasks remain, "end" if complete
    """
    pass

def aggregate_node(state: GraphState) -> GraphState:
    """
    Aggregate all consensus results into final output.

    Args:
        state: Graph state with all consensus_results

    Returns:
        Updated state with final_output
    """
    pass
```

**Files Created:**
- `src/swarm_agents/workflow.py` (~150 LOC)

**Tests:**
- `test_workflow_has_required_nodes` - Verify all nodes exist
- `test_workflow_entry_point` - Verify entry is decompose
- `test_workflow_edges_correct` - Verify edge connections
- `test_workflow_compiles` - Verify graph compiles without errors

---

### US-5.3: Multi-Subtask Iteration

**Description:** Implement workflow logic to loop through all subtasks from decomposition, processing each through the agent pool and consensus pipeline.

**Acceptance Criteria:**
- Workflow iterates through all subtasks sequentially
- Each subtask goes through execute_agents -> consensus cycle
- Results stored per subtask in state
- Final aggregate combines all subtask solutions
- State tracks current subtask index
- Handles edge cases (no subtasks, single subtask)

**Implementation Details:**

**GraphState Definition:**

```python
from typing import TypedDict, List, Optional
from swarm_agents.types import SubTask, AgentResponse, ConsensusResult

class GraphState(TypedDict):
    """State object for LangGraph workflow."""

    # Input
    original_task: str

    # Decomposition phase
    subtasks: List[SubTask]
    current_subtask_index: int

    # Execution phase
    current_subtask: Optional[SubTask]
    agent_responses: List[AgentResponse]

    # Consensus phase
    consensus_results: List[ConsensusResult]
    current_consensus: Optional[ConsensusResult]

    # Aggregation phase
    final_output: Optional[str]
```

**Decompose Node Implementation:**

```python
def decompose_node(state: GraphState) -> GraphState:
    """Decompose task into subtasks."""
    decomposer = Decomposer(client)  # Access from closure
    result = decomposer.decompose(state["original_task"])

    return {
        **state,
        "subtasks": result.subtasks,
        "current_subtask_index": 0,
        "consensus_results": [],
    }
```

**Execute Agents Node Implementation:**

```python
def execute_agents_node(state: GraphState) -> GraphState:
    """Execute agent pool on current subtask."""
    subtasks = state["subtasks"]
    idx = state["current_subtask_index"]
    current_subtask = subtasks[idx]

    # Run agent pool
    agent_pool = AgentPool(client, agent_count=5)  # From closure
    responses = agent_pool.execute(current_subtask)

    return {
        **state,
        "current_subtask": current_subtask,
        "agent_responses": responses,
    }
```

**Consensus Node Wrapper:**

```python
def consensus_node_wrapper(state: GraphState) -> GraphState:
    """Process consensus and store result."""
    consensus_node = ConsensusNode()  # From closure
    result = consensus_node.process(
        state["current_subtask"],
        state["agent_responses"]
    )

    # Store consensus result
    consensus_results = state["consensus_results"]
    consensus_results.append(result)

    # Move to next subtask
    return {
        **state,
        "current_consensus": result,
        "consensus_results": consensus_results,
        "current_subtask_index": state["current_subtask_index"] + 1,
    }
```

**Should Continue Implementation:**

```python
def should_continue(state: GraphState) -> str:
    """Check if more subtasks remain."""
    current_idx = state["current_subtask_index"]
    total_subtasks = len(state["subtasks"])

    if current_idx < total_subtasks:
        return "continue"
    else:
        return "end"
```

**Aggregate Node Implementation:**

```python
def aggregate_node(state: GraphState) -> GraphState:
    """Aggregate all consensus results into final output."""
    consensus_results = state["consensus_results"]
    subtasks = state["subtasks"]

    # Build final output
    output_parts = []
    for i, (subtask, result) in enumerate(zip(subtasks, consensus_results)):
        output_parts.append(f"Subtask {i+1}: {subtask.description}")
        output_parts.append(f"Solution: {result.consensus_solution}")
        output_parts.append(f"Confidence: {result.confidence_score:.2f}")
        output_parts.append("")

    final_output = "\n".join(output_parts)

    return {
        **state,
        "final_output": final_output,
    }
```

**Files Modified:**
- `src/swarm_agents/workflow.py` (+50 LOC)
- `src/swarm_agents/state.py` (NEW, ~40 LOC)

**Tests:**
- `test_workflow_processes_single_subtask` - Verify single subtask flow
- `test_workflow_processes_multiple_subtasks` - Verify iteration through 3+ subtasks
- `test_workflow_aggregates_subtask_solutions` - Verify final output format
- `test_workflow_state_tracking` - Verify subtask index increments
- `test_workflow_end_to_end_atomic_task` - Full flow with atomic task
- `test_workflow_handles_no_consensus` - Handle consensus failures gracefully

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         LangGraph Workflow                       │
└─────────────────────────────────────────────────────────────────┘

                           ┌──────────────┐
                           │    START     │
                           └──────┬───────┘
                                  │
                                  ▼
                        ┌──────────────────┐
                        │  decompose_node  │
                        │  - Decomposer    │
                        └────────┬─────────┘
                                 │
                                 ▼
                      ┌────────────────────────┐
                      │  execute_agents_node   │
                      │  - AgentPool(count=5)  │
                      └──────────┬─────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ consensus_node  │
                        │ - ConsensusNode │
                        └────────┬────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ should_continue │
                        └────┬────────┬───┘
                             │        │
                   continue  │        │  end
                             │        │
                             ▼        ▼
                        ┌────────┐  ┌───────────────┐
                        │  LOOP  │  │ aggregate_node│
                        │  BACK  │  └───────┬───────┘
                        └────────┘          │
                                            ▼
                                       ┌────────┐
                                       │  END   │
                                       └────────┘
```

---

## File Structure

```
src/swarm_agents/
├── workflow.py          # Main workflow builder (~200 LOC)
│   ├── build_workflow()
│   ├── decompose_node()
│   ├── execute_agents_node()
│   ├── should_continue()
│   └── aggregate_node()
│
└── state.py             # GraphState definition (~40 LOC)
    └── GraphState (TypedDict)

tests/
├── test_workflow.py                  # Core workflow tests (~150 LOC)
│   ├── test_workflow_has_required_nodes
│   ├── test_workflow_entry_point
│   ├── test_workflow_edges_correct
│   ├── test_workflow_compiles
│   └── test_workflow_end_to_end_atomic_task
│
└── test_workflow_iteration.py        # Iteration tests (~200 LOC)
    ├── test_workflow_processes_single_subtask
    ├── test_workflow_processes_multiple_subtasks
    ├── test_workflow_aggregates_subtask_solutions
    ├── test_workflow_state_tracking
    ├── test_workflow_handles_no_consensus
    └── test_workflow_subtask_isolation
```

**Total Estimated LOC:** ~590

---

## Dependencies

**Python Packages:**
- `langgraph>=0.1.0` - Graph-based workflow orchestration
- `typing-extensions>=4.0.0` - TypedDict support

**Internal Dependencies:**
- `swarm_agents.decomposer.Decomposer` (Wave 2A)
- `swarm_agents.agent.AgentPool` (Wave 2B)
- `swarm_agents.nodes.ConsensusNode` (Wave 2C)
- `swarm_agents.types.SubTask`, `AgentResponse`, `ConsensusResult` (Wave 1)
- `swarm_agents.ollama.OllamaClient` (Wave 1)

---

## Testing Strategy

### Unit Tests

**test_workflow.py:**
```python
def test_workflow_has_required_nodes():
    """Verify all required nodes exist in graph."""
    client = OllamaClient()
    workflow = build_workflow(client)

    # Check nodes
    assert "decompose" in workflow.nodes
    assert "execute_agents" in workflow.nodes
    assert "consensus" in workflow.nodes
    assert "aggregate" in workflow.nodes

def test_workflow_entry_point():
    """Verify workflow starts at decompose."""
    client = OllamaClient()
    workflow = build_workflow(client)

    assert workflow.entry_point == "decompose"

def test_workflow_compiles():
    """Verify workflow compiles without errors."""
    client = OllamaClient()
    workflow = build_workflow(client)

    assert workflow is not None
    assert hasattr(workflow, "invoke")
```

**test_workflow_iteration.py:**
```python
def test_workflow_processes_multiple_subtasks():
    """Verify workflow iterates through all subtasks."""
    client = OllamaClient()
    workflow = build_workflow(client)

    initial_state = {
        "original_task": "Multi-part task requiring decomposition"
    }

    result = workflow.invoke(initial_state)

    # Verify all subtasks processed
    assert len(result["consensus_results"]) == len(result["subtasks"])
    assert result["current_subtask_index"] == len(result["subtasks"])

def test_workflow_aggregates_subtask_solutions():
    """Verify final output aggregates all subtask solutions."""
    client = OllamaClient()
    workflow = build_workflow(client)

    initial_state = {"original_task": "Complex task"}
    result = workflow.invoke(initial_state)

    final_output = result["final_output"]
    assert final_output is not None
    assert "Subtask" in final_output
    assert "Solution" in final_output
    assert "Confidence" in final_output
```

### Integration Tests

```python
def test_workflow_end_to_end_atomic_task():
    """Full workflow with atomic task (no decomposition)."""
    client = OllamaClient()
    workflow = build_workflow(client)

    result = workflow.invoke({
        "original_task": "What is 2+2?"
    })

    assert result["final_output"] is not None
    assert len(result["subtasks"]) == 1  # Atomic task
    assert result["consensus_results"][0].consensus_solution is not None

def test_workflow_handles_no_consensus():
    """Verify graceful handling when consensus fails."""
    # Mock consensus node to return no_consensus=True
    client = OllamaClient()
    workflow = build_workflow(client)

    # Use task that agents disagree on
    result = workflow.invoke({
        "original_task": "Controversial task with no clear answer"
    })

    # Should still complete workflow
    assert result["final_output"] is not None
```

---

## Acceptance Criteria

### US-5.2: Workflow Graph Construction
- [ ] `build_workflow()` function implemented
- [ ] Graph contains all 4 required nodes
- [ ] Entry point is decompose node
- [ ] Conditional edge logic for iteration
- [ ] Graph compiles successfully
- [ ] All tests pass

### US-5.3: Multi-Subtask Iteration
- [ ] Workflow loops through all subtasks
- [ ] Each subtask processed through full pipeline
- [ ] State tracks current subtask index
- [ ] Final output aggregates all results
- [ ] Handles atomic tasks (single subtask)
- [ ] Handles consensus failures gracefully
- [ ] All tests pass

---

## Integration Points

### Inputs from Previous Waves
- **Wave 2A:** Decomposer component
- **Wave 2B:** AgentPool component
- **Wave 2C:** ConsensusNode component

### Outputs for Next Waves
- **Wave 4A:** Compiled workflow graph for CLI integration
- **Testing:** End-to-end workflow for system tests

---

## Error Handling

```python
def decompose_node(state: GraphState) -> GraphState:
    """Decompose with error handling."""
    try:
        decomposer = Decomposer(client)
        result = decomposer.decompose(state["original_task"])

        # Ensure at least one subtask
        if not result.subtasks:
            result.subtasks = [SubTask(
                id="0",
                description=state["original_task"],
                dependencies=[],
                is_atomic=True
            )]

        return {
            **state,
            "subtasks": result.subtasks,
            "current_subtask_index": 0,
            "consensus_results": [],
        }
    except Exception as e:
        # Fallback: treat as atomic task
        return {
            **state,
            "subtasks": [SubTask(
                id="0",
                description=state["original_task"],
                dependencies=[],
                is_atomic=True
            )],
            "current_subtask_index": 0,
            "consensus_results": [],
        }
```

---

## Performance Considerations

- **Sequential Processing:** Each subtask is processed sequentially (required for dependency handling)
- **State Size:** GraphState grows with number of subtasks; consider streaming for very large decompositions
- **Memory:** All agent responses kept in state; may need cleanup for production
- **Checkpointing:** LangGraph supports checkpointing for long-running workflows

---

## Future Enhancements

1. **Parallel Subtask Execution:** For independent subtasks, execute agent pools in parallel
2. **Streaming Output:** Stream results as each subtask completes
3. **Checkpoint/Resume:** Save state for long-running tasks
4. **Dynamic Agent Count:** Adjust agent pool size per subtask complexity
5. **Retry Logic:** Retry failed subtasks with backoff

---

## Notes

- This is an **ORCHESTRATION** wave - it coordinates all previous components
- Must be completed before Wave 4A (CLI)
- LangGraph provides built-in state management and persistence
- Workflow is declarative and testable
- Graph structure allows for easy visualization and debugging

---

## Estimated Effort

- **Development:** 6-8 hours
- **Testing:** 4-6 hours
- **Documentation:** 1-2 hours
- **Total:** 11-16 hours

---

## Completion Checklist

- [ ] `src/swarm_agents/workflow.py` implemented
- [ ] `src/swarm_agents/state.py` implemented
- [ ] All node functions implemented
- [ ] Graph builds and compiles
- [ ] Conditional edges work correctly
- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] Error handling implemented
- [ ] Code reviewed
- [ ] Documentation updated
- [ ] Ready for Wave 4A integration
