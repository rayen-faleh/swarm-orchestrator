# Schaltwerk WAVE Workflow Guide

## Overview

This guide documents how to effectively use the Schaltwerk MCP server with the WAVE (Workflow And Parallel Execution) format to break down large implementation plans into parallelizable workflows that can be developed concurrently and merged back to the main branch.

## Philosophy

The WAVE approach enables:
- **Parallel Development**: Multiple independent features developed simultaneously
- **Reduced Time-to-Delivery**: Critical path optimization through parallelization
- **Clear Dependencies**: Explicit dependency tracking between waves
- **Isolated Development**: Each wave has its own branch and worktree
- **Incremental Integration**: Waves merge back when complete, reducing integration risk

## The WAVE Format

### Wave Naming Convention
```
wave-{number}{letter}-{descriptive-name}
```

**Examples:**
- `wave-13a-plan-document-structure`
- `wave-13b-memory-indexing-system`
- `wave-14a-plan-review-execution`

### Wave Structure

Each wave consists of:
1. **Markdown Specification** (`docs/*/wave-{number}{letter}-{name}.md`)
   - Overview and goals
   - Dependencies (requires/parallel with/enables)
   - User stories
   - Component architecture with code examples
   - Testing requirements
   - Performance benchmarks
   - Estimated effort

2. **Schaltwerk Draft Spec** (created via MCP)
   - Concise summary for AI agent
   - Key components and LOC estimates
   - Performance targets
   - Testing requirements
   - Reference to full markdown spec

3. **Git Branch** (`schaltwerk/wave-{number}{letter}-{name}`)
   - Created automatically by Schaltwerk
   - Isolated worktree for development
   - Merged back to master when complete

## Step-by-Step Workflow

### Phase 1: Analysis and Planning

#### 1. Analyze the Implementation Plan

Start with a comprehensive implementation plan document (e.g., `AGENT_2_0_IMPLEMENTATION_PLAN.md`).

**Identify:**
- Core pillars/epics
- User stories
- Dependencies between features
- Natural parallelization boundaries

#### 2. Group Stories into Waves

**Criteria for wave grouping:**
- **Foundation Waves**: No dependencies, can run in parallel
- **Enhancement Waves**: Depend on foundation, can run in parallel
- **Integration Waves**: Depend on multiple waves, usually sequential

**Example from Agent 2.0:**

```
Foundation (Parallel):
├── Wave 13A: Plan Document Structure (3-4 days)
├── Wave 13B: Memory Indexing System (5-6 days)
└── Wave 13C: Protocol Management (4-5 days)

Enhancement (Parallel, after foundation):
├── Wave 14A: Plan Review & Execution (6-7 days) [requires 13A]
├── Wave 14B: Memory Enhancement (6-7 days) [requires 13B]
└── Wave 14C: Context Engineering (5-6 days) [requires 13C]

Orchestration (Sequential):
├── Wave 15A: Orchestration Foundation (6-7 days) [requires 14A]
└── Wave 15B: Sub-Agent Specialization (7-8 days) [requires 15A, 14C]

Integration (Sequential):
├── Wave 16A: Component Integration (5-6 days) [requires all]
└── Wave 16B: Performance & Testing (7-8 days) [requires 16A]
```

**Parallelization Benefits:**
- Sequential: ~65 days
- With parallelization: ~45 days (30% reduction)

### Phase 2: Create Markdown Specifications

#### 3. Create Detailed Markdown Docs

For each wave, create a comprehensive markdown file in `docs/{project}/wave-{number}{letter}-{name}.md`.

**Template Structure:**

```markdown
# Wave {Number}{Letter}: {Name}

## Overview
Brief description and goals

## Dependencies
- **Requires**: Wave X (what must complete first)
- **Parallel with**: Wave Y, Z (can run simultaneously)
- **Enables**: Wave W (what this unblocks)

## User Stories
List of user stories from implementation plan

## Story {X}: {Story Name}

**As a** {role}
**I want** {goal}
**So that** {benefit}

### Acceptance Criteria
- [ ] Criterion 1
- [ ] Criterion 2

### Technical Implementation

#### Component Architecture

**1. ComponentName.java** (~XXX LOC)
```java
public class ComponentName {
    // Key methods with signatures
    public void exampleMethod();
}
```

#### Testing Strategy
- Unit tests
- Integration tests
- Performance benchmarks

#### Performance Benchmarks
- Operation X: < Y ms
- Memory usage: < Z MB

#### File Structure
```
mod/src/main/java/com/mineai/
├── component/
│   └── ComponentName.java
```

#### Dependencies to Add
```gradle
dependencies {
    implementation 'group:artifact:version'
}
```

## Success Criteria
- [ ] Tests passing
- [ ] Benchmarks met
- [ ] Documentation complete

## Estimated Effort
X-Y days (Z sprints)

## Next Wave
Wave {next} can begin after this completes
```

**Real Example:**

```markdown
# Wave 13A: Plan Document Data Structure

## Overview
Foundation wave implementing the core plan document data structure for
Agent 2.0's dynamic planning system.

## Dependencies
- **Requires**: None (foundation wave)
- **Parallel with**: Wave 13B, Wave 13C
- **Enables**: Wave 14A (Plan Review & Execution)

## User Story: Story 1.1 - Plan Document Data Structure

**As a** developer
**I want** a robust plan document data structure
**So that** agents can maintain and manipulate plans programmatically

### Acceptance Criteria
- [ ] Plan documents support CRUD operations on steps
- [ ] Steps have unique IDs, dependencies, and status tracking
- [ ] Plans can be serialized to/from markdown format
- [ ] Plans can be persisted to NBT for world saves
- [ ] Plan differences can be calculated for change detection

### Technical Implementation

#### Component Architecture

**1. PlanDocument.java** (~200 LOC)
```java
public class PlanDocument {
    private final UUID planId;
    private final String goalDescription;
    private final Map<String, PlanStep> steps;
    private final DirectedGraph<String> dependencyGraph;

    public PlanStep addStep(String description, List<String> dependsOn);
    public void removeStep(String stepId);
    public String toMarkdown();
    public static PlanDocument fromMarkdown(String markdown);
}
```

...
```

#### 4. Commit Markdown Documentation

```bash
git add docs/{project}/
git commit -m "docs: Add wave implementation specs for {project}

Wave structure:
- Wave {N}A-C: Foundation waves
- Wave {N+1}A-C: Enhancement waves
- ...

Total: X wave specs"
```

### Phase 3: Create Schaltwerk Draft Specs

#### 5. Create Draft Specs Using MCP

For each wave, create a **concise** draft spec that summarizes the markdown doc.

**MCP Command:**
```
mcp__schaltwerk__schaltwerk_spec_create
```

**Parameters:**
- `name`: Wave identifier (e.g., `wave-13a-plan-document-structure`)
- `content`: Concise summary (300-1000 words)

**Content Template:**

```markdown
# Wave {Number}{Letter}: {Name}

## Overview
{1-2 sentence description}

## Implementation Summary
{1 paragraph overview of what to implement}

## Key Components
1. **ComponentName.java** (~XXX LOC) - Brief description
2. **AnotherComponent.java** (~XXX LOC) - Brief description
...

## Dependencies
- {List required waves or libraries}

## Performance Targets
- Operation X: < Y ms
- Memory usage: < Z MB
...

## Testing Requirements
- Unit tests: {brief list}
- Integration tests: {brief list}
- Performance benchmarks: {brief list}

See full details in /docs/{project}/wave-{number}{letter}-{name}.md
```

**Real Example:**

```markdown
# Wave 13A: Plan Document Data Structure

## Overview
Foundation wave implementing the core plan document data structure for
Agent 2.0's dynamic planning system.

## Implementation Summary
Implement robust plan document data structure with CRUD operations,
dependency tracking, markdown serialization, NBT persistence, and change
detection. Uses JGraphT for DAG, Flexmark for markdown parsing.

## Key Components
1. **PlanDocument.java** (~200 LOC) - Core plan with steps and dependency graph
2. **PlanStep.java** (~150 LOC) - Individual step with state machine
3. **PlanSerializer.java** (~180 LOC) - Markdown/NBT conversion
4. **DependencyGraph.java** (~120 LOC) - DAG with topological sorting
5. **PlanDiff.java** (~100 LOC) - Change detection between plans

## Dependencies
- JGraphT 1.5.2 (graph algorithms)
- Flexmark 0.64.8 (markdown parsing)

## Performance Targets
- Plan creation: < 10ms for 50 steps
- Markdown parsing: < 5ms
- Dependency resolution: O(n), < 2ms for 50 steps

## Testing Requirements
- Unit tests: PlanDocumentTest, PlanSerializerTest, DependencyGraphTest
- Integration test: Plans persist across world save/load
- Performance benchmarks for all operations

See full details in /docs/agent-2-0/wave-13a-plan-document-structure.md
```

**Create ALL draft specs BEFORE starting any:**

```bash
# Create specs one by one (MCP requirement)
mcp__schaltwerk__schaltwerk_spec_create(
  name: "wave-13a-plan-document-structure",
  content: "..."
)

mcp__schaltwerk__schaltwerk_spec_create(
  name: "wave-13b-memory-indexing-system",
  content: "..."
)

mcp__schaltwerk__schaltwerk_spec_create(
  name: "wave-13c-protocol-management-system",
  content: "..."
)

# Continue for all waves...
```

### Phase 4: Start Parallel Waves

#### 6. Start Foundation Waves in Parallel

Once all draft specs are created, start the first parallel batch.

**MCP Command:**
```
mcp__schaltwerk__schaltwerk_draft_start
```

**Parameters:**
- `session_name`: Wave identifier
- `agent_type`: Usually "claude"
- `skip_permissions`: false (default)

**Start multiple waves simultaneously:**

```bash
# Start all foundation waves in parallel
mcp__schaltwerk__schaltwerk_draft_start(
  session_name: "wave-13a-plan-document-structure",
  agent_type: "claude"
)

mcp__schaltwerk__schaltwerk_draft_start(
  session_name: "wave-13b-memory-indexing-system",
  agent_type: "claude"
)

mcp__schaltwerk__schaltwerk_draft_start(
  session_name: "wave-13c-protocol-management-system",
  agent_type: "claude"
)
```

**What happens:**
- Schaltwerk creates a git worktree for each wave
- Creates branch `schaltwerk/wave-{number}{letter}-{name}`
- Spawns Claude agent with the draft spec as context
- Agent starts implementing autonomously

#### 7. Monitor Progress

**Check active sessions:**
```bash
mcp__schaltwerk__schaltwerk_list()
# or
mcp__schaltwerk__schaltwerk_get_current_tasks()
```

**Send follow-up messages if needed:**
```bash
mcp__schaltwerk__schaltwerk_send_message(
  session_name: "wave-13a-plan-document-structure",
  message: "Please prioritize the unit tests"
)
```

#### 8. Review Completed Waves

When a wave completes:

**Check diff summary:**
```bash
mcp__schaltwerk__schaltwerk_diff_summary(
  session: "wave-13a-plan-document-structure"
)
```

**Check specific file changes:**
```bash
mcp__schaltwerk__schaltwerk_diff_chunk(
  session: "wave-13a-plan-document-structure",
  path: "mod/src/main/java/com/mineai/planning/PlanDocument.java"
)
```

**Mark as reviewed:**
```bash
mcp__schaltwerk__schaltwerk_mark_session_reviewed(
  session_name: "wave-13a-plan-document-structure"
)
```

### Phase 5: Merge Waves

#### 9. Merge Completed Waves

**Option A: Merge via MCP (recommended)**
```bash
mcp__schaltwerk__schaltwerk_merge_session(
  session_name: "wave-13a-plan-document-structure",
  mode: "squash",
  commit_message: "feat(planning): Implement Wave 13A - Plan Document Structure

Implements core plan document data structure with:
- CRUD operations on steps with dependency tracking
- Markdown serialization with Flexmark
- NBT persistence for world saves
- Plan diff calculation for change detection

Components:
- PlanDocument.java (200 LOC)
- PlanStep.java (150 LOC)
- PlanSerializer.java (180 LOC)
- DependencyGraph.java (120 LOC)
- PlanDiff.java (100 LOC)

All tests passing, benchmarks met.

Closes wave-13a",
  cancel_after_merge: true
)
```

**Option B: Manual merge (for custom control)**
```bash
git checkout master
git merge --squash schaltwerk/wave-13a-plan-document-structure
git commit -m "feat(planning): Implement Wave 13A..."

# Clean up
mcp__schaltwerk__schaltwerk_cancel(
  session_name: "wave-13a-plan-document-structure"
)
```

#### 10. Start Next Wave Batch

Once dependencies are met, start the next parallel batch:

```bash
# After 13A, 13B, 13C complete, start 14A, 14B, 14C
mcp__schaltwerk__schaltwerk_draft_start(
  session_name: "wave-14a-plan-review-execution",
  agent_type: "claude"
)

mcp__schaltwerk__schaltwerk_draft_start(
  session_name: "wave-14b-memory-enhancement",
  agent_type: "claude"
)

mcp__schaltwerk__schaltwerk_draft_start(
  session_name: "wave-14c-context-engineering",
  agent_type: "claude"
)
```

## Complete Example: Agent 2.0 Waves

### Initial Setup

**1. Analyze AGENT_2_0_IMPLEMENTATION_PLAN.md and AGENT_2_0_TECHNICAL_IMPLEMENTATION_STORIES.md**

Identified 4 epics with multiple user stories:
- Epic 1: Dynamic Planning System (3 stories)
- Epic 2: Advanced Memory System (3 stories)
- Epic 3: Hierarchical Agent Architecture (3 stories)
- Epic 4: Extreme Context Engineering (3 stories)
- Epic 5: Integration and Performance (3 stories)

**2. Created wave structure:**

```
Foundation Waves (Parallel):
├── Wave 13A: Plan Document Structure
│   └── Story 1.1: Plan Document Data Structure
├── Wave 13B: Memory Indexing System
│   └── Story 2.1: Memory Indexing System
└── Wave 13C: Protocol Management System
    └── Story 4.1: Protocol Management System

Enhancement Waves (Parallel):
├── Wave 14A: Plan Review & Execution
│   ├── Story 1.2: Plan Review Engine
│   └── Story 1.3: Plan Execution Coordinator
├── Wave 14B: Memory Enhancement
│   ├── Story 2.2: Memory Lifecycle Management
│   └── Story 2.3: Context-Aware Memory Retrieval
└── Wave 14C: Context Engineering
    ├── Story 4.2: Dynamic Instruction Builder
    └── Story 4.3: Prompt Template Library

Orchestration Waves (Sequential):
├── Wave 15A: Orchestration Foundation
│   └── Story 3.1: Orchestrator Agent Base
└── Wave 15B: Sub-Agent Specialization
    ├── Story 3.2: Specialized Sub-Agent Framework
    └── Story 3.3: Task Decomposition Engine

Integration Waves (Sequential):
├── Wave 16A: Component Integration
│   └── Story 5.1: Component Integration Layer
└── Wave 16B: Performance & Testing
    ├── Story 5.2: Performance Optimization
    └── Story 5.3: End-to-End Testing Suite
```

### Markdown Creation

**3. Created 10 markdown files:**

```bash
docs/agent-2-0/
├── wave-13a-plan-document-structure.md
├── wave-13b-memory-indexing-system.md
├── wave-13c-protocol-management-system.md
├── wave-14a-plan-review-execution.md
├── wave-14b-memory-enhancement.md
├── wave-14c-context-engineering.md
├── wave-15a-orchestration-foundation.md
├── wave-15b-sub-agent-specialization.md
├── wave-16a-component-integration.md
└── wave-16b-performance-testing.md
```

**4. Committed documentation:**

```bash
git add docs/agent-2-0/
git commit -m "docs: Add Agent 2.0 wave implementation plans

Wave structure:
- Wave 13A-C: Foundation waves (planning, memory, protocols)
- Wave 14A-C: Enhancement waves (review/execution, lifecycle, context)
- Wave 15A-B: Orchestration waves (foundation, specialization)
- Wave 16A-B: Integration and performance waves

Total: 10 wave specs"
```

### Draft Spec Creation

**5. Created draft specs one by one:**

```python
# Wave 13A
mcp__schaltwerk__schaltwerk_spec_create(
  name="wave-13a-plan-document-structure",
  content="""
# Wave 13A: Plan Document Data Structure

## Overview
Foundation wave implementing the core plan document data structure for
Agent 2.0's dynamic planning system.

## Implementation Summary
Implement robust plan document data structure with CRUD operations,
dependency tracking, markdown serialization, NBT persistence, and change
detection. Uses JGraphT for DAG, Flexmark for markdown parsing.

## Key Components
1. **PlanDocument.java** (~200 LOC)
2. **PlanStep.java** (~150 LOC)
3. **PlanSerializer.java** (~180 LOC)
4. **DependencyGraph.java** (~120 LOC)
5. **PlanDiff.java** (~100 LOC)

## Dependencies
- JGraphT 1.5.2 (graph algorithms)
- Flexmark 0.64.8 (markdown parsing)

## Performance Targets
- Plan creation: < 10ms for 50 steps
- Markdown parsing: < 5ms
- Dependency resolution: O(n), < 2ms for 50 steps

## Testing Requirements
- Unit tests: PlanDocumentTest, PlanSerializerTest
- Integration test: Plans persist across world save/load
- Performance benchmarks for all operations

See full details in /docs/agent-2-0/wave-13a-plan-document-structure.md
"""
)

# Repeat for waves 13B, 13C, 14A, 14B, 14C, 15A, 15B, 16A, 16B
# ... (9 more spec_create calls)
```

### Parallel Execution

**6. Started foundation waves in parallel:**

```python
# Start Wave 13A
mcp__schaltwerk__schaltwerk_draft_start(
  session_name="wave-13a-plan-document-structure",
  agent_type="claude"
)

# Start Wave 13B
mcp__schaltwerk__schaltwerk_draft_start(
  session_name="wave-13b-memory-indexing-system",
  agent_type="claude"
)

# Start Wave 13C
mcp__schaltwerk__schaltwerk_draft_start(
  session_name="wave-13c-protocol-management-system",
  agent_type="claude"
)
```

**Result:**
```
✅ Wave 13A: Plan Document Structure - Active
✅ Wave 13B: Memory Indexing System - Active
✅ Wave 13C: Protocol Management System - Active
```

Three Claude agents now working simultaneously on foundation components.

**7. Monitor progress:**

```python
# Check all active sessions
mcp__schaltwerk__schaltwerk_list()

# Get detailed status
mcp__schaltwerk__schaltwerk_get_current_tasks(
  fields=["name", "status", "session_state", "branch", "ready_to_merge"]
)
```

**8. When waves complete, review and merge:**

```python
# Review Wave 13A
mcp__schaltwerk__schaltwerk_diff_summary(
  session="wave-13a-plan-document-structure"
)

# Mark as reviewed
mcp__schaltwerk__schaltwerk_mark_session_reviewed(
  session_name="wave-13a-plan-document-structure"
)

# Merge
mcp__schaltwerk__schaltwerk_merge_session(
  session_name="wave-13a-plan-document-structure",
  mode="squash",
  commit_message="feat(planning): Implement Wave 13A - Plan Document Structure...",
  cancel_after_merge=true
)
```

**9. Start next batch when dependencies met:**

```python
# After 13A completes, start 14A (which depends on 13A)
mcp__schaltwerk__schaltwerk_draft_start(
  session_name="wave-14a-plan-review-execution",
  agent_type="claude"
)

# After 13B completes, start 14B
# After 13C completes, start 14C
```

## Best Practices

### 1. Wave Sizing

**Optimal wave size:**
- 3-8 days of work
- 500-1500 LOC total
- 1-3 user stories
- Clear success criteria

**Too small:** Overhead of wave management exceeds benefit
**Too large:** Difficult to parallelize, longer feedback cycles

### 2. Dependency Management

**Clear dependency tracking:**
- Document in wave markdown: Requires/Parallel/Enables
- Don't start waves until dependencies complete
- Use `mcp__schaltwerk__schaltwerk_get_current_tasks` to check status

**Minimize dependencies:**
- Design for parallel execution
- Use interfaces to decouple components
- Defer integration to later waves

### 3. Draft Spec Quality

**Keep draft specs concise:**
- 300-1000 words
- Focus on "what" not "how"
- Reference full markdown for details
- Include concrete component names and LOC estimates

**Why concise?**
- Agent loads this as initial context
- Too much detail clutters decision-making
- Full markdown available for reference

### 4. Commit Strategy

**Commit markdown first:**
```bash
git add docs/
git commit -m "docs: Add wave specs"
```

**Why?**
- Ensures specs are versioned before execution
- Allows spec refinement before starting agents
- Provides single source of truth

### 5. Monitoring

**Regular checks:**
- Check progress every few hours
- Review diffs before marking as reviewed
- Send clarifications if agent is stuck

**Commands:**
```bash
# Quick status
mcp__schaltwerk__schaltwerk_list()

# Detailed status
mcp__schaltwerk__schaltwerk_get_current_tasks()

# Check specific wave
mcp__schaltwerk__schaltwerk_diff_summary(session="wave-13a-...")
```

### 6. Merging Strategy

**Squash merge benefits:**
- Clean main branch history
- Single commit per wave
- Easy to revert if needed

**Commit message template:**
```
{type}({scope}): Implement Wave {number}{letter} - {Name}

{1-2 sentence summary}

{Components list}

{Metrics/achievements}

Closes wave-{number}{letter}
```

### 7. Error Recovery

**If a wave fails:**

1. **Review the failure:**
```bash
mcp__schaltwerk__schaltwerk_diff_summary(session="wave-13a-...")
```

2. **Send clarification:**
```bash
mcp__schaltwerk__schaltwerk_send_message(
  session_name="wave-13a-...",
  message="The PlanDocument.java should use immutable data structures"
)
```

3. **If completely stuck, convert to spec for rework:**
```bash
mcp__schaltwerk__schaltwerk_convert_to_spec(
  session_name="wave-13a-..."
)
# Refine spec, then restart
mcp__schaltwerk__schaltwerk_draft_start(
  session_name="wave-13a-...",
  agent_type="claude"
)
```

## Common Pitfalls

### ❌ Starting waves before dependencies complete
**Problem:** Wave 14A starts before Wave 13A finishes
**Solution:** Check `ready_to_merge` status before starting dependent waves

### ❌ Draft specs too verbose
**Problem:** 5000-word draft spec with every implementation detail
**Solution:** Keep to 300-1000 words, reference markdown for details

### ❌ Not committing markdown before starting
**Problem:** Markdown only exists in memory, not versioned
**Solution:** Always `git commit` markdown before creating draft specs

### ❌ Ignoring wave completion
**Problem:** Wave completes but sits unmerged for days
**Solution:** Set up monitoring routine, merge within 24 hours

### ❌ Too many parallel waves
**Problem:** Starting 10 waves simultaneously
**Solution:** Limit to 3-5 parallel waves max to maintain oversight

## Quick Reference

### MCP Commands

```bash
# Create draft spec
mcp__schaltwerk__schaltwerk_spec_create(name, content)

# Start wave
mcp__schaltwerk__schaltwerk_draft_start(session_name, agent_type)

# List waves
mcp__schaltwerk__schaltwerk_list()
mcp__schaltwerk__schaltwerk_get_current_tasks(fields, status_filter)

# Send message
mcp__schaltwerk__schaltwerk_send_message(session_name, message)

# Review
mcp__schaltwerk__schaltwerk_diff_summary(session)
mcp__schaltwerk__schaltwerk_diff_chunk(session, path)
mcp__schaltwerk__schaltwerk_mark_session_reviewed(session_name)

# Merge
mcp__schaltwerk__schaltwerk_merge_session(session_name, mode, commit_message, cancel_after_merge)

# Cancel
mcp__schaltwerk__schaltwerk_cancel(session_name, force)

# Convert to spec for rework
mcp__schaltwerk__schaltwerk_convert_to_spec(session_name)
```

### Wave Lifecycle

```
1. Analysis
   └─> Identify epics and user stories

2. Planning
   └─> Group into waves with dependencies

3. Documentation
   └─> Create markdown specs
   └─> Commit to git

4. Spec Creation
   └─> Create concise draft specs via MCP

5. Execution
   └─> Start waves in parallel batches
   └─> Monitor progress

6. Review
   └─> Check diffs
   └─> Mark as reviewed

7. Merge
   └─> Squash merge to master
   └─> Cancel session

8. Next Batch
   └─> Start dependent waves
```

## Conclusion

The WAVE workflow with Schaltwerk MCP enables:
- **30-50% reduction** in total implementation time through parallelization
- **Clear progress tracking** with isolated branches
- **Autonomous development** with AI agents
- **Incremental integration** reducing merge conflicts
- **Easy rollback** with squash merge strategy

By following this guide, you can effectively manage complex, multi-month projects with parallel development streams that converge cleanly into production-ready features.
