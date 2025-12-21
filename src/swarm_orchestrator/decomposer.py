"""
Task decomposer using Claude CLI to break complex tasks into subtasks.

Uses the `claude` CLI for authentication, so no API key is needed -
it uses your existing Claude Code login (Max subscription, etc.).

Research-backed decomposition based on:
- MAKER (arXiv:2511.09030): Maximal Agentic Decomposition for atomic subtasks
- Agentless (arXiv:2407.01489): Hierarchical localization and scope reduction
- Select-Then-Decompose (arXiv:2510.17922): Adaptive decomposition strategies
- SWE-bench: Real-world software patch statistics (1.7 files, 32.8 LOC average)
"""

import json
import re
import subprocess
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CodeInsight:
    """
    Represents a code analysis finding from exploration.

    Captures relevant files, patterns, and dependencies discovered
    during codebase exploration that can guide agent implementation.
    """
    file_path: str
    description: str = ""
    patterns: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to dict for prompt injection."""
        return {
            "file_path": self.file_path,
            "description": self.description,
            "patterns": self.patterns,
            "dependencies": self.dependencies,
        }


@dataclass
class WebResearchFinding:
    """
    Represents a web research finding from exploration.

    Captures API documentation, examples, and best practices
    discovered through web research.
    """
    source: str
    summary: str = ""
    relevance: str = ""

    def to_dict(self) -> dict:
        """Serialize to dict for prompt injection."""
        return {
            "source": self.source,
            "summary": self.summary,
            "relevance": self.relevance,
        }


@dataclass
class ExplorationResult:
    """
    Container for exploration findings to pass to subagents.

    Combines code insights, web research findings, and a synthesized
    context summary that provides guidance for agent implementation.
    """
    code_insights: list[CodeInsight] = field(default_factory=list)
    web_findings: list[WebResearchFinding] = field(default_factory=list)
    context_summary: str = ""

    def to_dict(self) -> dict:
        """Serialize entire result to dict for prompt embedding."""
        return {
            "code_insights": [i.to_dict() for i in self.code_insights],
            "web_findings": [f.to_dict() for f in self.web_findings],
            "context_summary": self.context_summary,
        }


# Research-backed decomposition prompt
DECOMPOSE_PROMPT = """You are a task decomposition expert for AI coding agents.

# Your Goal
Break down the user's request into atomic, sequential subtasks that AI agents can implement with high accuracy.

# Research-Backed Guidelines (IMPORTANT)

## Atomicity Test (from MAKER research)
A subtask is atomic when:
> "A correct solution is likely to be sampled, and no incorrect solution is more likely."

Ask yourself: "Can an AI agent complete this correctly >90% of the time with focused context?"

## Optimal Scope Per Subtask (from SWE-bench & Agentless research)
Real-world software patches average 1.7 files and 32.8 lines changed. Use this as guidance:

| Metric              | Target    | Hard Limit | Rationale                     |
|---------------------|-----------|------------|-------------------------------|
| Files changed       | 1-2       | 3 max      | Maintains cognitive focus     |
| Functions modified  | 1-3       | 5 max      | Manageable complexity         |
| Lines of code       | 30-80     | 150 max    | Avoids context degradation    |
| New dependencies    | 0-1       | 2 max      | Reduces integration risk      |

If a subtask exceeds these limits, break it down further.

## ReAct-Style Structure (Implement + Verify)
Each subtask MUST include both implementation AND verification bundled together.
Do NOT create separate "implement X" and "test X" subtasks.

Good: "Add user authentication endpoint with tests for valid/invalid credentials"
Bad: "Add user authentication endpoint" + "Write tests for authentication"

## Hierarchical Localization (from Agentless research)
Before detailing a subtask, identify:
1. Which files will be modified/created
2. Which functions/classes are involved
3. What the specific changes are

This helps agents focus and reduces context pollution.

## Sequential Dependencies
Subtasks execute SEQUENTIALLY. Each subtask:
- Can assume all previous subtasks are complete and merged
- Should build naturally on prior work
- Must be independently verifiable

Order from foundation → features → integration:
1. Data models / types first
2. Core logic second
3. API / interface layer third
4. Integration / glue code last

# Output Format

Output ONLY valid JSON (no markdown wrapping, no explanation outside JSON):

{{
  "is_atomic": true or false,
  "reasoning": "Brief explanation of decomposition strategy (1-2 sentences)",
  "subtasks": [
    {{
      "id": "subtask-id",
      "title": "Short descriptive title (3-6 words)",
      "description": "What to implement and why (1-2 sentences)",
      "scope": {{
        "files": ["path/to/file1.py", "path/to/file2.py"],
        "estimated_loc": 50,
        "functions": ["function_name", "ClassName.method"]
      }},
      "implementation": "Specific implementation goal - WHAT not HOW",
      "verification": "How to verify: tests to write, behavior to check",
      "success_criteria": [
        "Criterion 1: specific, measurable",
        "Criterion 2: specific, measurable"
      ],
      "depends_on": []
    }},
    {{
      "id": "subtask-2",
      "depends_on": ["subtask-id"],
      ...
    }}
  ]
}}

# Rules

1. **IDs**: lowercase with hyphens (e.g., "add-auth-endpoint", "fix-login-bug")

2. **For ATOMIC tasks** (simple, single-focus):
   - Return is_atomic: true
   - Still provide full subtask structure with scope and verification

3. **For COMPLEX tasks**:
   - Break into 2-5 subtasks maximum
   - Each subtask must be atomic by the definition above
   - If >5 subtasks needed, note this but proceed

4. **Scope Estimation**:
   - Be realistic about files and LOC
   - When uncertain, estimate conservatively (smaller)
   - List actual file paths if you can infer them from context

5. **Implementation Field**:
   - Describe WHAT needs to be done, not HOW
   - DO NOT include code snippets or specific approaches
   - Let agents discover their own solutions

6. **Verification Field**:
   - Always include testing requirements
   - Specify what behavior to verify
   - Include edge cases to handle

7. **Success Criteria**:
   - Must be specific and measurable
   - Include both functional and quality criteria
   - "Tests pass" should always be one criterion

# Examples

## Example: Good Atomic Task
{{
  "is_atomic": true,
  "reasoning": "Single endpoint addition with clear scope, estimated ~40 LOC",
  "subtasks": [{{
    "id": "add-health-endpoint",
    "title": "Add health check endpoint",
    "description": "Add a /health endpoint for monitoring service availability",
    "scope": {{
      "files": ["src/api/routes.py", "tests/test_routes.py"],
      "estimated_loc": 40,
      "functions": ["health_check", "test_health_endpoint"]
    }},
    "implementation": "Create GET /health endpoint returning service status and version",
    "verification": "Test returns 200 with valid JSON, test handles edge cases",
    "success_criteria": [
      "GET /health returns 200 with status and version",
      "Response time < 100ms",
      "Unit tests pass"
    ],
    "depends_on": []
  }}]
}}

## Example: Good Complex Task Decomposition
{{
  "is_atomic": false,
  "reasoning": "User auth requires data model, core logic, then API layer - natural 3-subtask split",
  "subtasks": [
    {{
      "id": "add-user-model",
      "title": "Add User data model",
      "description": "Create User model with authentication fields",
      "scope": {{
        "files": ["src/models/user.py", "tests/test_models.py"],
        "estimated_loc": 60,
        "functions": ["User", "test_user_creation", "test_password_hashing"]
      }},
      "implementation": "Create User model with email, hashed password, and timestamps",
      "verification": "Test user creation, password hashing, and validation",
      "success_criteria": [
        "User model with required fields exists",
        "Passwords are hashed, never stored plain",
        "Unit tests pass"
      ],
      "depends_on": []
    }},
    {{
      "id": "add-auth-service",
      "title": "Add authentication service",
      "description": "Implement login/logout logic using the User model",
      "scope": {{
        "files": ["src/services/auth.py", "tests/test_auth.py"],
        "estimated_loc": 80,
        "functions": ["AuthService.login", "AuthService.logout", "AuthService.verify_token"]
      }},
      "implementation": "Create AuthService with JWT-based login, logout, and token verification",
      "verification": "Test valid login, invalid credentials, token expiry, logout",
      "success_criteria": [
        "Login returns valid JWT for correct credentials",
        "Invalid credentials return appropriate error",
        "Token verification works correctly",
        "Unit tests pass"
      ],
      "depends_on": ["add-user-model"]
    }},
    {{
      "id": "add-auth-endpoints",
      "title": "Add auth API endpoints",
      "description": "Expose authentication via REST API endpoints",
      "scope": {{
        "files": ["src/api/auth_routes.py", "tests/test_auth_routes.py"],
        "estimated_loc": 70,
        "functions": ["login_endpoint", "logout_endpoint", "me_endpoint"]
      }},
      "implementation": "Create POST /auth/login, POST /auth/logout, GET /auth/me endpoints",
      "verification": "Integration tests for all endpoints with valid and invalid inputs",
      "success_criteria": [
        "All endpoints return correct status codes",
        "Error responses are consistent",
        "Integration tests pass"
      ],
      "depends_on": ["add-auth-service"]
    }}
  ]
}}

# User's Request

{query}

Analyze this request and produce the optimal decomposition following the guidelines above.
"""


@dataclass
class SubtaskScope:
    """
    Defines the expected scope of a subtask.

    Based on research from SWE-bench and Agentless:
    - Target: 1-2 files, 30-80 LOC
    - Hard limits: 3 files max, 150 LOC max
    """
    files: list[str] = field(default_factory=list)
    estimated_loc: int = 50  # Default to middle of target range
    functions: list[str] = field(default_factory=list)

    def is_within_limits(self) -> bool:
        """Check if scope is within research-backed limits."""
        return (
            len(self.files) <= 3 and
            self.estimated_loc <= 150 and
            len(self.functions) <= 5
        )

    def get_warnings(self) -> list[str]:
        """Get warnings if scope exceeds targets (but not hard limits)."""
        warnings = []
        if len(self.files) > 2:
            warnings.append(f"Scope spans {len(self.files)} files (target: 1-2)")
        if self.estimated_loc > 80:
            warnings.append(f"Estimated {self.estimated_loc} LOC (target: 30-80)")
        if len(self.functions) > 3:
            warnings.append(f"Touches {len(self.functions)} functions (target: 1-3)")
        return warnings


@dataclass
class Subtask:
    """
    A single subtask with scope and verification requirements.

    Follows ReAct-style structure: each subtask includes both
    implementation AND verification bundled together.
    """
    id: str
    title: str
    description: str
    scope: SubtaskScope
    implementation: str  # WHAT to do (not HOW)
    verification: str  # How to verify (tests, criteria)
    success_criteria: list[str]
    depends_on: list[str] = field(default_factory=list)

    @property
    def prompt(self) -> str:
        """
        Generate the full prompt for agents.

        This is the prompt that gets sent to competing agents.
        """
        scope_info = ""
        if self.scope.files:
            scope_info += f"- **Target files**: {', '.join(self.scope.files)}\n"
        if self.scope.estimated_loc:
            scope_info += f"- **Estimated scope**: ~{self.scope.estimated_loc} lines of code\n"
        if self.scope.functions:
            scope_info += f"- **Focus areas**: {', '.join(self.scope.functions)}\n"

        criteria_list = '\n'.join(f'- [ ] {c}' for c in self.success_criteria)

        return f"""# Task: {self.title}

## Description
{self.description}

## Scope
{scope_info}
## Implementation Goal
{self.implementation}

## Verification Requirements
{self.verification}

## Success Criteria
{criteria_list}

## Important Guidelines
- Stay within the specified scope - avoid modifying unrelated files
- Include tests as part of your implementation
- Commit when all success criteria are met
- Prefer minimal, focused changes over large refactors
"""


@dataclass
class DecompositionResult:
    """Result of task decomposition."""
    is_atomic: bool
    subtasks: list[Subtask]
    original_query: str
    reasoning: str = ""  # Explanation of decomposition strategy

    def get_scope_warnings(self) -> dict[str, list[str]]:
        """Get scope warnings for all subtasks."""
        warnings = {}
        for subtask in self.subtasks:
            subtask_warnings = subtask.scope.get_warnings()
            if subtask_warnings:
                warnings[subtask.id] = subtask_warnings
        return warnings

    def total_estimated_loc(self) -> int:
        """Get total estimated lines of code across all subtasks."""
        return sum(st.scope.estimated_loc for st in self.subtasks)


class DecomposerError(Exception):
    """Raised when decomposition fails."""
    pass


class Decomposer:
    """
    Decomposes user queries into subtasks using Claude.

    Uses the `claude` CLI for authentication, so it works with:
    - Claude Max subscription
    - Claude Pro subscription
    - API key (if configured in claude CLI)

    No separate API key needed!
    """

    def __init__(self, use_api: bool = False, model: Optional[str] = None, timeout: int = 120, cli_tool: str = "claude"):
        """
        Initialize the decomposer.

        Args:
            use_api: If True, use Anthropic API directly (requires ANTHROPIC_API_KEY).
                     If False (default), use CLI (uses your login).
            model: Model to use (only applies when use_api=True)
            timeout: Timeout in seconds for CLI calls (default: 120)
            cli_tool: CLI tool to use for decomposition ("claude" or "opencode")
        """
        self.use_api = use_api
        self.model = model or "claude-sonnet-4-20250514"
        self.timeout = timeout
        self.cli_tool = cli_tool
        self._api_client = None

    def _get_api_client(self):
        """Lazily create API client only if needed."""
        if self._api_client is None:
            from anthropic import Anthropic
            self._api_client = Anthropic()
        return self._api_client

    def decompose(
        self,
        query: str,
        exploration_result: Optional[ExplorationResult] = None,
    ) -> DecompositionResult:
        """Decompose a query into subtasks."""
        # Build exploration context section if available
        exploration_context = self._format_exploration_for_decomposer(exploration_result)
        prompt = DECOMPOSE_PROMPT.format(query=query + exploration_context)

        if self.use_api:
            text = self._call_api(prompt)
        else:
            text = self._call_cli(prompt)

        return self._parse_response(text, query)

    def _format_exploration_for_decomposer(
        self,
        exploration_result: Optional[ExplorationResult],
    ) -> str:
        """Format exploration results for the decomposition prompt."""
        if not exploration_result:
            return ""

        sections = ["\n\n# Exploration Findings\nThe following context was gathered from codebase exploration:"]

        if exploration_result.context_summary:
            sections.append(f"\n## Summary\n{exploration_result.context_summary}")

        if exploration_result.code_insights:
            insights = "\n## Relevant Files"
            for insight in exploration_result.code_insights:
                insights += f"\n- {insight.file_path}: {insight.description}"
                if insight.patterns:
                    insights += f" (patterns: {', '.join(insight.patterns)})"
                if insight.dependencies:
                    insights += f" (deps: {', '.join(insight.dependencies)})"
            sections.append(insights)

        return "\n".join(sections)

    def _build_command(self, prompt: str) -> list[str]:
        """Build CLI command based on configured tool."""
        if self.cli_tool == "opencode":
            return ["opencode", "-p", prompt]
        if self.cli_tool == "cursor":
            return ["cursor-agent", "-p", prompt]
        # Default to claude
        return ["claude", "-p", prompt, "--output-format", "text"]

    def _call_cli(self, prompt: str) -> str:
        """
        Call the CLI tool for decomposition.

        Uses your existing CLI tool authentication (Max/Pro subscription).
        """
        cmd = self._build_command(prompt)
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            if result.returncode != 0:
                error_msg = result.stderr or "Unknown error"
                raise DecomposerError(f"{self.cli_tool} CLI failed: {error_msg}")

            return result.stdout

        except FileNotFoundError:
            raise DecomposerError(
                f"{self.cli_tool} CLI not found. Please install the required CLI tool."
            )
        except subprocess.TimeoutExpired:
            raise DecomposerError(f"{self.cli_tool} CLI timed out after {self.timeout} seconds")

    def _call_api(self, prompt: str) -> str:
        """Call Claude via the Anthropic API (requires API key)."""
        client = self._get_api_client()
        response = client.messages.create(
            model=self.model,
            max_tokens=4096,  # Increased for structured decomposition output
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def _parse_scope(self, scope_data: dict | None) -> SubtaskScope:
        """Parse scope data into SubtaskScope dataclass."""
        if not scope_data:
            return SubtaskScope()

        return SubtaskScope(
            files=scope_data.get("files", []),
            estimated_loc=scope_data.get("estimated_loc", 50),
            functions=scope_data.get("functions", []),
        )

    def _parse_subtask(self, st: dict, index: int, original_query: str) -> Subtask:
        """Parse a single subtask from JSON data."""
        # Parse scope
        scope = self._parse_scope(st.get("scope"))

        # Get success criteria (ensure it's a list)
        success_criteria = st.get("success_criteria", [])
        if isinstance(success_criteria, str):
            success_criteria = [success_criteria]

        # Get depends_on (ensure it's a list)
        depends_on = st.get("depends_on", [])
        if isinstance(depends_on, str):
            depends_on = [depends_on] if depends_on else []

        return Subtask(
            id=st.get("id", f"task-{index}"),
            title=st.get("title", st.get("description", f"Task {index + 1}")[:50]),
            description=st.get("description", ""),
            scope=scope,
            implementation=st.get("implementation", st.get("prompt", original_query)),
            verification=st.get("verification", "Verify the implementation works correctly"),
            success_criteria=success_criteria if success_criteria else ["Implementation complete", "Tests pass"],
            depends_on=depends_on,
        )

    def _parse_response(self, text: str, original_query: str) -> DecompositionResult:
        """Parse the JSON response from Claude."""
        # Extract JSON (handle potential markdown wrapping)
        json_match = re.search(r"\{[\s\S]*\}", text)
        if not json_match:
            raise DecomposerError(f"Could not parse JSON from response: {text}")

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            raise DecomposerError(f"Invalid JSON in response: {e}")

        # Validate required fields
        if "is_atomic" not in data or "subtasks" not in data:
            raise DecomposerError(f"Missing required fields in response: {data}")

        # Convert to dataclasses with enhanced parsing
        subtasks = [
            self._parse_subtask(st, i, original_query)
            for i, st in enumerate(data["subtasks"])
        ]

        return DecompositionResult(
            is_atomic=data["is_atomic"],
            subtasks=subtasks,
            original_query=original_query,
            reasoning=data.get("reasoning", ""),
        )


def decompose_task(
    query: str,
    use_api: bool = False,
    timeout: int = 120,
    exploration_result: Optional[ExplorationResult] = None,
    cli_tool: str = "claude",
) -> DecompositionResult:
    """
    Convenience function to decompose a task.

    Args:
        query: The task to decompose
        use_api: If True, use Anthropic API (requires ANTHROPIC_API_KEY).
                 If False (default), use CLI (uses your login).
        timeout: Timeout in seconds for CLI calls (default: 120)
        exploration_result: Optional exploration findings to provide context
        cli_tool: CLI tool to use for decomposition ("claude" or "opencode")
    """
    decomposer = Decomposer(use_api=use_api, timeout=timeout, cli_tool=cli_tool)
    return decomposer.decompose(query, exploration_result=exploration_result)


def validate_decomposition(result: DecompositionResult) -> tuple[bool, list[str]]:
    """
    Validate a decomposition result against research-backed guidelines.

    Returns:
        Tuple of (is_valid, list_of_warnings)
        - is_valid: True if all subtasks are within hard limits
        - warnings: List of warning messages for soft limit violations
    """
    warnings = []
    is_valid = True

    # Check each subtask
    for subtask in result.subtasks:
        scope = subtask.scope

        # Hard limit checks (fail validation)
        if len(scope.files) > 3:
            warnings.append(
                f"[{subtask.id}] EXCEEDS LIMIT: {len(scope.files)} files (max: 3)"
            )
            is_valid = False

        if scope.estimated_loc > 150:
            warnings.append(
                f"[{subtask.id}] EXCEEDS LIMIT: {scope.estimated_loc} LOC (max: 150)"
            )
            is_valid = False

        if len(scope.functions) > 5:
            warnings.append(
                f"[{subtask.id}] EXCEEDS LIMIT: {len(scope.functions)} functions (max: 5)"
            )
            is_valid = False

        # Soft limit checks (warnings only)
        subtask_warnings = scope.get_warnings()
        for w in subtask_warnings:
            warnings.append(f"[{subtask.id}] {w}")

    # Check total LOC across all subtasks
    total_loc = result.total_estimated_loc()
    if total_loc > 500:
        warnings.append(
            f"Total estimated LOC ({total_loc}) is high - consider if task should be split"
        )

    # Check for missing dependencies
    subtask_ids = {st.id for st in result.subtasks}
    for subtask in result.subtasks:
        for dep in subtask.depends_on:
            if dep not in subtask_ids:
                warnings.append(
                    f"[{subtask.id}] depends on unknown subtask: {dep}"
                )
                is_valid = False

    return is_valid, warnings


# Research-backed scope limits for reference
SCOPE_LIMITS = {
    "files": {"target": 2, "max": 3},
    "estimated_loc": {"target": 80, "max": 150},
    "functions": {"target": 3, "max": 5},
}
