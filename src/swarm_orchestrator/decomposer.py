"""
Task decomposer using Claude CLI to break complex tasks into subtasks.

Uses the `claude` CLI for authentication, so no API key is needed -
it uses your existing Claude Code login (Max subscription, etc.).
"""

import json
import re
import subprocess
from dataclasses import dataclass
from typing import Optional


DECOMPOSE_PROMPT = """You are a task decomposer for software engineering tasks.

Given a coding task, analyze it and determine if it should be:
1. ATOMIC - A single, focused change (one feature, one bug fix, one refactor)
2. COMPLEX - Multiple independent changes that should be done separately

Output ONLY valid JSON (no markdown, no explanation):
{{
  "is_atomic": true or false,
  "subtasks": [
    {{
      "id": "task-1",
      "description": "Brief description of what to do",
      "prompt": "The task description - WHAT to do, not HOW"
    }}
  ]
}}

Rules:
- For ATOMIC tasks: Return exactly one subtask with the original task as the prompt
- For COMPLEX tasks: Break into 2-5 independent subtasks
- Each subtask should be completable independently
- IDs should be lowercase with hyphens (e.g., "add-auth", "fix-login")

IMPORTANT - Prompt Guidelines:
- The prompt should describe WHAT needs to be done, not HOW to do it
- DO NOT include implementation suggestions, code snippets, or specific approaches
- DO NOT prescribe which files to modify or what functions to create
- Let the agents independently discover their own solutions
- Keep prompts minimal and goal-focused

Example of BAD prompt (too prescriptive):
"Add a timeout parameter to the run function. Modify cli.py to accept --timeout flag, pass it to Orchestrator constructor, and update the Orchestrator class to use self.timeout in wait loops."

Example of GOOD prompt (goal-focused):
"The --timeout flag is being ignored. Fix the timeout functionality so it works as expected."

TASK: {query}
"""


@dataclass
class Subtask:
    id: str
    description: str
    prompt: str


@dataclass
class DecompositionResult:
    is_atomic: bool
    subtasks: list[Subtask]
    original_query: str


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

    def __init__(self, use_api: bool = False, model: Optional[str] = None):
        """
        Initialize the decomposer.

        Args:
            use_api: If True, use Anthropic API directly (requires ANTHROPIC_API_KEY).
                     If False (default), use claude CLI (uses your login).
            model: Model to use (only applies when use_api=True)
        """
        self.use_api = use_api
        self.model = model or "claude-sonnet-4-20250514"
        self._api_client = None

    def _get_api_client(self):
        """Lazily create API client only if needed."""
        if self._api_client is None:
            from anthropic import Anthropic
            self._api_client = Anthropic()
        return self._api_client

    def decompose(self, query: str) -> DecompositionResult:
        """Decompose a query into subtasks."""
        prompt = DECOMPOSE_PROMPT.format(query=query)

        if self.use_api:
            text = self._call_api(prompt)
        else:
            text = self._call_cli(prompt)

        return self._parse_response(text, query)

    def _call_cli(self, prompt: str) -> str:
        """
        Call Claude via the CLI.

        Uses your existing Claude Code authentication (Max/Pro subscription).
        """
        try:
            result = subprocess.run(
                ["claude", "-p", prompt, "--output-format", "text"],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                error_msg = result.stderr or "Unknown error"
                raise DecomposerError(f"Claude CLI failed: {error_msg}")

            return result.stdout

        except FileNotFoundError:
            raise DecomposerError(
                "Claude CLI not found. Please install Claude Code: "
                "https://claude.ai/download"
            )
        except subprocess.TimeoutExpired:
            raise DecomposerError("Claude CLI timed out after 60 seconds")

    def _call_api(self, prompt: str) -> str:
        """Call Claude via the Anthropic API (requires API key)."""
        client = self._get_api_client()
        response = client.messages.create(
            model=self.model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

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

        # Convert to dataclasses
        subtasks = [
            Subtask(
                id=st.get("id", f"task-{i}"),
                description=st.get("description", ""),
                prompt=st.get("prompt", original_query),
            )
            for i, st in enumerate(data["subtasks"])
        ]

        return DecompositionResult(
            is_atomic=data["is_atomic"],
            subtasks=subtasks,
            original_query=original_query,
        )


def decompose_task(query: str, use_api: bool = False) -> DecompositionResult:
    """
    Convenience function to decompose a task.

    Args:
        query: The task to decompose
        use_api: If True, use Anthropic API (requires ANTHROPIC_API_KEY).
                 If False (default), use claude CLI (uses your login).
    """
    decomposer = Decomposer(use_api=use_api)
    return decomposer.decompose(query)
