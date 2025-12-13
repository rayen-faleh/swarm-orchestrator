"""
Task decomposer using Claude API to break complex tasks into subtasks.
"""

import json
import re
from dataclasses import dataclass
from anthropic import Anthropic


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
      "prompt": "Detailed prompt for Claude Code agent with specific instructions"
    }}
  ]
}}

Rules:
- For ATOMIC tasks: Return exactly one subtask with the original task as the prompt
- For COMPLEX tasks: Break into 2-5 independent subtasks
- Each subtask should be completable independently
- Prompts should be specific and actionable
- IDs should be lowercase with hyphens (e.g., "add-auth", "fix-login")

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


class Decomposer:
    """Decomposes user queries into subtasks using Claude."""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.client = Anthropic()
        self.model = model

    def decompose(self, query: str) -> DecompositionResult:
        """Decompose a query into subtasks."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            messages=[
                {"role": "user", "content": DECOMPOSE_PROMPT.format(query=query)}
            ],
        )

        # Extract text content
        text = response.content[0].text

        # Parse JSON (handle potential markdown wrapping)
        json_match = re.search(r"\{[\s\S]*\}", text)
        if not json_match:
            raise ValueError(f"Could not parse JSON from response: {text}")

        data = json.loads(json_match.group())

        # Convert to dataclasses
        subtasks = [
            Subtask(
                id=st["id"],
                description=st["description"],
                prompt=st["prompt"],
            )
            for st in data["subtasks"]
        ]

        return DecompositionResult(
            is_atomic=data["is_atomic"],
            subtasks=subtasks,
            original_query=query,
        )


def decompose_task(query: str) -> DecompositionResult:
    """Convenience function to decompose a task."""
    decomposer = Decomposer()
    return decomposer.decompose(query)
