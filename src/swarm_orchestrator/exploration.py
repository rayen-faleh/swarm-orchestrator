"""
Exploration executor for analyzing codebase and performing web research.

Uses Claude CLI to explore the codebase and gather context before
decomposition to help agents understand relevant patterns and files.
"""

import json
import re
import subprocess
from typing import Optional

from .decomposer import CodeInsight, WebResearchFinding, ExplorationResult


# Keywords indicating task complexity that warrants exploration
COMPLEX_KEYWORDS = [
    "implement", "integrate", "refactor", "add", "create", "build",
    "authentication", "api", "database", "cache", "websocket", "oauth",
    "graphql", "async", "distributed", "microservice", "payment", "stripe",
    "notification", "real-time", "rate limit", "caching",
]

# Keywords indicating simple tasks
SIMPLE_KEYWORDS = [
    "fix typo", "update version", "add comment", "rename", "format",
    "fix null", "fix undefined", "update readme", "bump version",
]


def needs_exploration(query: str) -> bool:
    """
    Detect if a task requires exploration or is straightforward.

    Uses keyword-based heuristics to determine complexity.
    Simple tasks (typos, version bumps) skip exploration.
    Complex tasks (new features, integrations) benefit from it.

    Args:
        query: The task description

    Returns:
        True if exploration would be beneficial, False otherwise
    """
    query_lower = query.lower()

    # Check for simple task indicators first
    for keyword in SIMPLE_KEYWORDS:
        if keyword in query_lower:
            return False

    # Check for complex task indicators
    for keyword in COMPLEX_KEYWORDS:
        if keyword in query_lower:
            return True

    # Default: explore if query is longer (likely more complex)
    return len(query.split()) > 10


# Exploration prompt for Claude CLI
EXPLORATION_PROMPT = """You are a codebase exploration expert. Analyze the task and identify relevant code patterns, files, and any external documentation that would help.

# Task
{query}

# Instructions
1. Identify relevant files in the codebase that relate to this task
2. Note any patterns, architectures, or conventions used
3. List dependencies that might be relevant
4. If web research is enabled, suggest relevant documentation URLs

{web_research_section}

# Output Format
Return ONLY valid JSON (no markdown wrapping):
{{
    "needs_exploration": true/false,
    "code_insights": [
        {{
            "file_path": "path/to/file.py",
            "description": "What this file does",
            "patterns": ["pattern1", "pattern2"],
            "dependencies": ["dep1", "dep2"]
        }}
    ],
    "web_findings": [
        {{
            "source": "https://docs.example.com",
            "summary": "What this documentation covers",
            "relevance": "Why it's relevant to the task"
        }}
    ],
    "context_summary": "Brief synthesis of findings (1-2 sentences)"
}}

# Rules
- Set needs_exploration to false for trivial tasks (typo fixes, version bumps)
- code_insights should list 1-5 most relevant files
- web_findings should only include highly relevant documentation
- context_summary should help an AI agent understand the context quickly
"""


class ExplorationExecutor:
    """
    Explores codebase and performs web research before decomposition.

    Uses Claude CLI to analyze the codebase and identify relevant
    patterns, files, and external documentation for a given task.
    """

    def __init__(
        self,
        timeout: int = 120,
        enable_web_research: bool = False,
        working_dir: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize the exploration executor.

        Args:
            timeout: Timeout in seconds for Claude CLI calls
            enable_web_research: Whether to include web research in exploration
            working_dir: Working directory for codebase analysis
            model: Model to use for exploration (e.g., 'claude-3-haiku')
        """
        self.timeout = timeout
        self.enable_web_research = enable_web_research
        self.working_dir = working_dir
        self.model = model

    def explore(self, query: str) -> ExplorationResult:
        """
        Explore the codebase for a given task.

        Calls Claude CLI with an exploration-focused prompt to analyze
        relevant code patterns and identify key files.

        Args:
            query: The task to explore

        Returns:
            ExplorationResult with code insights, web findings, and context
        """
        prompt = self._build_prompt(query)

        try:
            response = self._call_cli(prompt)
            return self._parse_response(response)
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
            # Return empty result on errors rather than crashing
            return ExplorationResult()
        except Exception:
            return ExplorationResult()

    def _build_prompt(self, query: str) -> str:
        """Build the exploration prompt."""
        web_section = ""
        if self.enable_web_research:
            web_section = """
## Web Research
Search for relevant documentation, API references, or best practices.
Include URLs to official documentation that would help with implementation.
"""
        return EXPLORATION_PROMPT.format(
            query=query,
            web_research_section=web_section,
        )

    def _call_cli(self, prompt: str) -> str:
        """
        Call Claude via the CLI.

        Uses your existing Claude Code authentication (Max/Pro subscription).
        """
        cmd = ["claude", "-p", prompt, "--output-format", "text"]
        if self.model:
            cmd.extend(["--model", self.model])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.timeout,
            cwd=self.working_dir,
        )

        if result.returncode != 0:
            return ""

        return result.stdout

    def _parse_response(self, text: str) -> ExplorationResult:
        """Parse the JSON response from Claude."""
        if not text:
            return ExplorationResult()

        # Extract JSON (handle potential markdown wrapping)
        json_match = re.search(r"\{[\s\S]*\}", text)
        if not json_match:
            return ExplorationResult()

        data = json.loads(json_match.group())

        # Parse code insights
        code_insights = []
        for insight_data in data.get("code_insights", []):
            code_insights.append(CodeInsight(
                file_path=insight_data.get("file_path", ""),
                description=insight_data.get("description", ""),
                patterns=insight_data.get("patterns", []),
                dependencies=insight_data.get("dependencies", []),
            ))

        # Parse web findings
        web_findings = []
        for finding_data in data.get("web_findings", []):
            web_findings.append(WebResearchFinding(
                source=finding_data.get("source", ""),
                summary=finding_data.get("summary", ""),
                relevance=finding_data.get("relevance", ""),
            ))

        return ExplorationResult(
            code_insights=code_insights,
            web_findings=web_findings,
            context_summary=data.get("context_summary", ""),
        )
