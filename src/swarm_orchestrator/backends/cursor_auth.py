"""
Cursor authentication helper module.

Provides functions to check authentication status and trigger browser-based login.
"""

import os
import subprocess


def is_authenticated() -> bool:
    """
    Check if cursor-agent is authenticated.

    Returns True if:
    - CURSOR_API_KEY environment variable is set (takes precedence), OR
    - cursor-agent status command exits with code 0
    """
    if os.environ.get("CURSOR_API_KEY"):
        return True

    try:
        result = subprocess.run(
            ["cursor-agent", "status"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def login() -> bool:
    """
    Trigger browser-based cursor-agent login.

    Returns True if login command succeeds, False otherwise.
    """
    try:
        result = subprocess.run(
            ["cursor-agent", "login"],
            timeout=120,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def check_and_prompt_login(auto: bool = False) -> bool:
    """
    Check authentication and prompt for login if needed.

    Args:
        auto: If True, attempt login without prompting user.

    Returns True if authenticated (either already or after login).
    """
    if is_authenticated():
        return True

    if not auto:
        response = input("Cursor not authenticated. Login now? [y/N]: ")
        if response.lower() != "y":
            return False

    return login()
