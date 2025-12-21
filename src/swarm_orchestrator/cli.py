"""
CLI entry point for the Swarm Orchestrator.
"""

import json
import subprocess
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel

from . import __version__
from .orchestrator import Orchestrator
from .decomposer import decompose_task
from .installation import detect_installation_context
from .config import SwarmConfig, load_config, save_config, get_backend_choices, BACKENDS
from .backends import cursor_auth
from .backends.base import WorktreeBackend


console = Console()


def _get_worktree_backend(config: SwarmConfig | None = None) -> WorktreeBackend:
    """Create and return the appropriate worktree backend based on config."""
    if config is None:
        config = load_config()

    if config.worktree_backend == "schaltwerk":
        from .backends.schaltwerk import SchaltwerkWorktreeBackend
        return SchaltwerkWorktreeBackend()
    elif config.worktree_backend == "git-native":
        from .backends.git_native import GitNativeWorktreeBackend
        return GitNativeWorktreeBackend()
    else:
        raise ValueError(f"Unknown worktree backend: {config.worktree_backend}")


def _get_agent_backend(config: SwarmConfig | None = None):
    """Create and return the appropriate agent backend based on config."""
    if config is None:
        config = load_config()

    if config.agent_backend == "git-native":
        from .backends.git_native import GitNativeAgentBackend
        return GitNativeAgentBackend()
    else:
        return None


def _get_git_root() -> Path:
    """Get the root directory of the git repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        )
        return Path(result.stdout.strip())
    except subprocess.CalledProcessError:
        # Not a git repo, use current directory
        return Path.cwd()


def _update_global_claude_config(mcp_server_config: dict) -> None:
    """
    Update the global ~/.claude.json with swarm-orchestrator MCP config.

    This ensures agents in worktrees can access the MCP server, since
    worktrees don't inherit project-level .mcp.json.

    Args:
        mcp_server_config: The MCP server config dict to set for swarm-orchestrator
    """
    config_path = Path.home() / ".claude.json"

    # Load existing config or start fresh
    config = {}
    if config_path.exists():
        try:
            content = config_path.read_text()
            loaded = json.loads(content)
            if isinstance(loaded, dict):
                config = loaded
        except json.JSONDecodeError:
            pass  # Start with empty config if malformed

    # Ensure mcpServers exists
    if "mcpServers" not in config:
        config["mcpServers"] = {}

    # Add/update swarm-orchestrator entry
    config["mcpServers"]["swarm-orchestrator"] = mcp_server_config

    # Write back
    config_path.write_text(json.dumps(config, indent=2))


def _get_swarm_mcp_config(repo_root: Path) -> dict:
    """
    Generate MCP server config with absolute paths.

    This ensures all agents (in worktrees) can access the same state file
    and use the same Python environment.

    The config format depends on installation type:
    - Local install in Python project: Uses uv run --project for venv support
    - Global install (pipx/system) or non-Python project: Uses direct swarm command
    """
    # Use absolute path to state file in main repo
    state_file = str(repo_root / ".swarm" / "state.json")

    # Detect installation context
    ctx = detect_installation_context(repo_root)

    # Use uv-based approach only for local installs in Python projects
    if ctx.is_local_install and ctx.in_python_project:
        return {
            "command": "uv",
            "args": [
                "run",
                "--project",
                str(repo_root),
                "python",
                "-m",
                "swarm_orchestrator.swarm_mcp.server",
                "--state-file",
                state_file,
            ],
        }

    # Global install or non-Python project: use direct swarm command
    return {
        "command": "swarm",
        "args": [
            "server",
            "--state-file",
            state_file,
        ],
    }


@click.group()
@click.version_option(version=__version__)
def main():
    """🐝 Swarm Orchestrator - Multi-agent consensus using Claude Code

    Run multiple AI agents in parallel on the same task and use voting
    to select the best solution. Agents work in isolated git worktrees.

    \b
    Configuration:
      Config file: .swarm/config.json (created by 'swarm init')
      CLI flags override config file settings.

    \b
    Backends:
      --worktree-backend  Git worktree isolation (schaltwerk)
      --agent-backend     Agent execution (schaltwerk)
      --cli-tool          CLI tool for decomposition and agents (claude, opencode, cursor, anthropic-api)

    \b
    Quick start:
      swarm init           Initialize config in current project
      swarm run "task"     Run agents on a task
      swarm status         Show running sessions
    """
    pass


@main.command()
@click.argument("query")
@click.option(
    "--agents", "-a",
    default=3,
    help="Number of agents to spawn per subtask (default: 3)",
)
@click.option(
    "--timeout", "-t",
    default=120,
    help="Timeout in seconds for decomposition step (default: 120). Agents have no timeout.",
)
@click.option(
    "--auto-merge",
    is_flag=True,
    help="Automatically merge the winning solution after consensus is reached.",
)
@click.option(
    "--skip-exploration",
    is_flag=True,
    help="Skip codebase exploration phase (useful for simple tasks).",
)
@click.option(
    "--config", "-c",
    type=click.Path(exists=True),
    help="Path to config file. Default: .swarm/config.json",
)
@click.option(
    "--worktree-backend",
    type=click.Choice(get_backend_choices("worktree")),
    help="Worktree isolation backend. schaltwerk: Schaltwerk MCP for git worktrees (default).",
)
@click.option(
    "--agent-backend",
    type=click.Choice(get_backend_choices("agent")),
    help="Agent execution backend. schaltwerk: Schaltwerk MCP to spawn Claude agents (default).",
)
@click.option(
    "--cli-tool",
    type=click.Choice(get_backend_choices("cli_tool")),
    help="CLI tool for decomposition and agent execution. claude (default), opencode, cursor, or anthropic-api.",
)
@click.option(
    "--llm-model",
    default=None,
    help="Model for anthropic-api backend (default: claude-sonnet-4-20250514). Ignored with other cli_tools.",
)
@click.option(
    "--exploration-model",
    default=None,
    help="Model for exploration phase (default: claude-haiku-3-5). Uses a smaller model for cost efficiency.",
)
def run(
    query: str,
    agents: int,
    timeout: int,
    auto_merge: bool,
    skip_exploration: bool,
    config: str | None,
    worktree_backend: str | None,
    agent_backend: str | None,
    cli_tool: str | None,
    llm_model: str | None,
    exploration_model: str | None,
):
    """Run a task through the multi-agent consensus system.

    \b
    Backend Configuration:
      Backends can be set via config file (.swarm/config.json) or CLI flags.
      CLI flags override config file settings.

    \b
    CLI Tool Selection:
      claude (default): Uses Claude Code CLI for both decomposition and agents.
      opencode: Uses OpenCode CLI for agents, Claude CLI for decomposition.
      cursor: Uses Cursor CLI for agents, Claude CLI for decomposition.
      anthropic-api: Uses Anthropic API for decomposition (requires ANTHROPIC_API_KEY).
    """
    console.print(
        Panel.fit(
            f"[bold blue]🐝 Swarm Orchestrator v{__version__}[/]",
            border_style="blue",
        )
    )

    try:
        # Load config from file or use defaults
        swarm_config = load_config(config)

        # Override with CLI flags if provided
        if worktree_backend:
            swarm_config.worktree_backend = worktree_backend
        if agent_backend:
            swarm_config.agent_backend = agent_backend
        if cli_tool:
            swarm_config.cli_tool = cli_tool
        if llm_model:
            swarm_config.llm_model = llm_model
        if exploration_model:
            swarm_config.exploration_model = exploration_model

        orchestrator = Orchestrator(
            agent_count=agents,
            timeout=timeout,
            console=console,
            auto_merge=auto_merge,
            skip_exploration=skip_exploration,
            config=swarm_config,
        )
        result = orchestrator.run(query)

        # Final summary
        console.print("\n" + "━" * 50)
        if result.overall_success:
            all_merged = all(r.merged for r in result.subtask_results)
            if auto_merge:
                if all_merged:
                    console.print("[bold green]✅ Consensus reached and all winners merged![/]")
                else:
                    console.print("[bold yellow]✅ Consensus reached, but some merges failed.[/]")
                    console.print("[dim]Check the output above for details.[/]")
            else:
                # Dashboard was shown for review - provide appropriate message
                if all_merged:
                    console.print("[bold green]✅ All tasks complete and merged![/]")
                else:
                    console.print("[bold green]✅ All tasks complete.[/]")
                    console.print("[dim]Use 'swarm watch' to review and merge remaining sessions.[/]")
        else:
            no_consensus = sum(1 for r in result.subtask_results if not r.vote_result.consensus_reached)
            console.print(
                f"[bold yellow]⚠️  {no_consensus} subtask(s) had no clear consensus[/]"
            )
            console.print("[dim]All sessions kept for manual review.[/]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/]")
        raise SystemExit(1)
    except Exception as e:
        console.print(f"\n[bold red]Error:[/] {e}")
        raise SystemExit(1)


@main.command()
@click.argument("query")
def decompose(query: str):
    """Decompose a task into subtasks (dry run, no agents spawned)."""
    console.print(f"\n[bold]🔍 Decomposing:[/] {query}\n")

    try:
        result = decompose_task(query)

        if result.is_atomic:
            console.print("[cyan]Type:[/] Atomic (single task)")
        else:
            console.print(f"[cyan]Type:[/] Complex ({len(result.subtasks)} subtasks)")

        console.print("\n[bold]Subtasks:[/]")
        for st in result.subtasks:
            console.print(f"\n  [bold cyan]{st.id}[/]")
            console.print(f"  [dim]Description:[/] {st.description}")
            console.print(f"  [dim]Prompt:[/]")
            for line in st.prompt.split("\n")[:5]:  # First 5 lines
                console.print(f"    {line}")
            if st.prompt.count("\n") > 5:
                console.print("    ...")

    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")
        raise SystemExit(1)


@main.command()
def status():
    """Show status of running swarm sessions."""
    from .schaltwerk import get_client

    console.print("\n[bold]📊 Swarm Status[/]\n")

    try:
        client = get_client()
        sessions = client.get_session_list()

        if not sessions:
            console.print("[dim]No active swarm sessions[/]")
            return

        for session in sessions:
            if session.name.startswith("wave-"):
                continue  # Skip non-swarm sessions

            status_color = {
                "running": "yellow",
                "completed": "green",
                "reviewed": "cyan",
                "spec": "dim",
            }.get(session.session_state, "white")

            console.print(
                f"  [{status_color}]●[/] {session.name} "
                f"[dim]({session.session_state})[/]"
            )

    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")
        raise SystemExit(1)


def _prompt_backend_selection(
    backend_type: str,
    title: str,
    is_interactive: bool,
) -> str:
    """Prompt user to select a backend option.

    Args:
        backend_type: Type of backend (worktree, agent, llm)
        title: Display title for the prompt
        is_interactive: Whether to prompt or use defaults

    Returns:
        Selected backend name
    """
    options = BACKENDS.get(backend_type, {})
    option_list = list(options.keys())

    if not is_interactive or not option_list:
        return option_list[0] if option_list else ""

    console.print(f"\n[bold]{title}[/]")
    for i, (name, desc) in enumerate(options.items(), 1):
        console.print(f"  [cyan]{i}[/]) {name}: [dim]{desc}[/]")

    while True:
        choice = click.prompt(
            "Select option",
            type=int,
            default=1,
        )
        if 1 <= choice <= len(option_list):
            return option_list[choice - 1]
        console.print(f"[red]Invalid choice. Please enter 1-{len(option_list)}[/]")


@main.command()
@click.option(
    "--force", "-f",
    is_flag=True,
    help="Overwrite existing configuration",
)
@click.option(
    "--non-interactive",
    is_flag=True,
    help="Use default values without prompting (for scripts)",
)
def init(force: bool, non_interactive: bool):
    """Initialize swarm orchestrator in the current project.

    \b
    Interactively prompts for backend selections:
      - Worktree backend (git isolation)
      - Agent backend (agent execution)
      - CLI Tool (decomposition and agent execution)

    \b
    Creates:
      .mcp.json           MCP server configuration for Claude Code
      .swarm/             Directory for state persistence
      .swarm/config.json  Backend configuration with your selections

    Uses absolute paths so agents in worktrees can share state.
    """
    console.print("\n[bold]🐝 Initializing Swarm Orchestrator[/]\n")

    # Get repo root for absolute paths
    repo_root = _get_git_root()
    console.print(f"   Repository root: {repo_root}")

    mcp_config_path = Path(".mcp.json")
    swarm_dir = Path(".swarm")

    # Load or create MCP config
    if mcp_config_path.exists():
        config = json.loads(mcp_config_path.read_text())
    else:
        config = {"mcpServers": {}}

    # Check if already configured
    if "swarm-orchestrator" in config.get("mcpServers", {}) and not force:
        console.print("[yellow]⚠️  Swarm orchestrator already configured[/]")
        console.print("   Use --force to overwrite existing configuration")
        return

    # Determine if we should prompt interactively
    is_interactive = not non_interactive

    # Interactive backend selection
    worktree_backend = _prompt_backend_selection(
        "worktree", "Worktree Backend (git isolation)", is_interactive
    )
    agent_backend = _prompt_backend_selection(
        "agent", "Agent Backend (agent execution)", is_interactive
    )
    cli_tool = _prompt_backend_selection(
        "cli_tool", "CLI Tool (decomposition and agent execution)", is_interactive
    )

    # Optional model selection for anthropic-api
    llm_model = "claude-sonnet-4-20250514"
    if cli_tool == "anthropic-api" and is_interactive:
        console.print("\n[bold]Model Selection[/]")
        console.print("  [cyan]1[/]) claude-sonnet-4-20250514 [dim](default, fast)[/]")
        console.print("  [cyan]2[/]) claude-opus-4-20250514 [dim](most capable)[/]")
        model_choice = click.prompt("Select model", type=int, default=1)
        if model_choice == 2:
            llm_model = "claude-opus-4-20250514"

    # Add swarm-orchestrator server config with absolute paths
    if "mcpServers" not in config:
        config["mcpServers"] = {}

    config["mcpServers"]["swarm-orchestrator"] = _get_swarm_mcp_config(repo_root)

    # Write MCP config (project-level)
    mcp_config_path.write_text(json.dumps(config, indent=2))
    console.print(f"\n   ✓ Created/updated {mcp_config_path}")

    # Update global ~/.claude.json so agents in worktrees have MCP access
    mcp_server_config = _get_swarm_mcp_config(repo_root)
    _update_global_claude_config(mcp_server_config)
    console.print(f"   ✓ Updated ~/.claude.json (global MCP config)")

    # Create .swarm directory
    swarm_dir.mkdir(exist_ok=True)
    console.print(f"   ✓ Created {swarm_dir}/")

    # Create .gitignore for .swarm
    gitignore_path = swarm_dir / ".gitignore"
    if not gitignore_path.exists():
        gitignore_path.write_text("state.json\n")
        console.print(f"   ✓ Created {gitignore_path}")

    # Create swarm config with selected backends
    swarm_config = SwarmConfig(
        worktree_backend=worktree_backend,
        agent_backend=agent_backend,
        cli_tool=cli_tool,
        llm_model=llm_model,
    )
    save_config(swarm_config)
    console.print(f"   ✓ Created {swarm_dir}/config.json")

    console.print("\n[bold green]✅ Swarm orchestrator initialized![/]")
    console.print(f"\n[dim]Configuration:[/]")
    console.print(f"   Worktree: {worktree_backend}")
    console.print(f"   Agent: {agent_backend}")
    console.print(f"   CLI Tool: {cli_tool}")
    if cli_tool == "anthropic-api":
        console.print(f"   Model: {llm_model}")
    console.print(f"\n[dim]State file:[/] {repo_root / '.swarm' / 'state.json'}")
    console.print("\n[dim]Next steps:[/]")
    console.print("   1. Restart Claude Code to load the new MCP server")
    console.print("   2. Run [cyan]swarm run \"your task\"[/] to start")


@main.group()
def config():
    """View and modify swarm configuration.

    \b
    Commands:
      show    Display current configuration
      set     Update a configuration value
    """
    pass


# Map CLI keys (with hyphens) to config keys (with underscores)
CONFIG_KEYS = {
    "worktree-backend": ("worktree_backend", "worktree"),
    "agent-backend": ("agent_backend", "agent"),
    "cli-tool": ("cli_tool", "cli_tool"),
    "llm-model": ("llm_model", None),
    "llm-timeout": ("llm_timeout", None),
    "exploration-model": ("exploration_model", None),
}


def _config_exists() -> bool:
    """Check if swarm config exists."""
    return Path(".swarm/config.json").exists()


@config.command("show")
def config_show():
    """Display current configuration in a formatted table."""
    if not _config_exists():
        console.print("[bold red]Error:[/] Swarm not initialized.")
        console.print("Run [cyan]swarm init[/] first.")
        raise SystemExit(1)

    swarm_config = load_config()
    config_dict = swarm_config.to_dict()

    console.print("\n[bold]🐝 Swarm Configuration[/]\n")
    console.print(f"   [dim]Config file:[/] .swarm/config.json\n")

    from rich.table import Table
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Setting", style="white")
    table.add_column("Current Value", style="green")
    table.add_column("Valid Options", style="dim")

    for cli_key, (config_key, backend_type) in CONFIG_KEYS.items():
        current_value = config_dict.get(config_key, "")
        if backend_type:
            valid_options = ", ".join(get_backend_choices(backend_type))
        else:
            valid_options = "(any)" if config_key == "llm_model" else "(integer)"
        table.add_row(cli_key, str(current_value), valid_options)

    console.print(table)


@config.command("set")
@click.argument("key")
@click.argument("value")
def config_set(key: str, value: str):
    """Set a configuration value.

    \b
    KEY is one of: worktree-backend, agent-backend, cli-tool, llm-model, llm-timeout
    VALUE must be valid for the given key.

    \b
    Examples:
      swarm config set cli-tool anthropic-api
      swarm config set llm-model claude-opus-4-20250514
    """
    if not _config_exists():
        console.print("[bold red]Error:[/] Swarm not initialized.")
        console.print("Run [cyan]swarm init[/] first.")
        raise SystemExit(1)

    if key not in CONFIG_KEYS:
        valid_keys = ", ".join(CONFIG_KEYS.keys())
        console.print(f"[bold red]Error:[/] Unknown config key '{key}'")
        console.print(f"Valid keys: {valid_keys}")
        raise SystemExit(1)

    config_key, backend_type = CONFIG_KEYS[key]

    # Validate value for backend keys
    if backend_type:
        valid_choices = get_backend_choices(backend_type)
        if value not in valid_choices:
            console.print(f"[bold red]Error:[/] Invalid value '{value}' for {key}")
            console.print(f"Valid options: {', '.join(valid_choices)}")
            raise SystemExit(1)

    # Handle llm-timeout as integer
    if config_key == "llm_timeout":
        try:
            value = int(value)
        except ValueError:
            console.print(f"[bold red]Error:[/] {key} must be an integer")
            raise SystemExit(1)

    # Load, update, and save config
    swarm_config = load_config()
    setattr(swarm_config, config_key, value)
    save_config(swarm_config)

    console.print(f"[green]✓[/] Set {key} = {value}")


@main.command()
@click.option(
    "--state-file",
    default=".swarm/state.json",
    help="Path to state persistence file",
)
def server(state_file: str):
    """Run the swarm MCP server (for internal use).

    This command is called by Claude Code via the MCP configuration.
    You typically don't need to run this manually.
    """
    from .swarm_mcp.server import SwarmMCPServer

    config = load_config()
    mcp_server = SwarmMCPServer(
        persistence_path=state_file,
        compression_config=config,
    )
    mcp_server.run_stdio()


def _check_initialized() -> bool:
    """Check if swarm is initialized in current directory."""
    mcp_config_path = Path(".mcp.json")

    if not mcp_config_path.exists():
        return False

    try:
        config = json.loads(mcp_config_path.read_text())
        return "swarm-orchestrator" in config.get("mcpServers", {})
    except (json.JSONDecodeError, KeyError):
        return False


@main.group()
def cursor():
    """Cursor authentication commands.

    \b
    Manage Cursor CLI agent authentication for the cursor-cli backend.
    Use 'login' for interactive browser-based authentication or
    set CURSOR_API_KEY environment variable for automated workflows.
    """
    pass


@cursor.command()
def login():
    """Authenticate with Cursor via browser-based login.

    \b
    Opens your browser to authenticate with Cursor. This is the
    recommended method for interactive use. For CI/CD or automated
    workflows, use CURSOR_API_KEY environment variable instead.
    """
    console.print("\n[bold]🔑 Cursor Login[/]\n")
    console.print("Opening browser for authentication...")

    if cursor_auth.login():
        console.print("\n[bold green]✅ Successfully logged in to Cursor![/]")
    else:
        console.print("\n[bold red]❌ Login failed.[/]")
        console.print("[dim]Make sure cursor-agent is installed and try again.[/]")
        raise SystemExit(1)


@cursor.command()
def status():
    """Check Cursor authentication status.

    \b
    Shows whether you are currently authenticated with Cursor.
    Authentication can be via browser login or CURSOR_API_KEY env var.
    """
    console.print("\n[bold]🔑 Cursor Auth Status[/]\n")

    if cursor_auth.is_authenticated():
        console.print("[green]✓[/] Authenticated")
    else:
        console.print("[yellow]✗[/] Not authenticated")
        console.print("\n[dim]To authenticate, run:[/]")
        console.print("  swarm cursor login")
        console.print("\n[dim]Or set CURSOR_API_KEY environment variable.[/]")


@main.command()
@click.option(
    "--active",
    is_flag=True,
    help="Show only active (running) sessions",
)
@click.option(
    "--reviewed",
    is_flag=True,
    help="Show only reviewed sessions ready to merge",
)
def sessions(active: bool, reviewed: bool):
    """List all swarm sessions with their status.

    \b
    Shows sessions created by swarm for agent work, including:
    - Session name
    - Current status (running, reviewed, etc.)
    - Git branch name
    """
    console.print("\n[bold]📋 Swarm Sessions[/]\n")

    try:
        backend = _get_worktree_backend()

        filter_type = "all"
        if active:
            filter_type = "active"
        elif reviewed:
            filter_type = "reviewed"

        sessions_list = backend.list_sessions(filter_type)

        if not sessions_list:
            console.print("[dim]No sessions found[/]")
            return

        from rich.table import Table
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Name", style="white")
        table.add_column("Status", style="yellow")
        table.add_column("Branch", style="dim")

        for session in sessions_list:
            status_style = {
                "running": "yellow",
                "reviewed": "green",
                "spec": "dim",
            }.get(session.status, "white")
            table.add_row(
                session.name,
                f"[{status_style}]{session.status}[/]",
                session.branch,
            )

        console.print(table)

    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")
        raise SystemExit(1)


@main.command()
@click.argument("session_name")
def diff(session_name: str):
    """Show the diff for a session.

    \b
    Displays the changes made in a session compared to its parent branch.
    Shows files changed and the full unified diff output.
    """
    console.print(f"\n[bold]📝 Diff for session:[/] {session_name}\n")

    try:
        backend = _get_worktree_backend()

        session = backend.get_session(session_name)
        if session is None:
            console.print(f"[bold red]Error:[/] Session '{session_name}' not found")
            raise SystemExit(1)

        diff_result = backend.get_diff(session_name)

        if not diff_result.files:
            console.print("[dim]No changes in this session[/]")
            return

        console.print(f"[cyan]Files changed:[/] {len(diff_result.files)}")
        for f in diff_result.files:
            console.print(f"  • {f}")

        console.print("\n[bold]Diff:[/]")
        console.print(diff_result.content)

    except SystemExit:
        raise
    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")
        raise SystemExit(1)


@main.command()
@click.argument("session_name")
@click.option(
    "-m", "--message",
    default=None,
    help="Commit message for the merge (default: auto-generated)",
)
def merge(session_name: str, message: str | None):
    """Merge a session's changes to the parent branch.

    \b
    Squash-merges the session's changes back to the parent branch
    and cleans up the session worktree and branch.
    """
    console.print(f"\n[bold]🔀 Merging session:[/] {session_name}\n")

    try:
        backend = _get_worktree_backend()

        session = backend.get_session(session_name)
        if session is None:
            console.print(f"[bold red]Error:[/] Session '{session_name}' not found")
            raise SystemExit(1)

        commit_msg = message or f"Merge session: {session_name}"
        backend.merge_session(session_name, commit_msg)

        console.print(f"[bold green]✅ Session '{session_name}' merged successfully![/]")

    except SystemExit:
        raise
    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")
        raise SystemExit(1)


@main.command()
@click.argument("session_name")
def stop(session_name: str):
    """Stop a running agent session.

    \b
    Terminates the agent process for the specified session.
    The session worktree and metadata are preserved.
    """
    console.print(f"\n[bold]🛑 Stopping session:[/] {session_name}\n")

    try:
        agent_backend = _get_agent_backend()
        if agent_backend is None:
            console.print("[bold red]Error:[/] Agent backend does not support stopping")
            raise SystemExit(1)

        if not hasattr(agent_backend, "stop_agent"):
            console.print("[bold red]Error:[/] Agent backend does not support stopping")
            raise SystemExit(1)

        stopped = agent_backend.stop_agent(session_name)

        if stopped:
            console.print(f"[bold green]✅ Session '{session_name}' stopped![/]")
        else:
            console.print(f"[yellow]Session '{session_name}' was not running or not found[/]")

    except SystemExit:
        raise
    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")
        raise SystemExit(1)


@main.command()
@click.argument("session_name")
@click.option(
    "--force", "-f",
    is_flag=True,
    help="Force deletion even if uncommitted changes exist",
)
def delete(session_name: str, force: bool):
    """Delete a session and clean up resources.

    \b
    Removes the session's worktree, branch, and metadata.
    If an agent is running, it will be stopped first.
    """
    console.print(f"\n[bold]🗑️  Deleting session:[/] {session_name}\n")

    try:
        # Stop any running agent first
        agent_backend = _get_agent_backend()
        if agent_backend and hasattr(agent_backend, "stop_agent"):
            agent_backend.stop_agent(session_name)

        backend = _get_worktree_backend()

        session = backend.get_session(session_name)
        if session is None:
            console.print(f"[bold red]Error:[/] Session '{session_name}' not found")
            raise SystemExit(1)

        backend.delete_session(session_name, force=force)

        console.print(f"[bold green]✅ Session '{session_name}' deleted![/]")

    except SystemExit:
        raise
    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")
        raise SystemExit(1)


@main.command()
def watch():
    """Launch interactive TUI dashboard for session management.

    \b
    Opens a live-updating terminal UI that shows all sessions
    and allows keyboard-driven navigation and actions.

    \b
    Keyboard controls:
      j/↓    Move selection down
      k/↑    Move selection up
      d      Toggle diff preview for selected session
      m      Merge selected session
      s      Stop selected session (agent process)
      x      Delete selected session (worktree + branch)
      r      Refresh session list
      q      Quit dashboard
    """
    from .tui import SessionsDashboard

    try:
        backend = _get_worktree_backend()
        agent_backend = _get_agent_backend()
        dashboard = SessionsDashboard(backend=backend, agent_backend=agent_backend)
        dashboard.run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
