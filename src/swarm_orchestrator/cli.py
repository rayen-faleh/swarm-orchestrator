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


console = Console()


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


def _get_swarm_mcp_config(repo_root: Path) -> dict:
    """
    Generate MCP server config with absolute paths.

    This ensures all agents (in worktrees) can access the same state file
    and use the same Python environment.
    """
    # Use absolute path to state file in main repo
    state_file = str(repo_root / ".swarm" / "state.json")

    # Use uv to run the server from the main repo's environment
    # This ensures the swarm_orchestrator package is available
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


@click.group()
@click.version_option(version=__version__)
def main():
    """🐝 Swarm Orchestrator - Multi-agent consensus using Claude Code"""
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
def run(query: str, agents: int, timeout: int, auto_merge: bool):
    """Run a task through the multi-agent consensus system."""
    console.print(
        Panel.fit(
            f"[bold blue]🐝 Swarm Orchestrator v{__version__}[/]",
            border_style="blue",
        )
    )

    try:
        orchestrator = Orchestrator(
            agent_count=agents,
            timeout=timeout,
            console=console,
            auto_merge=auto_merge,
        )
        result = orchestrator.run(query)

        # Final summary
        console.print("\n" + "━" * 50)
        if result.overall_success:
            if auto_merge:
                all_merged = all(r.merged for r in result.subtask_results)
                if all_merged:
                    console.print("[bold green]✅ Consensus reached and all winners merged![/]")
                else:
                    console.print("[bold yellow]✅ Consensus reached, but some merges failed.[/]")
                    console.print("[dim]Check the output above for details.[/]")
            else:
                console.print("[bold green]✅ Consensus reached! Winner session ready for your review.[/]")
                console.print("[dim]Review the changes above, then merge when satisfied.[/]")
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


@main.command()
@click.option(
    "--force", "-f",
    is_flag=True,
    help="Overwrite existing configuration",
)
def init(force: bool):
    """Initialize swarm orchestrator in the current project.

    Creates .mcp.json configuration for the swarm MCP server
    and sets up the .swarm directory for state persistence.

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

    # Add swarm-orchestrator server config with absolute paths
    if "mcpServers" not in config:
        config["mcpServers"] = {}

    config["mcpServers"]["swarm-orchestrator"] = _get_swarm_mcp_config(repo_root)

    # Write config
    mcp_config_path.write_text(json.dumps(config, indent=2))
    console.print(f"   ✓ Created/updated {mcp_config_path}")

    # Create .swarm directory
    swarm_dir.mkdir(exist_ok=True)
    console.print(f"   ✓ Created {swarm_dir}/")

    # Create .gitignore for .swarm
    gitignore_path = swarm_dir / ".gitignore"
    if not gitignore_path.exists():
        gitignore_path.write_text("state.json\n")
        console.print(f"   ✓ Created {gitignore_path}")

    console.print("\n[bold green]✅ Swarm orchestrator initialized![/]")
    console.print(f"\n[dim]State file:[/] {repo_root / '.swarm' / 'state.json'}")
    console.print("\n[dim]Next steps:[/]")
    console.print("   1. Restart Claude Code to load the new MCP server")
    console.print("   2. Run [cyan]swarm run \"your task\"[/] to start")


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

    mcp_server = SwarmMCPServer(persistence_path=state_file)
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


if __name__ == "__main__":
    main()
