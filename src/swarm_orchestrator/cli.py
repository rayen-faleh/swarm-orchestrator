"""
CLI entry point for the Swarm Orchestrator.
"""

import click
from rich.console import Console
from rich.panel import Panel

from . import __version__
from .orchestrator import Orchestrator
from .decomposer import decompose_task


console = Console()


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
    default=600,
    help="Timeout in seconds for agent completion (default: 600)",
)
def run(query: str, agents: int, timeout: int):
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
        )
        result = orchestrator.run(query)

        # Final summary
        console.print("\n" + "━" * 50)
        if result.overall_success:
            console.print("[bold green]✅ All tasks completed successfully![/]")
        else:
            failed = sum(1 for r in result.subtask_results if not r.merged)
            console.print(
                f"[bold yellow]⚠️  {failed} subtask(s) need manual review[/]"
            )

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


if __name__ == "__main__":
    main()
