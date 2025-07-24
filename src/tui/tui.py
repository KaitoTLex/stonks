# tui_dashboard.py

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich import box


def render_dual_agent_dashboard(history_dqn, history_mc):
    console = Console()

    steps = list(range(len(history_dqn["portfolio_value"])))
    dqn_values = history_dqn["portfolio_value"]
    mc_values = history_mc["portfolio_value"]

    table = Table(title="Agent Comparison (DQN vs Monte Carlo)", box=box.ROUNDED)
    table.add_column("Step", justify="right", style="cyan")
    table.add_column("DQN Portfolio", justify="right", style="green")
    table.add_column("MC Portfolio", justify="right", style="magenta")

    for i, (dqn_val, mc_val) in enumerate(zip(dqn_values, mc_values)):
        table.add_row(str(i), f"${dqn_val:.2f}", f"${mc_val:.2f}")

    console.rule("[bold yellow]Per-Step Portfolio Values")
    console.print(table)

    # Summary panels
    final_dqn = dqn_values[-1]
    final_mc = mc_values[-1]
    initial = dqn_values[0]  # Assumes both started with same balance

    console.rule("[bold cyan]📊 Final Portfolio Comparison")

    panel_dqn = Panel.fit(
        f"[bold green]DQN Final Portfolio:\n${final_dqn:.2f}",
        title="DQN Agent",
        border_style="green",
    )
    panel_mc = Panel.fit(
        f"[bold magenta]MC Final Portfolio:\n${final_mc:.2f}",
        title="Monte Carlo Agent",
        border_style="magenta",
    )

    console.print(panel_dqn, panel_mc, justify="center")

    # Summary progress bars
    console.rule("[bold blue]📈 Overall Growth")
    progress = Progress(
        TextColumn("{task.description}"),
        BarColumn(bar_width=30),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        expand=True,
    )
    with progress:
        progress.add_task("DQN Growth", total=initial, completed=final_dqn)
        progress.add_task("MC Growth ", total=initial, completed=final_mc)
