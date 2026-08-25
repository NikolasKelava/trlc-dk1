"""``dk1`` — the one entry point for operating this DK1 cell."""

from __future__ import annotations

import typer

from ..config import ConfigError
from . import config_cmds, find_cmds, policy_cmds, sim_cmds, study_cmds, teleop_cmds

HELP = """Operate the bimanual TRLC-DK1: devices, teleoperation, policies.

Every command that can move the arms says so in its own --help, and warns
again on stderr before it acts. Stopping never moves the arms.
"""

app = typer.Typer(
    name="dk1",
    help=HELP,
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
)

app.add_typer(config_cmds.app, name="config", help="Inspect and check dk1.toml.")
app.add_typer(find_cmds.app, name="find", help="Identify the arm serial ports and the cameras.")
app.command("teleop", help=teleop_cmds.HELP)(teleop_cmds.teleop)
app.add_typer(
    policy_cmds.app,
    name="policy",
    help="Evaluate MolmoAct2: check, smoke, dryrun, run.",
)
app.add_typer(
    sim_cmds.app,
    name="sim",
    help="The MuJoCo cell. Nothing here touches the arms.",
)
app.add_typer(
    study_cmds.app,
    name="study",
    help="The scored comparison: scene photographs and the score sheets.",
)


def main() -> None:
    """Console entry point. Turns config problems into a clean message."""
    try:
        app()
    except ConfigError as exc:
        typer.secho(f"\nconfig error: {exc}\n", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc


__all__ = ["app", "main"]
