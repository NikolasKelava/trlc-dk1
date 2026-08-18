"""Shared safety notices.

Every command that can cause the arms to move says so, in its ``--help`` and
again on stderr immediately before it does it. Two things are worth stating
plainly every time:

* Connecting a follower is itself motion: it energises all motors and self-zeroes
  both grippers by closing them until they stall.
* Stopping never moves the arms. Return-to-home is opt-in, because sweeping the
  arms home is the last thing you want when you stopped because something is
  wrong.
"""

from __future__ import annotations

import sys

import typer

#: Appended to the ``--help`` of any command that can move the arms.
MOTION_HELP = (
    "\n\n[!] CAUSES MOTION. Connecting energises every motor and self-zeroes both "
    "grippers by closing them until they stall. Clear the workspace and keep the "
    "e-stop in reach."
)

#: For commands that connect but never command a pose.
ENERGISE_HELP = (
    "\n\n[!] ENERGISES THE ARMS. Connecting energises every motor and self-zeroes "
    "both grippers by closing them until they stall. No pose is ever commanded, but "
    "the arms are live and holding position throughout."
)


def confirm_motion(what: str, *, assume_yes: bool = False) -> None:
    """Print the pre-connect warning and require an explicit go-ahead.

    Args:
        what: short description of what is about to happen.
        assume_yes: skip the prompt (``--yes``). The warning is still printed.
    """
    typer.secho("", err=True)
    typer.secho("  " + "!" * 68, fg=typer.colors.YELLOW, err=True)
    typer.secho(f"  ABOUT TO: {what}", fg=typer.colors.YELLOW, bold=True, err=True)
    typer.secho("", err=True)
    typer.secho(
        "  Connecting the follower energises every arm motor and self-zeroes",
        fg=typer.colors.YELLOW,
        err=True,
    )
    typer.secho(
        "  BOTH grippers by driving them closed until they stall.",
        fg=typer.colors.YELLOW,
        err=True,
    )
    typer.secho(
        "  Clear the workspace, keep hands clear of the grippers, e-stop in reach.",
        fg=typer.colors.YELLOW,
        err=True,
    )
    typer.secho("  " + "!" * 68, fg=typer.colors.YELLOW, err=True)
    typer.secho("", err=True)

    if assume_yes:
        typer.secho("  (--yes given, continuing)", err=True)
        return
    if not sys.stdin.isatty():
        raise typer.BadParameter(
            "Refusing to energise the arms from a non-interactive session without --yes."
        )
    typer.confirm("  Workspace clear — continue?", abort=True, err=True)
