"""Shared safety notices.

Every command that can cause the arms to move says so, in its ``--help`` and
again on stderr immediately before it does it. Two things are worth stating
plainly every time:

* Connecting a follower is itself motion: it energises all motors and self-zeroes
  both grippers by driving them OPEN against their stop and taking that as zero.
  Verified on the hardware in Phase 3: they do not close on connect, and a
  gripper standing open reports 0.0 — which is what 0 = open means here.
* Connecting a *leader* is motion too, which is easier to forget because the
  leaders are otherwise passive handles: ``DK1Leader.configure`` torques the
  leader gripper servo and drives it to ``gripper_open_pos``. A finger resting in
  a leader trigger gets pushed.
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
    "grippers by driving them open against their stop. Clear the workspace and keep "
    "the e-stop in reach."
)

#: For commands that connect but never command a pose.
ENERGISE_HELP = (
    "\n\n[!] ENERGISES THE ARMS. Connecting energises every motor and self-zeroes "
    "both grippers by driving them open against their stop. No pose is ever "
    "commanded, but the arms are live and holding position throughout."
)


#: Appended to the ``--help`` of any command that connects the leader arms.
LEADER_HELP = (
    "\n\n[!] ALSO MOVES THE LEADERS. Connecting a leader torques its gripper servo "
    "and drives it open. Keep fingers out of the leader triggers."
)


def confirm_motion(what: str, *, assume_yes: bool = False, notes: list[str] | None = None) -> None:
    """Print the pre-connect warning and require an explicit go-ahead.

    Args:
        what: short description of what is about to happen.
        assume_yes: skip the prompt (``--yes``). The warning is still printed.
        notes: extra lines shown inside the banner, for motion specific to one
            command — the leader grippers opening during teleoperation, say.
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
        "  BOTH grippers by driving them OPEN against their stop.",
        fg=typer.colors.YELLOW,
        err=True,
    )
    for note in notes or []:
        typer.secho(f"  {note}", fg=typer.colors.YELLOW, err=True)
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
