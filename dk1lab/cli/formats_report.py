"""Render a camera's capture-profile check. Shared by ``find`` and ``config``.

Kept out of both command modules so ``dk1 find cameras`` and ``dk1 config check
--formats`` cannot drift into reporting the same hardware fact differently.
"""

from __future__ import annotations

import typer

from ..config import CaptureProfile
from ..discovery.formats import ProbeError, check_profiles, probe

INDENT = " " * 6


def report_formats(
    device: str, profiles: dict[str, CaptureProfile], *, indent: str = INDENT
) -> bool:
    """Probe ``device`` and print how each configured profile fared.

    A profile the camera does not advertise is a rollout-time failure otherwise:
    OpenCV accepts an unavailable size and silently substitutes the nearest one
    it does support, so the policy would quietly receive a different aspect ratio
    than it was trained on.

    Returns:
        ``True`` if every profile is offered.
    """
    try:
        modes = probe(device)
    except ProbeError as exc:
        typer.secho(f"{indent}could not read capture formats: {exc}", fg=typer.colors.YELLOW)
        return False

    ok = True
    for check in check_profiles(modes, profiles):
        label = f"[capture.{check.profile}] {check.wanted}"
        if check.ok:
            typer.secho(f"{indent}{label}: offered", fg=typer.colors.GREEN)
            continue
        ok = False
        typer.secho(f"{indent}{label}: NOT offered", fg=typer.colors.RED)
        if check.alternatives:
            closest = "; ".join(str(mode) for mode in check.alternatives)
            typer.echo(f"{indent}  closest advertised: {closest}")
        if (gap := check.aspect_gap()) is not None:
            best = check.alternatives[0]
            typer.secho(
                f"{indent}  aspect mismatch: the profile wants {check.aspect:.3f}, the closest "
                f"mode is {best.width / best.height:.3f} (off by {gap:.3f}). MolmoAct2 resizes "
                f"every view to 378x378, so a different aspect ratio stretches the scene "
                f"differently than training did.",
                fg=typer.colors.YELLOW,
            )
    return ok
