"""``dk1 doctor`` — what the machine is doing, and what it was doing when it stopped.

    dk1 doctor watch     sample the machine once a second until Ctrl-C
    dk1 doctor report    read the last telemetry file back, and the boot record

This exists because the machine froze hard six times over 2026-08-25..26 and
every time the kernel journal simply ended — no OOM, no oops, no thermal
message, nothing. That fault was the platform firmware and was fixed by a BIOS
update on 2026-08-27 (`docs/CRASH.md`, closed); these commands stay because they
are what a future freeze would be diagnosed from. Nothing here touches a motor
or a camera.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from ..telemetry import DEFAULT_INTERVAL_S, DEFAULT_TELEMETRY_DIR, Telemetry, read, summary

app = typer.Typer(no_args_is_help=True, help=__doc__)


HELP_WATCH = """Sample the machine once a second into a JSON-lines file, until Ctrl-C.

PSU total power and the +12 V rail, PSU temperature, CPU package temperature,
GPU temperature, power, utilisation, memory and clock, load, free memory and IO
stall. Every line is flushed and fsynced, so the file survives a hard freeze and
its LAST LINE is the state the machine was in.

Run this in a second terminal alongside anything that has frozen before —
including work that is not ours. Nothing here opens a camera or a motor."""


@app.command("watch", help=HELP_WATCH)
def watch(
    label: Annotated[
        str, typer.Option("--label", help="Goes in the filename. Say what you are doing.")
    ] = "watch",
    interval_s: Annotated[
        float, typer.Option("--interval", help="Seconds between samples.")
    ] = DEFAULT_INTERVAL_S,
    directory: Annotated[
        Path, typer.Option("--dir", help="Where the file goes.")
    ] = DEFAULT_TELEMETRY_DIR,
    seconds: Annotated[
        float, typer.Option("--seconds", help="Stop after this long. 0 runs until Ctrl-C.")
    ] = 0.0,
) -> None:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    monitor = Telemetry(directory / f"{stamp}-{label}.jsonl", interval_s=interval_s)
    monitor.start()
    for note in monitor.sources.notes:
        typer.secho(f"  {note}", fg=typer.colors.YELLOW)
    typer.secho(f"sampling every {interval_s:g} s -> {monitor.path}", fg=typer.colors.GREEN)
    typer.echo("Ctrl-C to stop. If the machine freezes, the last line is what it was doing.\n")

    started = time.monotonic()
    try:
        while seconds <= 0 or time.monotonic() - started < seconds:
            time.sleep(min(1.0, interval_s))
            latest = monitor.sources.sample()
            typer.echo(
                "  ".join(
                    f"{key} {latest[key]}"
                    for key in ("clock", "psu_w", "psu_12v_v", "cpu_c", "gpu_c", "gpu_w")
                    if latest.get(key) is not None
                )
            )
    except KeyboardInterrupt:
        typer.echo()
    finally:
        monitor.stop()
    typer.secho(f"{monitor.samples} samples -> {monitor.path}", fg=typer.colors.GREEN)


HELP_REPORT = """Read a telemetry file back: the extremes, and the last line before it ended.

With no argument it takes the newest file in logs/. A file that ends without a
`stop` event ended with the machine — that is the interesting case, and the last
sample is the reading to argue from."""


@app.command("report", help=HELP_REPORT)
def report(
    path: Annotated[
        Path | None, typer.Argument(help="A telemetry .jsonl. Default: the newest in logs/.")
    ] = None,
    directory: Annotated[
        Path, typer.Option("--dir", help="Where to look for the newest file.")
    ] = DEFAULT_TELEMETRY_DIR,
    tail: Annotated[int, typer.Option("--tail", help="How many final samples to print.")] = 5,
) -> None:
    if path is None:
        candidates = sorted(Path(directory).glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            typer.secho(f"no telemetry files in {directory}", fg=typer.colors.YELLOW)
            raise typer.Exit(code=1)
        path = candidates[-1]

    rows = read(path)
    if not rows:
        typer.secho(f"{path} holds no samples", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)

    typer.secho(f"\n{path}", bold=True)
    for line in summary(rows):
        typer.echo(f"  {line}")

    ended_cleanly = any(row.get("event") == "stop" for row in rows)
    typer.secho(
        "\n  ends with a stop event: the process left on its own"
        if ended_cleanly
        else "\n  NO stop event: this file ends where the machine did",
        fg=typer.colors.GREEN if ended_cleanly else typer.colors.RED,
    )

    typer.secho(f"\nlast {tail} sample(s)", bold=True)
    for row in rows[-tail:]:
        typer.echo("  " + "  ".join(f"{k}={v}" for k, v in row.items() if k != "t"))
    typer.echo()


__all__ = ["app", "report", "watch"]
