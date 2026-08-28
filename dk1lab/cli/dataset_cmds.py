"""``dk1 dataset`` — read a recorded dataset back, and derive the cropped copy.

    dk1 dataset check study/demos             is what was recorded what we meant?
    dk1 dataset crop study/demos study/demos-optimized    the R1 lens, materialised

Neither command opens a ``/dev`` node or energises anything. ``check`` decodes no
video either — it reads ``meta/`` and this fork's ``dk1_notes.jsonl``, so it is
seconds and is safe to run while something else is using the GPU. ``crop`` does
use the video encoder, and says so.

``STUDY.md`` Phase 3 records the demonstrations under ``--profile common`` — the
full lens — so that one day of hands serves both R1 and A1. ``crop`` is the half
of that bargain which has to be kept: R1 rolls out under ``optimized``, so its
training frames have to carry the wrist crop, and :mod:`dk1lab.recrop` applies
the same box the camera would have.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ..config import DEFAULT_CONFIG_PATH, load


app = typer.Typer(no_args_is_help=True, help=__doc__)

ConfigOpt = Annotated[Path, typer.Option("--config", "-c", help="Path to dk1.toml.")]


HELP_CHECK = """Read a recorded dataset's metadata back and say whether it looks right.

Answers the question a recording session leaves open: is what is on disk the
dataset we meant to record? It reports the episodes, the frames, the task string,
the three camera streams and the scene each episode was labelled with, and then
lists everything its own metadata can show to be wrong — a mixed run profile, a
second task string, a camera that stopped, an episode whose metadata a crash ate.

Decodes no video, loads no model, opens no device. Safe to run at any time."""


@app.command("check", help=HELP_CHECK)
def check(
    directory: Annotated[Path, typer.Argument(help="The dataset directory.")],
    holdout: Annotated[
        int,
        typer.Option("--holdout", help="Preview the validation split this many episodes wide."),
    ] = 10,
) -> None:
    from ..dataset import DatasetError, summarise
    from ..finetune import FinetuneError, split_episodes
    from ..recrop import lens_profile, read_lens

    try:
        summary = summarise(directory)
    except DatasetError as exc:
        typer.secho(f"\n{exc}\n", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    typer.secho(f"\n{summary.root}", bold=True)
    typer.echo(
        f"  {summary.repo_id or '(no repo id)'} · {summary.codebase_version} · "
        f"{summary.robot_type or 'unknown robot'} · {summary.fps} Hz"
    )
    typer.echo(
        f"  {summary.episodes} episode(s), {summary.frames} frames "
        f"({summary.seconds / 60:.1f} min), video {summary.video_bytes / 1e9:.2f} GB"
    )
    for task in summary.tasks:
        typer.echo(f"  task: {task!r}")
    for key, (width, height) in summary.cameras.items():
        typer.echo(f"  {key}: {width}x{height}")

    lens = lens_profile(summary.root, summary.notes)
    record = read_lens(summary.root)
    typer.echo(f"  lens: {lens or 'UNKNOWN — neither dk1_lens.json nor the notes say'}")
    if record:
        for key, box in (record.get("streams") or {}).items():
            typer.echo(f"    {key}: {box.get('describe') or box}")

    if summary.by_scene:
        spread = ", ".join(
            f"scene {key}: {count}" if key is not None else f"unlabelled: {count}"
            for key, count in summary.by_scene.items()
        )
        typer.echo(f"  scenes: {spread}")
    for name, values in summary.settings.items():
        if values:
            typer.echo(f"  {name}: {values[0] if len(values) == 1 else values}")

    if summary.lengths:
        shortest, longest = min(summary.lengths), max(summary.lengths)
        mean = summary.frames / len(summary.lengths)
        typer.echo(
            f"  episode length: {shortest}-{longest} frames, mean {mean:.0f} "
            f"({mean / summary.fps:.1f} s)"
            if summary.fps
            else f"  episode length: {shortest}-{longest} frames"
        )

    # The split is shown here rather than only at training time: an unusable
    # hold-out is a fact about the recording, and it is cheaper to find out now
    # than after the arms are packed away.
    if summary.episodes:
        try:
            split = split_episodes(summary.scenes, holdout)
            typer.echo(f"  hold-out: {split.describe()}")
            typer.echo(f"    validation episodes {list(split.holdout)}")
        except FinetuneError as exc:
            typer.secho(f"  hold-out: {exc}", fg=typer.colors.YELLOW)

    if summary.problems:
        typer.secho(f"\n{len(summary.problems)} problem(s):", fg=typer.colors.RED, bold=True)
        for problem in summary.problems:
            typer.secho(f"  - {problem}", fg=typer.colors.RED)
        typer.echo()
        raise typer.Exit(code=1)

    typer.secho("\nreads back clean.\n", fg=typer.colors.GREEN)


HELP_CROP = """Copy a dataset and apply the `optimized` wrist crop to its frames.

R1 is MolmoAct2 on the tuned rig, so its training frames have to carry the wrist
crop this cell rolls out under; the demonstrations are recorded without it, so
that the same bytes also serve A1. This is the derivation between the two, and
it is the same box `dk1lab/crop.py` puts in the camera — read it off
`dk1 config show`, never from prose.

The source is read only. The copy keeps every frame's size, count and timestamp,
so its `meta/` still describes it and the only difference is pixels.

It re-encodes two video streams with the GPU encoder and takes minutes. Nothing
is energised and no arm moves, but do not run it while a policy holds the GPU."""


@app.command("crop", help=HELP_CROP)
def crop(
    source: Annotated[Path, typer.Argument(help="The dataset recorded under --profile common.")],
    destination: Annotated[
        Path | None,
        typer.Argument(help="Where the cropped copy goes. Default: <source>-optimized."),
    ] = None,
    vcodec: Annotated[
        str | None,
        typer.Option(
            "--vcodec",
            help=(
                "Codec for the rewritten streams. Default: the source's own settings, "
                "so the copy differs in pixels and nothing else."
            ),
        ),
    ] = None,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Replace the destination if it exists.")
    ] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print the boxes and write nothing.")
    ] = False,
    config: ConfigOpt = DEFAULT_CONFIG_PATH,
) -> None:
    from ..dataset import DatasetError, summarise
    from ..recrop import RecropError, crop_dataset, frame_size, image_key, plan, read_info

    destination = destination or source.with_name(source.name + "-optimized")
    settings = load(config, require_devices=False)

    try:
        summary = summarise(source)
        info = read_info(source)
        # The box is in pixels, so it is planned at the frames' own size rather
        # than at whatever [capture.*] happens to say today.
        width, height = frame_size(info, image_key("left"))
        crop_plan = plan(settings, width=width, height=height)
    except (DatasetError, RecropError) as exc:
        typer.secho(f"\n{exc}\n", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    typer.secho(f"\n{source} -> {destination}", bold=True)
    typer.echo(f"  {summary.episodes} episode(s), {summary.frames} frames at {width}x{height}")
    if not crop_plan:
        typer.secho(
            f"\n{config} asks for no crop on any camera, so there is nothing to apply.\n",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    for key, line in crop_plan.lines.items():
        typer.echo(f"  {key}: {line}")
    unchanged = [key for key in summary.cameras if key not in crop_plan.boxes]
    typer.echo(f"  copied unchanged: {unchanged}")

    if dry_run:
        typer.secho("\n--dry-run: nothing was written.\n", fg=typer.colors.GREEN)
        return

    typer.secho(
        "\nre-encoding — this uses the video encoder and takes minutes. "
        "Do not run it while a policy holds the GPU.\n",
        fg=typer.colors.YELLOW,
    )
    try:
        report = crop_dataset(
            source,
            destination,
            crop_plan,
            vcodec=vcodec,
            overwrite=overwrite,
            say=typer.echo,
            notes={"capture": {"width": width, "height": height}},
        )
    except RecropError as exc:
        typer.secho(f"\n{exc}\n", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    typer.secho(f"\n{report.describe()}", fg=typer.colors.GREEN)
    typer.echo(f"wrote {report.destination}\n")


__all__ = ["app", "check", "crop"]
