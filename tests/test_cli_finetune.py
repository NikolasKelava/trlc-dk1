"""`dk1 dataset` and `dk1 policy finetune` at the command-line boundary.

The arithmetic is covered by ``test_finetune.py`` and ``test_recrop.py``. What is
new here is the gate: that a fine-tune **refuses to start** when the dataset's
lens does not match the row's profile, or when the dataset does not read back
clean — because both of those produce a checkpoint that looks exactly like the
one that was asked for and was trained for a camera or a cell that does not
exist.

Nothing here trains anything: every path goes through ``--dry-run``, which writes
the run directory and returns. No GPU, no robot, no LeRobot.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from dk1lab import finetune
from dk1lab.cli.main import app
from dk1lab.layout import CAMERA_NAMES


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def demos(tmp_path):
    """A 45-episode dataset's metadata, recorded under `common`. No video files.

    ``dk1 dataset check`` and ``dk1 policy finetune`` both read metadata only, so
    a fixture with no mp4 in it exercises everything they do — apart from the
    one check that ``videos/`` is not empty, which is set up below.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = tmp_path / "demos"
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "repo_id": "dk1/demos",
                "robot_type": "bi_dk1_follower_safe",
                "fps": 30,
                "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
                "features": {
                    f"observation.images.{name}": {
                        "dtype": "video",
                        "shape": [720, 1280, 3],
                        "names": ["height", "width", "channels"],
                    }
                    for name in CAMERA_NAMES
                },
            }
        )
    )
    pq.write_table(
        pa.table(
            {
                "episode_index": list(range(45)),
                "length": [600] * 45,
                "tasks": [["put the dice in the bowl"]] * 45,
            }
        ),
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )
    (root / "dk1_notes.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "episode": index,
                    "scene": 1 + index // 15,
                    "profile": "common",
                    "capture": "policy",
                    "fps": 30,
                    "vcodec": "h264_nvenc",
                }
            )
            for index in range(45)
        )
        + "\n"
    )
    # Enough for "videos/ is not empty"; nothing here decodes one.
    stream = root / "videos" / "observation.images.top" / "chunk-000"
    stream.mkdir(parents=True)
    (stream / "file-000.mp4").write_bytes(b"\x00" * 16)
    return root


def cropped(demos):
    """Mark a dataset as carrying the optimized lens, the way `dk1 dataset crop` does."""
    (demos / "dk1_lens.json").write_text(json.dumps({"profile": "optimized", "streams": {}}))
    return demos


# --------------------------------------------------------------------------- #
# dk1 dataset check
# --------------------------------------------------------------------------- #


def test_a_clean_dataset_reads_back_clean(runner, demos):
    result = runner.invoke(app, ["dataset", "check", str(demos)])
    assert result.exit_code == 0, result.output
    assert "45 episode(s), 27000 frames" in result.output
    assert "put the dice in the bowl" in result.output
    assert "scene 1: 15, scene 2: 15, scene 3: 15" in result.output
    assert "lens: common" in result.output


def test_the_check_previews_the_validation_split(runner, demos):
    """An unusable hold-out is a fact about the recording, and it is cheaper to
    find out while the cell is still set up than after it is packed away."""
    result = runner.invoke(app, ["dataset", "check", str(demos)])
    assert "41 train, 4 held out" in result.output
    assert "scene 1" in result.output and "scene 3" in result.output


def test_a_dataset_with_problems_exits_non_zero(runner, demos):
    (demos / "dk1_notes.jsonl").unlink()
    result = runner.invoke(app, ["dataset", "check", str(demos)])
    assert result.exit_code == 1
    assert "no dk1_notes.jsonl" in result.output


def test_a_directory_that_is_not_a_dataset_says_which_directory_to_point_at(runner, tmp_path):
    result = runner.invoke(app, ["dataset", "check", str(tmp_path)])
    assert result.exit_code == 1
    assert "not a LeRobot dataset" in result.output


# --------------------------------------------------------------------------- #
# dk1 dataset crop
# --------------------------------------------------------------------------- #


def test_crop_dry_run_prints_the_box_and_writes_nothing(runner, demos, repo_config, tmp_path):
    destination = tmp_path / "out"
    result = runner.invoke(
        app,
        ["dataset", "crop", str(demos), str(destination), "--dry-run", "-c", str(repo_config)],
    )
    assert result.exit_code == 0, result.output
    assert "observation.images.left" in result.output
    assert "deg H" in result.output
    assert "copied unchanged: ['observation.images.top']" in result.output
    assert not destination.exists()


def test_crop_defaults_the_destination_beside_the_source(runner, demos, repo_config):
    result = runner.invoke(
        app, ["dataset", "crop", str(demos), "--dry-run", "-c", str(repo_config)]
    )
    assert f"{demos}-optimized" in result.output


# --------------------------------------------------------------------------- #
# dk1 policy finetune — the gate
# --------------------------------------------------------------------------- #


def base(demos, checkpoint_dir, config_file, runs, row):
    return [
        "policy",
        "finetune",
        "--row",
        row,
        "--dataset-dir",
        str(demos),
        "--checkpoint",
        str(checkpoint_dir),
        "--runs-dir",
        str(runs),
        "--dry-run",
        "-c",
        str(config_file),
    ]


def test_R1_refuses_a_dataset_that_still_carries_the_full_lens(
    runner, demos, checkpoint_dir, config_file, tmp_path
):
    """The invariant STUDY.md names as the one that will be violated if any is.

    Train on frames the policy will never be shown again and the fine-tune is for
    a camera that does not exist — and nothing about the checkpoint says so.
    """
    result = runner.invoke(
        app, base(demos, checkpoint_dir, config_file, tmp_path / "runs", "R1")
    )
    assert result.exit_code == 1
    assert "carries the common lens" in result.output
    assert "dk1 dataset crop" in result.output


def test_A1_trains_on_the_demonstrations_as_recorded(
    runner, demos, checkpoint_dir, config_file, tmp_path
):
    runs = tmp_path / "runs"
    result = runner.invoke(app, base(demos, checkpoint_dir, config_file, runs, "A1"))
    assert result.exit_code == 0, result.output
    assert "lens        common" in result.output
    assert "--dry-run: nothing was trained" in result.output


def test_R1_accepts_the_cropped_copy(runner, demos, checkpoint_dir, config_file, tmp_path):
    runs = tmp_path / "runs"
    result = runner.invoke(
        app, base(cropped(demos), checkpoint_dir, config_file, runs, "R1")
    )
    assert result.exit_code == 0, result.output
    assert "lens        optimized" in result.output


def test_A1_refuses_the_cropped_copy(runner, demos, checkpoint_dir, config_file, tmp_path):
    """The gate has to work both ways: A1 on cropped frames is R1 wearing A1's name."""
    result = runner.invoke(
        app, base(cropped(demos), checkpoint_dir, config_file, tmp_path / "runs", "A1")
    )
    assert result.exit_code == 1
    assert "carries the optimized lens" in result.output


def test_a_dataset_with_problems_is_refused_unless_forced(
    runner, demos, checkpoint_dir, config_file, tmp_path
):
    (demos / "dk1_notes.jsonl").unlink()
    arguments = base(demos, checkpoint_dir, config_file, tmp_path / "runs", "A1")
    assert runner.invoke(app, arguments).exit_code == 1

    result = runner.invoke(app, arguments + ["--force"])
    # With no notes the lens is unknown, so the lens gate stops it first — which
    # is the right order: a dataset that cannot say what it is must not be forced
    # past the check that reads what it is.
    assert "carries the unknown lens" in result.output


def test_a_zero_shot_row_is_refused(runner, demos, checkpoint_dir, config_file, tmp_path):
    result = runner.invoke(
        app, base(demos, checkpoint_dir, config_file, tmp_path / "runs", "A0")
    )
    assert result.exit_code == 2
    assert "no such row" in result.output


# --------------------------------------------------------------------------- #
# dk1 policy finetune — what it writes
# --------------------------------------------------------------------------- #


@pytest.fixture
def run_dir(runner, demos, checkpoint_dir, config_file, tmp_path):
    runs = tmp_path / "runs"
    result = runner.invoke(app, base(demos, checkpoint_dir, config_file, runs, "A1"))
    assert result.exit_code == 0, result.output
    return next(runs.iterdir())


def test_the_run_directory_is_written_before_anything_is_trained(run_dir):
    """A run directory that only appears on success is missing from exactly the
    runs worth explaining."""
    assert (run_dir / "dk1_run.json").is_file()
    assert (run_dir / "dk1.toml").is_file()
    assert (run_dir / "command.txt").is_file()
    assert (run_dir / "dk1_command.txt").is_file()


def test_the_recorded_command_is_a_runnable_lerobot_train_line(run_dir):
    line = (run_dir / "command.txt").read_text()
    assert line.startswith("lerobot-train ")
    assert "--peft.r=32" in line
    assert "--peft.lora_alpha=16" in line
    assert "--policy.image_keys=" in line


def test_the_run_directory_names_the_dataset_and_its_split(run_dir):
    record = json.loads((run_dir / "dk1_run.json").read_text())
    assert record["dataset_episodes"] == 45
    assert record["dataset_lens"] == "common"
    assert len(record["split"]["holdout"]) == finetune.DEFAULT_BUDGET.holdout
    assert len(record["split"]["train"]) == 45 - finetune.DEFAULT_BUDGET.holdout


def test_the_run_directory_is_named_for_its_row(run_dir):
    assert run_dir.name.startswith("A1-")


def test_the_banner_says_the_inversion_goes_off(
    runner, demos, checkpoint_dir, config_file, tmp_path
):
    """Confirmed behaviourally on the first episode of the scored row, per
    STUDY.md — but it has to be said before the training run, not after it."""
    result = runner.invoke(
        app, base(demos, checkpoint_dir, config_file, tmp_path / "runs", "A1")
    )
    assert "--no-invert-gripper" in result.output
    assert "flip every grasp" in result.output


def test_the_flags_move_the_budget(runner, demos, checkpoint_dir, config_file, tmp_path):
    runs = tmp_path / "runs"
    result = runner.invoke(
        app,
        base(demos, checkpoint_dir, config_file, runs, "A1")
        + ["--steps", "500", "--batch-size", "1", "--eval-every", "50", "--holdout", "6"],
    )
    assert result.exit_code == 0, result.output
    record = json.loads((next(runs.iterdir()) / "dk1_run.json").read_text())
    assert record["budget"]["steps"] == 500
    assert record["budget"]["eval_steps"] == 50
    # save_freq follows eval_steps, so every checkpoint has a loss beside it.
    assert record["budget"]["save_freq"] == 50
    assert len(record["split"]["holdout"]) == 6


# --------------------------------------------------------------------------- #
# dk1 policy curve
# --------------------------------------------------------------------------- #


def test_the_curve_names_the_checkpoint_to_deploy(runner, run_dir):
    (run_dir / "train.log").write_text(
        "step 1000: eval_loss=0.4000\nstep 2000: eval_loss=0.3000\nstep 3000: eval_loss=0.3500\n"
    )
    for step in ("001000", "002000", "003000"):
        (run_dir / "train" / "checkpoints" / step).mkdir(parents=True)

    result = runner.invoke(app, ["policy", "curve", str(run_dir)])
    assert result.exit_code == 0, result.output
    assert "best: step 2000" in result.output
    assert "train/checkpoints/002000/pretrained_model" in result.output
    assert "--no-invert-gripper" in result.output


def test_the_curve_skips_a_best_loss_with_no_checkpoint(runner, run_dir):
    (run_dir / "train.log").write_text(
        "step 1000: eval_loss=0.4000\nstep 2000: eval_loss=0.3000\n"
    )
    (run_dir / "train" / "checkpoints" / "001000").mkdir(parents=True)
    result = runner.invoke(app, ["policy", "curve", str(run_dir)])
    assert "best: step 1000" in result.output


def test_the_curve_warns_when_the_last_evaluation_is_the_best(runner, run_dir):
    """Still improving means the budget may be short rather than the recipe wrong."""
    (run_dir / "train.log").write_text(
        "step 1000: eval_loss=0.4000\nstep 2000: eval_loss=0.3000\n"
    )
    for step in ("001000", "002000"):
        (run_dir / "train" / "checkpoints" / step).mkdir(parents=True)
    result = runner.invoke(app, ["policy", "curve", str(run_dir)])
    assert "still improving" in result.output


def test_a_run_with_no_log_says_so(runner, run_dir):
    result = runner.invoke(app, ["policy", "curve", str(run_dir)])
    assert result.exit_code == 1
    assert "does not exist" in result.output


def test_the_banner_says_what_the_adapter_reaches(
    runner, demos, checkpoint_dir, config_file, tmp_path
):
    """Under MolmoAct2's own default the 578M action expert is frozen, and that
    has to be visible before the night is spent rather than after it."""
    result = runner.invoke(
        app, base(demos, checkpoint_dir, config_file, tmp_path / "runs", "A1")
    )
    assert "action expert stays FROZEN" in result.output


def test_adapting_the_action_expert_reaches_the_command_line(
    runner, demos, checkpoint_dir, config_file, tmp_path
):
    runs = tmp_path / "runs"
    result = runner.invoke(
        app,
        base(demos, checkpoint_dir, config_file, runs, "A1") + ["--adapt", "vlm+expert"],
    )
    assert result.exit_code == 0, result.output
    line = (next(runs.iterdir()) / "command.txt").read_text()
    assert "--peft.target_modules=" in line
    assert "action_expert" in line


def test_a_misspelled_adapt_is_refused_before_anything_is_written(
    runner, demos, checkpoint_dir, config_file, tmp_path
):
    runs = tmp_path / "runs"
    result = runner.invoke(
        app, base(demos, checkpoint_dir, config_file, runs, "A1") + ["--adapt", "expert"]
    )
    assert result.exit_code == 1
    assert not runs.exists()
