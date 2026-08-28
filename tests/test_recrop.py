"""Applying the `optimized` crop to a recorded dataset, and reading one back.

The demonstrations are recorded through the full lens so that one day of hands
serves both R1 and A1; this is the derivation between the two rows, and it is
the only place in this study where a frame is rewritten rather than captured.

Two properties carry everything else:

* **only the wrist streams change, and every frame keeps its size and count.**
  The copy inherits the source's ``meta/`` — every timestamp, every episode's
  frame range — and that is only still true if this holds. A video one frame
  short would shift every observation against its action, silently;
* **the box is the camera's box.** :func:`dk1lab.recrop.plan` goes through
  :func:`dk1lab.fov.crop_box`, the same call :mod:`dk1lab.cameras` makes. A
  second implementation would be a second lens, and R1 would be fine-tuned for a
  camera that does not exist.

Real mp4 files are written and read here — small ones — because the arithmetic
was never the risk. Nothing needs a robot or a GPU.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from dk1lab import recrop
from dk1lab.config import load
from dk1lab.dataset import DatasetError, summarise
from dk1lab.fov import crop_box
from dk1lab.layout import CAMERA_NAMES

WIDTH, HEIGHT, FRAMES = 160, 90, 12


@pytest.fixture
def settings(config_file):
    return load(config_file, require_devices=False)


def write_video(path, frames=FRAMES, width=WIDTH, height=HEIGHT):
    """A short mp4 with a distinctive pattern per frame, so a crop is visible."""
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as target:
        stream = target.add_stream("libx264", 30, options={"crf": "18"})
        stream.pix_fmt, stream.width, stream.height = "yuv420p", width, height
        for index in range(frames):
            image = np.zeros((height, width, 3), dtype=np.uint8)
            # A bright band down the left edge: cropped away, it proves the box
            # was applied rather than the file merely re-encoded.
            image[:, : width // 8] = 255
            image[:, width // 2 :, 1] = 128
            image[index % height, :, 2] = 255
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            packet = stream.encode(frame)
            if packet:
                target.mux(packet)
        packet = stream.encode()
        if packet:
            target.mux(packet)


@pytest.fixture
def dataset(tmp_path):
    """A v3.0-shaped dataset: three camera streams, notes, episode metadata.

    Built by hand rather than through ``DatasetSession`` so it costs no torch
    import and no encoder, and so a test can make one field wrong on purpose.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = tmp_path / "demos"
    features = {
        f"observation.images.{name}": {
            "dtype": "video",
            "shape": [HEIGHT, WIDTH, 3],
            "names": ["height", "width", "channels"],
            "info": {"video.fps": 30.0, "video.codec": "h264", "video.g": 4, "video.crf": 30},
        }
        for name in CAMERA_NAMES
    }
    info = {
        "codebase_version": "v3.0",
        "repo_id": "dk1/demos",
        "robot_type": "bi_dk1_follower_safe",
        "fps": 30,
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
    }
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps(info))

    episodes, scenes = 6, [1, 1, 2, 2, 3, 3]
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "episode_index": list(range(episodes)),
                "length": [FRAMES // episodes * 2] * episodes,
                "tasks": [["put the dice in the bowl"]] * episodes,
            }
        ),
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )
    (root / "dk1_notes.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "episode": index,
                    "scene": scenes[index],
                    "profile": "common",
                    "capture": "policy",
                    "fps": 30,
                    "vcodec": "h264_nvenc",
                }
            )
            for index in range(episodes)
        )
        + "\n"
    )
    for name in CAMERA_NAMES:
        write_video(
            root / "videos" / f"observation.images.{name}" / "chunk-000" / "file-000.mp4"
        )
    return root


# --------------------------------------------------------------------------- #
# The plan
# --------------------------------------------------------------------------- #


def test_only_the_cameras_with_a_target_are_cropped(settings):
    """`dk1.toml` gives the wrists a target_hfov and the top view none."""
    plan = recrop.plan(settings, width=WIDTH, height=HEIGHT)
    assert set(plan.boxes) == {"observation.images.left"}
    assert "observation.images.top" not in plan.boxes


def test_the_box_is_the_cameras_own_box(settings):
    """Through `fov.crop_box`, not a second implementation of it."""
    plan = recrop.plan(settings, width=WIDTH, height=HEIGHT)
    device = settings.camera("left")
    assert plan.boxes["observation.images.left"] == crop_box(
        WIDTH,
        HEIGHT,
        device.hfov,
        device.target_hfov,
        inset=device.crop_inset,
        shift_x=device.crop_shift_x,
        shift_y=device.crop_shift_y,
    )


def test_the_real_config_crops_both_wrists_and_not_the_top(repo_config):
    """The cell's own dk1.toml, at the capture size it records demonstrations at."""
    plan = recrop.plan(load(repo_config, require_devices=False), width=1280, height=720)
    assert set(plan.boxes) == {"observation.images.left", "observation.images.right"}


def test_the_plan_describes_the_field_of_view_it_achieves(settings):
    """Not the one that was asked for: the rounding leaves it wider and the inset
    then narrows it, and printing the request would hide exactly that."""
    plan = recrop.plan(settings, width=WIDTH, height=HEIGHT)
    assert "deg H" in plan.lines["observation.images.left"]


def test_a_camera_name_is_turned_into_its_image_key():
    assert recrop.image_key("top") == "observation.images.top"
    with pytest.raises(recrop.RecropError):
        recrop.image_key("wrist_left")


# --------------------------------------------------------------------------- #
# Doing it
# --------------------------------------------------------------------------- #


def test_the_copy_keeps_every_frame_and_its_size(dataset, settings, tmp_path):
    """The property the inherited metadata rests on."""
    import av

    plan = recrop.plan(settings, width=WIDTH, height=HEIGHT)
    destination = tmp_path / "demos-optimized"
    report = recrop.crop_dataset(dataset, destination, plan, vcodec="h264")

    assert report.frames == FRAMES
    with av.open(str(destination / "videos" / "observation.images.left" / "chunk-000" / "file-000.mp4")) as handle:
        stream = handle.streams.video[0]
        assert (int(stream.width), int(stream.height)) == (WIDTH, HEIGHT)
        assert sum(1 for _ in handle.decode(stream)) == FRAMES


def test_the_uncropped_stream_is_copied_byte_for_byte(dataset, settings, tmp_path):
    """The top view is not re-encoded, so it does not pay a generation of loss."""
    plan = recrop.plan(settings, width=WIDTH, height=HEIGHT)
    destination = tmp_path / "demos-optimized"
    original = (dataset / "videos" / "observation.images.top" / "chunk-000" / "file-000.mp4").read_bytes()
    recrop.crop_dataset(dataset, destination, plan, vcodec="h264")
    copied = (destination / "videos" / "observation.images.top" / "chunk-000" / "file-000.mp4").read_bytes()
    assert copied == original


def test_the_pixels_actually_change(dataset, settings, tmp_path):
    """The bright band down the left edge is outside the crop, so it goes."""
    import av

    plan = recrop.plan(settings, width=WIDTH, height=HEIGHT)
    destination = tmp_path / "demos-optimized"
    recrop.crop_dataset(dataset, destination, plan, vcodec="h264")

    def first_frame(root):
        path = root / "videos" / "observation.images.left" / "chunk-000" / "file-000.mp4"
        with av.open(str(path)) as handle:
            return next(handle.decode(handle.streams.video[0])).to_ndarray(format="rgb24")

    before, after = first_frame(dataset), first_frame(destination)
    assert before.shape == after.shape
    # The band is 1/8 of the frame; the crop keeps roughly the middle 70%, so the
    # left edge of the cropped picture is no longer saturated white.
    assert before[:, :4].mean() > 200
    assert after[:, :4].mean() < 200


def test_the_source_is_never_written_to(dataset, settings, tmp_path):
    """It is the one artefact of this study a day of hands cannot buy back."""
    before = {
        path: path.stat().st_mtime_ns
        for path in sorted(dataset.rglob("*"))
        if path.is_file()
    }
    plan = recrop.plan(settings, width=WIDTH, height=HEIGHT)
    recrop.crop_dataset(dataset, tmp_path / "out", plan, vcodec="h264")
    after = {
        path: path.stat().st_mtime_ns
        for path in sorted(dataset.rglob("*"))
        if path.is_file()
    }
    assert before == after


def test_the_metadata_is_carried_over_unchanged(dataset, settings, tmp_path):
    plan = recrop.plan(settings, width=WIDTH, height=HEIGHT)
    destination = tmp_path / "out"
    recrop.crop_dataset(dataset, destination, plan, vcodec="h264")
    assert (destination / "meta" / "info.json").read_text() == (
        dataset / "meta" / "info.json"
    ).read_text()
    assert (destination / "dk1_notes.jsonl").read_text() == (
        dataset / "dk1_notes.jsonl"
    ).read_text()


def test_the_copy_says_what_lens_it_carries(dataset, settings, tmp_path):
    plan = recrop.plan(settings, width=WIDTH, height=HEIGHT)
    destination = tmp_path / "out"
    recrop.crop_dataset(dataset, destination, plan, vcodec="h264")

    lens = recrop.read_lens(destination)
    assert lens["profile"] == "optimized"
    assert lens["source"] == str(dataset)
    assert "observation.images.left" in lens["streams"]
    assert lens["unchanged"] == ["observation.images.top", "observation.images.right"]


def test_an_existing_destination_is_refused_unless_asked(dataset, settings, tmp_path):
    """An hour of transcoding is cheap next to deleting the wrong directory."""
    plan = recrop.plan(settings, width=WIDTH, height=HEIGHT)
    destination = tmp_path / "out"
    destination.mkdir()
    with pytest.raises(recrop.RecropError) as excinfo:
        recrop.crop_dataset(dataset, destination, plan, vcodec="h264")
    assert "--overwrite" in str(excinfo.value)


def test_overwrite_replaces_it(dataset, settings, tmp_path):
    plan = recrop.plan(settings, width=WIDTH, height=HEIGHT)
    destination = tmp_path / "out"
    destination.mkdir()
    (destination / "stale").write_text("x")
    recrop.crop_dataset(dataset, destination, plan, vcodec="h264", overwrite=True)
    assert not (destination / "stale").exists()


def test_a_box_planned_for_another_frame_size_is_refused(dataset, settings, tmp_path):
    """The box is in pixels, so a box for the wrong frame is a different lens —
    and the resulting dataset would look entirely normal."""
    plan = recrop.plan(settings, width=WIDTH * 2, height=HEIGHT * 2)
    with pytest.raises(recrop.RecropError) as excinfo:
        recrop.crop_dataset(dataset, tmp_path / "out", plan, vcodec="h264")
    assert "crop was computed for" in str(excinfo.value)


def test_a_directory_that_is_not_a_dataset_says_so(tmp_path):
    with pytest.raises(recrop.RecropError) as excinfo:
        recrop.read_info(tmp_path)
    assert "meta/info.json" in str(excinfo.value)


def test_a_config_with_no_crop_refuses_rather_than_copying(dataset, repo_config, tmp_path):
    """Under `--profile common` there is nothing to apply, and a copy that changed
    nothing while claiming the optimized lens is the worst possible outcome."""
    from dk1lab.runprofile import apply, resolve

    settings = apply(load(repo_config, require_devices=False), resolve("common"))
    with pytest.raises(recrop.RecropError) as excinfo:
        recrop.crop_dataset(dataset, tmp_path / "out", recrop.plan(settings, width=WIDTH, height=HEIGHT))
    assert "no crop" in str(excinfo.value)


def test_nothing_is_left_behind_when_a_transcode_fails(dataset, settings, tmp_path, monkeypatch):
    """A half-written video beside the real one would be picked up by the glob."""
    plan = recrop.plan(settings, width=WIDTH, height=HEIGHT)
    monkeypatch.setattr(
        recrop, "_verify", lambda *args, **kwargs: (_ for _ in ()).throw(recrop.RecropError("no"))
    )
    destination = tmp_path / "out"
    with pytest.raises(recrop.RecropError):
        recrop.crop_dataset(dataset, destination, plan, vcodec="h264")
    assert not list(destination.rglob("*.dk1crop.mp4"))


# --------------------------------------------------------------------------- #
# What lens a dataset carries
# --------------------------------------------------------------------------- #


def test_a_recorded_dataset_reports_the_profile_from_its_notes(dataset):
    from dk1lab.finetune import read_notes

    assert recrop.lens_profile(dataset, read_notes(dataset)) == "common"


def test_the_lens_file_outranks_the_notes(dataset, settings, tmp_path):
    """They answer different questions — what it was captured as, and what was
    done to it — and the second is the later fact."""
    from dk1lab.finetune import read_notes

    plan = recrop.plan(settings, width=WIDTH, height=HEIGHT)
    destination = tmp_path / "out"
    recrop.crop_dataset(dataset, destination, plan, vcodec="h264")
    assert recrop.lens_profile(destination, read_notes(destination)) == "optimized"


def test_a_dataset_that_says_nothing_reports_no_lens(tmp_path):
    """It must not be trained on without somebody looking at it."""
    assert recrop.lens_profile(tmp_path, []) is None


def test_episodes_recorded_under_two_profiles_report_no_lens(tmp_path):
    notes = [{"episode": 0, "profile": "common"}, {"episode": 1, "profile": "optimized"}]
    assert recrop.lens_profile(tmp_path, notes) is None


# --------------------------------------------------------------------------- #
# Reading a dataset back
# --------------------------------------------------------------------------- #


def test_a_recorded_dataset_reads_back_clean(dataset):
    summary = summarise(dataset)
    assert summary.problems == []
    assert summary.episodes == 6
    assert summary.tasks == ("put the dice in the bowl",)
    assert summary.by_scene == {1: 2, 2: 2, 3: 2}
    assert set(summary.cameras) == {f"observation.images.{name}" for name in CAMERA_NAMES}


def test_a_second_task_string_is_a_problem(dataset):
    """The prompt has to be byte-identical on every frame — it is what both
    fine-tunes condition on."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = dataset / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(path)
    pq.write_table(
        table.set_column(
            table.column_names.index("tasks"),
            "tasks",
            pa.array([["put the dice in the bowl"]] * 5 + [["pick up the dice"]]),
        ),
        path,
    )
    assert any("task strings" in problem for problem in summarise(dataset).problems)


def test_a_missing_camera_is_a_problem(dataset):
    import shutil

    shutil.rmtree(dataset / "videos" / "observation.images.right")
    info = json.loads((dataset / "meta" / "info.json").read_text())
    del info["features"]["observation.images.right"]
    (dataset / "meta" / "info.json").write_text(json.dumps(info))
    assert any("missing camera" in problem for problem in summarise(dataset).problems)


def test_episodes_recorded_under_two_profiles_are_a_problem(dataset):
    """One dataset is one observation path; two is two experiments mixed."""
    lines = (dataset / "dk1_notes.jsonl").read_text().splitlines()
    record = json.loads(lines[-1])
    record["profile"] = "optimized"
    (dataset / "dk1_notes.jsonl").write_text("\n".join(lines[:-1] + [json.dumps(record)]) + "\n")
    assert any("disagree about profile" in problem for problem in summarise(dataset).problems)


def test_the_episode_metadata_a_crash_ate_is_reported(dataset):
    """The 2026-08-25 freeze left seven episodes' videos intact and their
    per-frame state gone. It has to be visible before a night is spent on it."""
    import shutil

    shutil.rmtree(dataset / "meta" / "episodes")
    problems = summarise(dataset).problems
    assert any("meta/episodes/ is missing" in problem for problem in problems)


def test_a_dataset_with_no_notes_cannot_be_stratified_and_says_so(dataset):
    (dataset / "dk1_notes.jsonl").unlink()
    assert any("no dk1_notes.jsonl" in problem for problem in summarise(dataset).problems)


def test_a_directory_that_is_not_a_dataset_raises(tmp_path):
    with pytest.raises(DatasetError):
        summarise(tmp_path)


# --------------------------------------------------------------------------- #
# The gripper command that was never executed
# --------------------------------------------------------------------------- #


def with_data(root, action, state=None):
    """Give a dataset fixture a data/ table, so it can be clamped."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    action = np.asarray(action, dtype=np.float64)
    state = np.asarray(state if state is not None else action, dtype=np.float64)
    directory = root / "data" / "chunk-000"
    directory.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "episode_index": list(range(len(action))),
                "action": pa.array(action.tolist(), type=pa.list_(pa.float32(), 14)),
                "observation.state": pa.array(state.tolist(), type=pa.list_(pa.float32(), 14)),
            }
        ),
        directory / "file-000.parquet",
    )
    return root


def rows(*values):
    """One 14-D row per value, with that value in both gripper channels."""
    from dk1lab.layout import GRIPPER_INDICES

    out = []
    for value in values:
        row = [0.5] * 14
        for index in GRIPPER_INDICES:
            row[index] = value
        out.append(row)
    return out


def test_a_gripper_command_past_the_stop_is_clipped(dataset):
    """The robot clips to [0, 1] internally, so 1.03 describes a motion that did
    not happen — and MolmoAct2 refuses it, which is how it was found."""
    import pyarrow.parquet as pq

    from dk1lab.dataset import clamp_gripper
    from dk1lab.layout import GRIPPER_INDICES

    with_data(dataset, rows(0.06, 1.0234, 0.5, -0.01))
    report = clamp_gripper(dataset)
    assert report.written and report.changed == 2
    low, high = report.before[GRIPPER_INDICES[0]]  # float32 on disk
    assert (low, high) == (pytest.approx(-0.01, abs=1e-6), pytest.approx(1.0234, abs=1e-6))

    action = np.array(
        pq.read_table(dataset / "data" / "chunk-000" / "file-000.parquet")
        .column("action")
        .to_pylist(),
        dtype=np.float64,
    )
    for index in GRIPPER_INDICES:
        assert action[:, index].max() <= 1.0
        assert action[:, index].min() >= 0.0


def test_the_arm_joints_are_not_touched(dataset):
    """They are radians and legitimately outside [0, 1]; clipping them would pin
    every arm to a wrist's worth of travel."""
    import pyarrow.parquet as pq

    from dk1lab.dataset import clamp_gripper

    table = [[3.125] * 14, [-1.577] * 14]
    with_data(dataset, table)
    clamp_gripper(dataset)
    action = np.array(
        pq.read_table(dataset / "data" / "chunk-000" / "file-000.parquet")
        .column("action")
        .to_pylist(),
        dtype=np.float64,
    )
    arm = [i for i in range(14) if i not in __import__("dk1lab.layout", fromlist=["x"]).GRIPPER_INDICES]
    assert action[:, arm].max() == pytest.approx(3.125, abs=1e-4)
    assert action[:, arm].min() == pytest.approx(-1.577, abs=1e-4)


def test_the_measured_state_is_left_alone(dataset):
    """Clipping a measurement would be inventing data. It is also never needed:
    the follower physically cannot pass its own stop."""
    import pyarrow.parquet as pq

    from dk1lab.dataset import clamp_gripper
    from dk1lab.layout import GRIPPER_INDICES

    with_data(dataset, rows(1.03), state=rows(1.03))
    clamp_gripper(dataset)
    state = np.array(
        pq.read_table(dataset / "data" / "chunk-000" / "file-000.parquet")
        .column("observation.state")
        .to_pylist(),
        dtype=np.float64,
    )
    assert state[0][GRIPPER_INDICES[0]] == pytest.approx(1.03, abs=1e-4)


def test_a_dry_run_reports_and_writes_nothing(dataset):
    from dk1lab.dataset import CLAMP_FILE, clamp_gripper

    with_data(dataset, rows(1.03, 0.5))
    report = clamp_gripper(dataset, dry_run=True)
    assert report.needed and not report.written
    assert not (dataset / CLAMP_FILE).exists()


def test_a_dataset_already_in_range_is_not_rewritten(dataset):
    from dk1lab.dataset import CLAMP_FILE, clamp_gripper

    with_data(dataset, rows(0.0, 0.5, 1.0))
    report = clamp_gripper(dataset)
    assert not report.needed and not report.written
    assert not (dataset / CLAMP_FILE).exists()


def test_clamping_is_idempotent(dataset):
    from dk1lab.dataset import clamp_gripper

    with_data(dataset, rows(1.03, 0.5))
    assert clamp_gripper(dataset).changed == 1
    assert clamp_gripper(dataset).changed == 0


def test_what_was_recorded_is_written_down_before_it_is_changed(dataset):
    """A repair that leaves no trace of what it repaired is indistinguishable
    from a recording that was always like that."""
    from dk1lab.dataset import CLAMP_FILE, clamp_gripper

    with_data(dataset, rows(1.03, 0.5))
    clamp_gripper(dataset)
    record = json.loads((dataset / CLAMP_FILE).read_text())
    assert record["range"] == [0.0, 1.0]
    assert record["changed"] == 1
    assert record["recorded_range"]["6"][1] == pytest.approx(1.03, abs=1e-4)


def test_the_stats_stop_describing_a_command_that_no_longer_exists(dataset):
    """`lerobot_train` hands `dataset.meta.stats` to the processors, so a max of
    1.03 left behind would still describe a frame that is not in the data."""
    from dk1lab.dataset import clamp_gripper
    from dk1lab.layout import GRIPPER_INDICES

    (dataset / "meta" / "stats.json").write_text(
        json.dumps(
            {
                "action": {
                    name: [1.03 if i in GRIPPER_INDICES else 9.0 for i in range(14)]
                    for name in ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99")
                }
            }
        )
    )
    with_data(dataset, rows(1.03, 0.5))
    clamp_gripper(dataset)
    stats = json.loads((dataset / "meta" / "stats.json").read_text())["action"]
    assert stats["max"][GRIPPER_INDICES[0]] == pytest.approx(1.0)
    # Every other channel is untouched: rewriting statistics that did not change
    # is how a repair turns into a difference nobody can account for.
    assert stats["max"][0] == 9.0


def test_the_check_reports_an_out_of_range_gripper(dataset):
    """Reported by `dk1 dataset check` rather than discovered by a training run,
    which is how it was discovered the first time."""
    with_data(dataset, rows(1.03, 0.5))
    problems = summarise(dataset).problems
    assert any("gripper" in problem and "dk1 dataset clamp" in problem for problem in problems)


def test_the_check_is_quiet_when_the_gripper_is_in_range(dataset):
    with_data(dataset, rows(0.0, 0.5, 1.0))
    assert not any("gripper" in problem for problem in summarise(dataset).problems)
