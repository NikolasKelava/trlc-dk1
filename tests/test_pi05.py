"""π0.5's adaptation to this cell — the three gaps, and the one that must be loud.

Nothing here loads weights, touches a GPU or reaches the network. What it holds
is the part that would be wrong silently: the camera correspondence, the shape of
the borrowed statistics, and that a checkpoint carrying none of its own is
*reported* rather than quietly deployed.
"""

from __future__ import annotations

import json

import pytest

from dk1lab import checkpoint as ckpt
from dk1lab import pi05
from dk1lab.layout import ACTION_KEYS, CAMERA_NAMES, DOF, IMAGE_KEYS


def stats_payload(dim: int = DOF) -> dict:
    """A stats.json shaped like the one the study borrows."""
    values = [float(i) for i in range(dim)]
    entry = {
        "mean": values,
        "std": values,
        "min": values,
        "max": values,
        "q01": values,
        "q99": values,
    }
    return {"observation.state": dict(entry), "action": dict(entry)}


def info_payload(names=ACTION_KEYS) -> dict:
    return {
        "robot_type": "bi_dk1_follower",
        "features": {
            "observation.state": {"shape": [len(names)], "names": list(names)},
            "action": {"shape": [len(names)], "names": list(names)},
        },
    }


@pytest.fixture
def dataset_dir(tmp_path):
    """A LeRobot dataset directory holding only its metadata."""
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "stats.json").write_text(json.dumps(stats_payload()))
    (meta / "info.json").write_text(json.dumps(info_payload()))
    return tmp_path


# --------------------------------------------------------------------------- #
# The cameras
# --------------------------------------------------------------------------- #


def test_the_rename_covers_every_camera_this_cell_has():
    assert list(pi05.IMAGE_RENAME) == list(IMAGE_KEYS)
    assert len(pi05.IMAGE_RENAME) == len(CAMERA_NAMES)


def test_the_overhead_view_becomes_the_exterior_one_and_the_wrists_keep_their_sides():
    """The model embeds the views by position; this is the correspondence."""
    renamed = pi05.IMAGE_RENAME
    assert renamed["observation.images.top"].endswith("base_0_rgb")
    assert renamed["observation.images.left"].endswith("left_wrist_0_rgb")
    assert renamed["observation.images.right"].endswith("right_wrist_0_rgb")


def test_the_rename_is_derived_from_the_layout_not_written_out():
    """A camera rename cannot leave this table behind."""
    assert pi05.image_rename_map() == pi05.IMAGE_RENAME


# --------------------------------------------------------------------------- #
# The borrowed statistics
# --------------------------------------------------------------------------- #


def test_statistics_are_read_from_a_local_dataset(dataset_dir):
    stats = pi05.load_norm_stats(dataset_dir)
    assert set(stats.stats) == set(pi05.STATS_FEATURES)
    assert stats.layout_verified is True
    assert "borrowed" in stats.describe()


def test_a_stats_file_can_be_named_directly(dataset_dir):
    stats = pi05.load_norm_stats(dataset_dir / "meta" / "stats.json")
    assert stats.layout_verified is True


def test_statistics_for_a_different_robot_are_refused(tmp_path):
    """Same file shape, wrong width: the failure that looks like a working policy."""
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "stats.json").write_text(json.dumps(stats_payload(dim=32)))
    with pytest.raises(pi05.Pi05Error) as excinfo:
        pi05.load_norm_stats(tmp_path)
    assert "32-D" in str(excinfo.value)


def test_statistics_without_the_quantiles_are_refused(tmp_path):
    """The checkpoint normalises with QUANTILES, so mean and std are not enough."""
    meta = tmp_path / "meta"
    meta.mkdir()
    payload = stats_payload()
    for entry in payload.values():
        del entry["q01"]
    (meta / "stats.json").write_text(json.dumps(payload))
    with pytest.raises(pi05.Pi05Error) as excinfo:
        pi05.load_norm_stats(tmp_path)
    assert "q01" in str(excinfo.value)


def test_a_dataset_whose_channels_mean_something_else_is_refused(tmp_path):
    """Right width, wrong meaning per slot — every joint would move by the wrong amount."""
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "stats.json").write_text(json.dumps(stats_payload()))
    shuffled = list(reversed(ACTION_KEYS))
    (meta / "info.json").write_text(json.dumps(info_payload(shuffled)))
    with pytest.raises(pi05.Pi05Error) as excinfo:
        pi05.load_norm_stats(tmp_path)
    assert "different meaning per slot" in str(excinfo.value)


def test_statistics_without_an_info_file_are_used_but_say_they_were_not_checked(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "stats.json").write_text(json.dumps(stats_payload()))
    stats = pi05.load_norm_stats(tmp_path)
    assert stats.layout_verified is False
    assert "NOT verified" in stats.describe()


def test_a_missing_dataset_says_what_the_option_wants(tmp_path):
    with pytest.raises(pi05.Pi05Error) as excinfo:
        pi05.load_norm_stats(tmp_path / "nothing")
    assert "meta/stats.json" in str(excinfo.value)


def test_the_borrowed_normalisation_is_stated_in_every_description(dataset_dir):
    lines = " ".join(pi05.describe_stats(pi05.load_norm_stats(dataset_dir)))
    assert "BORROWED NORMALISATION" in lines


# --------------------------------------------------------------------------- #
# The processor overrides
# --------------------------------------------------------------------------- #


def test_the_overrides_carry_the_rename_the_stats_and_the_device(dataset_dir):
    pre, post = pi05.processor_overrides(pi05.load_norm_stats(dataset_dir), device="cuda")
    assert pre["rename_observations_processor"]["rename_map"] == pi05.IMAGE_RENAME
    assert pre["device_processor"]["device"] == "cuda"
    for step, overrides in (("normalizer_processor", pre), ("unnormalizer_processor", post)):
        features = overrides[step]["features"]
        assert set(features) == set(pi05.STATS_FEATURES)
        assert all(feature["shape"] == [DOF] for feature in features.values())


# --------------------------------------------------------------------------- #
# Reading the checkpoint
# --------------------------------------------------------------------------- #


PI05_CONFIG = {
    "type": "pi05",
    "device": "mps",
    "dtype": "float32",
    "chunk_size": 50,
    "n_action_steps": 50,
    "max_state_dim": 32,
    "max_action_dim": 32,
    "input_features": {
        "observation.images.base_0_rgb": {"type": "VISUAL", "shape": [3, 224, 224]},
        "observation.images.left_wrist_0_rgb": {"type": "VISUAL", "shape": [3, 224, 224]},
        "observation.images.right_wrist_0_rgb": {"type": "VISUAL", "shape": [3, 224, 224]},
        "observation.state": {"type": "STATE", "shape": [32]},
    },
    "output_features": {"action": {"type": "ACTION", "shape": [32]}},
}

PI05_PREPROCESSOR = {
    "name": "policy_preprocessor",
    "steps": [
        {"registry_name": "rename_observations_processor", "config": {"rename_map": {}}},
        {
            "registry_name": "normalizer_processor",
            "config": {"features": {}, "norm_map": {"STATE": "QUANTILES"}},
        },
    ],
}

PI05_POSTPROCESSOR = {
    "name": "policy_postprocessor",
    "steps": [{"registry_name": "unnormalizer_processor", "config": {"features": {}}}],
}


@pytest.fixture
def pi05_checkpoint(tmp_path):
    path = tmp_path / "pi05"
    path.mkdir()
    (path / "config.json").write_text(json.dumps(PI05_CONFIG))
    (path / "policy_preprocessor.json").write_text(json.dumps(PI05_PREPROCESSOR))
    (path / "policy_postprocessor.json").write_text(json.dumps(PI05_POSTPROCESSOR))
    return path


def test_a_pi05_checkpoint_is_read_and_judged_as_one(pi05_checkpoint):
    info = ckpt.read(pi05_checkpoint)
    assert info.policy_type == "pi05"
    assert (info.max_state_dim, info.max_action_dim) == (32, 32)
    # 32-D is the model's padded width, not a fault.
    assert ckpt.problems(info) == []


def test_the_missing_normalisation_is_reported_rather_than_passed_over(pi05_checkpoint):
    """The one thing about π0.5 that must never be silent."""
    said = " ".join(ckpt.notes(ckpt.read(pi05_checkpoint)))
    assert "BORROWED NORMALISATION" in said
    assert pi05.NORM_STATS_REPO in said


def test_the_gated_tokenizer_is_named_in_the_notes(pi05_checkpoint):
    said = " ".join(ckpt.notes(ckpt.read(pi05_checkpoint)))
    assert pi05.TOKENIZER_REPO in said
    assert "GATED" in said


def test_the_gripper_inversion_is_reported_as_off_for_pi05(pi05_checkpoint):
    said = " ".join(ckpt.notes(ckpt.read(pi05_checkpoint)))
    assert "inversion is OFF" in said


def test_image_features_that_do_not_match_the_rename_are_a_problem(pi05_checkpoint):
    """A mismatch feeds the wrong camera into each positional slot."""
    raw = json.loads((pi05_checkpoint / "config.json").read_text())
    raw["input_features"]["observation.images.somewhere_else"] = raw["input_features"].pop(
        "observation.images.base_0_rgb"
    )
    (pi05_checkpoint / "config.json").write_text(json.dumps(raw))
    found = ckpt.problems(ckpt.read(pi05_checkpoint))
    assert any("positionally" in problem for problem in found)


def test_an_unknown_policy_type_is_refused_by_name(pi05_checkpoint):
    raw = json.loads((pi05_checkpoint / "config.json").read_text())
    raw["type"] = "something_else"
    (pi05_checkpoint / "config.json").write_text(json.dumps(raw))
    found = ckpt.problems(ckpt.read(pi05_checkpoint))
    assert found and "something_else" in found[0]
