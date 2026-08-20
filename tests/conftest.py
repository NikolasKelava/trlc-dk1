"""Shared fixtures. Nothing here touches hardware."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

VALID_TOML = """\
version = 1

# a comment that must survive every surgical write
[arms.follower]
left = "/dev/ttyACM1"
right = "/dev/ttyACM3"

[arms.leader]
left = "/dev/ttyACM0"
right = "/dev/ttyACM2"

# camera comment
[cameras.top]
path = "/dev/v4l/by-path/pci-top-video-index0"
rotation = 180
hfov = 105.0

[cameras.left]
path = "/dev/v4l/by-path/pci-left-video-index0"
rotation = 180
hfov = 105.0
target_hfov = 87.0

[cameras.right]
path = "/dev/v4l/by-path/pci-right-video-index0"
rotation = 0

[capture.policy]
width = 640
height = 360
fps = 30
fourcc = "MJPG"

[capture.teleop]
width = 1280
height = 720
fps = 60
fourcc = "MJPG"

[policy]
checkpoint = "/does/not/exist/molmoact2_bf16"
"""


#: A MolmoAct2 checkpoint's three JSON files, shaped like the BimanualYAM one —
#: including the two settings this fork has to override (device cpu, float32) and
#: the saved pipelines' missing gripper inversion (joint_signs: null).
CHECKPOINT_CONFIG = {
    "type": "molmoact2",
    "norm_tag": "yam_dual_molmoact2",
    "setup_type": "bimanual yam robotic arms in molmoact2",
    "control_mode": "absolute joint pose",
    "action_mode": "continuous",
    "inference_action_mode": None,
    "model_dtype": "float32",
    "device": "cpu",
    "use_amp": False,
    "image_keys": [],
    "chunk_size": 30,
    "n_action_steps": 30,
    "input_features": {
        "observation.images.top": {"type": "VISUAL", "shape": [3, 224, 224]},
        "observation.state": {"type": "STATE", "shape": [14]},
    },
    "output_features": {"action": {"type": "ACTION", "shape": [14]}},
}

CHECKPOINT_PREPROCESSOR = {
    "name": "policy_preprocessor",
    "steps": [
        {
            "registry_name": "molmoact2_state_frame_transform",
            "config": {"joint_signs": None, "joint_offsets": None},
        },
        {
            "registry_name": "molmoact2_pack_inputs",
            "config": {
                "image_keys": [
                    "observation.images.top",
                    "observation.images.left",
                    "observation.images.right",
                ]
            },
        },
    ],
}

CHECKPOINT_POSTPROCESSOR = {
    "name": "policy_postprocessor",
    "steps": [
        {
            "registry_name": "molmoact2_action_frame_transform",
            "config": {"joint_signs": None, "joint_offsets": None},
        }
    ],
}


@pytest.fixture
def checkpoint_dir(tmp_path: Path) -> Path:
    """A checkpoint directory with real metadata and no weights."""
    path = tmp_path / "checkpoint"
    path.mkdir()
    (path / "config.json").write_text(json.dumps(CHECKPOINT_CONFIG))
    (path / "policy_preprocessor.json").write_text(json.dumps(CHECKPOINT_PREPROCESSOR))
    (path / "policy_postprocessor.json").write_text(json.dumps(CHECKPOINT_POSTPROCESSOR))
    return path


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    """A valid dk1.toml in a temp dir, with comments to prove writes preserve them."""
    path = tmp_path / "dk1.toml"
    path.write_text(VALID_TOML)
    return path


@pytest.fixture
def repo_config() -> Path:
    """The real dk1.toml tracked in this repo."""
    return REPO_ROOT / "dk1.toml"
