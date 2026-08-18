"""Reading a MolmoAct2 checkpoint: what would make it unsafe to deploy here.

All of this is JSON in, verdict out, so none of it needs a checkpoint on disk —
which is the point: the checks that matter are the ones cheap enough to run
before every rollout.
"""

from __future__ import annotations

import json

import pytest

from dk1lab import checkpoint as ckpt
from dk1lab.layout import DOF, IMAGE_KEYS, yam_joint_offsets, yam_joint_signs

# --------------------------------------------------------------------------- #
# Fixtures: the three JSON files, shaped like the real BimanualYAM checkpoint
# --------------------------------------------------------------------------- #


def config_json(**overrides):
    raw = {
        "type": "molmoact2",
        "norm_tag": "yam_dual_molmoact2",
        "setup_type": "bimanual yam robotic arms in molmoact2",
        "control_mode": "absolute joint pose",
        "action_mode": "continuous",
        "inference_action_mode": "continuous",
        "model_dtype": "bfloat16",
        "device": "cpu",
        "chunk_size": 30,
        "n_action_steps": 30,
        "image_keys": [],
        "joint_signs": None,
        "pretrained_path": "/somewhere/else",
        "input_features": {"observation.state": {"type": "STATE", "shape": [DOF]}},
        "output_features": {"action": {"type": "ACTION", "shape": [DOF]}},
    }
    raw.update(overrides)
    return raw


def preprocessor_json(image_keys=IMAGE_KEYS, signs=None, offsets=None):
    return {
        "name": "policy_preprocessor",
        "steps": [
            {"registry_name": "to_batch_processor", "config": {}},
            {
                "registry_name": ckpt.STATE_TRANSFORM_STEP,
                "config": {"joint_signs": signs, "joint_offsets": offsets},
            },
            {
                "registry_name": ckpt.PACK_INPUTS_STEP,
                "config": {"image_keys": list(image_keys)},
            },
        ],
    }


def postprocessor_json(signs=None, offsets=None):
    return {
        "name": "policy_postprocessor",
        "steps": [
            {
                "registry_name": ckpt.ACTION_TRANSFORM_STEP,
                "config": {"joint_signs": signs, "joint_offsets": offsets},
            }
        ],
    }


def info(config=None, pre=None, post=None):
    return ckpt.parse(
        config or config_json(),
        pre or preprocessor_json(),
        post or postprocessor_json(),
    )


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def test_the_pipeline_image_order_is_read_from_the_preprocessor_not_the_config():
    """config.json's image_keys are empty in the real checkpoint, and ignored anyway."""
    parsed = info()
    assert parsed.config_image_keys == []
    assert parsed.pipeline_image_keys == list(IMAGE_KEYS)


def test_the_vector_dimensions_come_from_the_feature_shapes():
    parsed = info()
    assert parsed.state_dim == DOF
    assert parsed.action_dim == DOF


def test_a_checkpoint_with_no_saved_inversion_reports_none():
    assert info().gripper_inversion_baked_in is False


def test_a_checkpoint_that_inverts_itself_says_so():
    parsed = info(
        pre=preprocessor_json(signs=yam_joint_signs(), offsets=yam_joint_offsets()),
        post=postprocessor_json(signs=yam_joint_signs(), offsets=yam_joint_offsets()),
    )
    assert parsed.gripper_inversion_baked_in is True


# --------------------------------------------------------------------------- #
# Problems — the things that must stop a rollout
# --------------------------------------------------------------------------- #


def test_the_real_shape_of_the_bimanual_yam_checkpoint_has_no_problems():
    assert ckpt.problems(info()) == []


def test_the_wrong_norm_tag_is_a_problem():
    """Wrong statistics denormalise into the wrong units — it looks like it works."""
    problems = ckpt.problems(info(config=config_json(norm_tag="single_arm")))
    assert any("norm_tag" in p for p in problems)


def test_a_different_policy_type_is_a_problem():
    assert any("policy type" in p for p in ckpt.problems(info(config=config_json(type="pi0"))))


@pytest.mark.parametrize("dim", [7, 16])
def test_a_vector_that_is_not_14_d_is_a_problem(dim):
    config = config_json(
        input_features={"observation.state": {"type": "STATE", "shape": [dim]}},
        output_features={"action": {"type": "ACTION", "shape": [dim]}},
    )
    problems = ckpt.problems(info(config=config))
    assert len(problems) == 2


def test_the_alphabetical_image_order_is_a_problem():
    """sorted() gives left/right/top, which is not the order it was trained on."""
    scrambled = sorted(IMAGE_KEYS)
    assert scrambled != list(IMAGE_KEYS)
    problems = ckpt.problems(info(pre=preprocessor_json(image_keys=scrambled)))
    assert any("wrong order" in p for p in problems)


def test_no_pinned_image_order_at_all_is_a_problem():
    problems = ckpt.problems(info(pre=preprocessor_json(image_keys=[])))
    assert any("no image keys" in p for p in problems)


def test_a_checkpoint_carrying_a_different_transform_is_a_problem():
    """Its transform plus ours would be two transforms, and neither is intended."""
    other = [1.0] * DOF
    problems = ckpt.problems(info(pre=preprocessor_json(signs=other, offsets=other)))
    assert any("combine the two" in p for p in problems)


def test_a_checkpoint_that_already_inverts_exactly_as_we_do_is_not_a_problem():
    parsed = info(
        pre=preprocessor_json(signs=yam_joint_signs(), offsets=yam_joint_offsets()),
        post=postprocessor_json(signs=yam_joint_signs(), offsets=yam_joint_offsets()),
    )
    assert ckpt.problems(parsed) == []


# --------------------------------------------------------------------------- #
# Notes — said out loud, but not reasons to stop
# --------------------------------------------------------------------------- #


def test_a_cpu_device_is_a_note_not_a_problem():
    """The converted checkpoint says "device": "cpu"; left alone it runs there."""
    assert ckpt.problems(info()) == []
    assert any("device" in n for n in ckpt.notes(info()))


def test_the_missing_inversion_is_explained_in_the_notes():
    assert any("joint_signs" in n for n in ckpt.notes(info()))


def test_a_float32_checkpoint_is_noted():
    notes = ckpt.notes(info(config=config_json(model_dtype="float32")))
    assert any("float32" in n for n in notes)


# --------------------------------------------------------------------------- #
# Locating and reading
# --------------------------------------------------------------------------- #


def test_a_hugging_face_repo_id_passes_through_unchanged():
    assert ckpt.resolve("lerobot/MolmoAct2-BimanualYAM-LeRobot") == (
        "lerobot/MolmoAct2-BimanualYAM-LeRobot"
    )


def test_a_local_path_is_expanded_and_absolute(tmp_path):
    (tmp_path / "ckpt").mkdir()
    assert ckpt.resolve(tmp_path / "ckpt") == str(tmp_path / "ckpt")


def test_a_path_that_does_not_exist_fails_rather_than_becoming_a_repo_id():
    with pytest.raises(ckpt.CheckpointError, match="does not exist"):
        ckpt.resolve("/no/such/checkpoint")


def test_reading_a_directory_missing_its_pipelines_says_which(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(config_json()))
    with pytest.raises(ckpt.CheckpointError, match="policy_preprocessor.json"):
        ckpt.read(tmp_path)


def test_reading_a_complete_directory_round_trips(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(config_json()))
    (tmp_path / "policy_preprocessor.json").write_text(json.dumps(preprocessor_json()))
    (tmp_path / "policy_postprocessor.json").write_text(json.dumps(postprocessor_json()))
    (tmp_path / "model.safetensors").write_bytes(b"0" * 32)

    parsed = ckpt.read(tmp_path)
    assert parsed.norm_tag == "yam_dual_molmoact2"
    assert parsed.weights_bytes == 32
    assert ckpt.problems(parsed) == []


def test_invalid_json_names_the_file(tmp_path):
    (tmp_path / "config.json").write_text("{not json")
    (tmp_path / "policy_preprocessor.json").write_text("{}")
    (tmp_path / "policy_postprocessor.json").write_text("{}")
    with pytest.raises(ckpt.CheckpointError, match="config.json"):
        ckpt.read(tmp_path)
