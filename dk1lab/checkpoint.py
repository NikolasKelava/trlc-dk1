"""Read a MolmoAct2 checkpoint's metadata without loading 11 GB of weights.

Everything here is JSON parsing, so it costs milliseconds, needs no GPU, no
LeRobot and no torch — which is what makes ``dk1 policy check`` something you can
run before every rollout instead of a thing you do once.

What it is actually for is the checkpoint's *pipelines*, not its config. A
LeRobot checkpoint stores two things that both claim to describe deployment:

``config.json``
    the policy config. This is what ``--policy.<key>=...`` overrides on the CLI,
    and what people read when they want to know how the policy runs.

``policy_preprocessor.json`` / ``policy_postprocessor.json``
    the *actual* pre/post-processing pipelines, saved step by step with their
    own copies of the same settings.

When a policy is loaded from a path — which is always, during rollout —
``lerobot.policies.factory.make_pre_post_processors`` takes the ``pretrained_path``
branch and rebuilds the pipelines from **those two JSON files**. The policy
config's own ``joint_signs`` / ``joint_offsets`` / ``image_keys`` are never
consulted. So ``--policy.joint_signs=...`` on a ``lerobot-rollout`` command line
parses, validates, is stored on the config, and then does nothing at all.

That matters here more than anywhere else, because the gripper inversion this
cell needs is expressed exactly that way — and the saved pipelines in the
BimanualYAM checkpoint have ``joint_signs: null``. :func:`problems` reports the
resulting state of affairs plainly, and :func:`dk1lab.policy.apply_gripper_inversion`
is what actually fixes it, on the loaded pipeline objects.

**Two policies, one reader.** Since the two-policy comparison this cell now runs
(``STUDY.md``) needs the same check for π0.5, :func:`problems` and :func:`notes`
dispatch on the checkpoint's own ``type``. The reading is shared — a LeRobot
policy directory is a LeRobot policy directory — and only the *judging* differs,
because what counts as a usable checkpoint is a fact about the policy: MolmoAct2
has to arrive 14-D with the right normalisation tag and the right image order,
while π0.5 arrives 32-D by design and brings no statistics at all. See
:mod:`dk1lab.pi05` for what is done about the latter.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .layout import DOF, IMAGE_KEYS

#: The policy type this fork deploys. Anything else is a different checkpoint.
EXPECTED_TYPE = "molmoact2"

#: Normalisation statistics tag for the bimanual YAM data MolmoAct2 was trained
#: on. Wrong tag = correct-looking actions in the wrong units.
EXPECTED_NORM_TAG = "yam_dual_molmoact2"

#: Prompt metadata baked into the checkpoint. Reported, not enforced.
EXPECTED_SETUP_TYPE = "bimanual yam robotic arms in molmoact2"
EXPECTED_CONTROL_MODE = "absolute joint pose"

#: Files a LeRobot policy directory must contain to be loadable.
REQUIRED_FILES = (
    "config.json",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
)

#: Registry names of the two pipeline steps that carry the gripper inversion.
STATE_TRANSFORM_STEP = "molmoact2_state_frame_transform"
ACTION_TRANSFORM_STEP = "molmoact2_action_frame_transform"

#: Registry name of the step that pins the camera order.
PACK_INPUTS_STEP = "molmoact2_pack_inputs"

#: Registry names of the two generic steps π0.5's adaptation goes through.
RENAME_STEP = "rename_observations_processor"
NORMALIZER_STEP = "normalizer_processor"

#: The policy types this fork knows how to judge.
KNOWN_TYPES = ("molmoact2", "pi05")


class CheckpointError(Exception):
    """Raised when a checkpoint cannot be read at all."""


# --------------------------------------------------------------------------- #
# Locating
# --------------------------------------------------------------------------- #


def resolve(spec: str | Path) -> str:
    """Expand ``~`` in a checkpoint spec, leaving Hugging Face repo ids alone.

    Returns a string because that is what LeRobot's loaders take; a local
    directory comes back absolute, a repo id comes back unchanged.
    """
    text = str(spec)
    expanded = Path(text).expanduser()
    if expanded.exists():
        return str(expanded.resolve())
    if text.startswith("~") or text.startswith("/") or text.startswith("."):
        # Meant as a path, and it is not there. Say so now rather than letting
        # huggingface_hub try to fetch a repo named "/home/...".
        raise CheckpointError(
            f"checkpoint {text!r} looks like a path but {expanded} does not exist"
        )
    return text


def is_local(spec: str | Path) -> bool:
    """True if ``spec`` names a directory on this machine."""
    return Path(str(spec)).expanduser().is_dir()


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CheckpointInfo:
    """What a checkpoint directory says about itself.

    ``pipeline_*`` fields come from the saved processor pipelines and are the
    ones that actually take effect at rollout; the others come from
    ``config.json``.
    """

    path: Path
    policy_type: str | None = None
    norm_tag: str | None = None
    setup_type: str | None = None
    control_mode: str | None = None
    action_mode: str | None = None
    inference_action_mode: str | None = None
    model_dtype: str | None = None
    device: str | None = None
    chunk_size: int | None = None
    n_action_steps: int | None = None
    state_dim: int | None = None
    action_dim: int | None = None
    pretrained_path: str | None = None
    config_image_keys: list[str] = field(default_factory=list)
    config_joint_signs: list[float] | None = None
    #: Image order as the *pipeline* will apply it — this is the one that counts.
    pipeline_image_keys: list[str] = field(default_factory=list)
    pipeline_state_signs: list[float] | None = None
    pipeline_state_offsets: list[float] | None = None
    pipeline_action_signs: list[float] | None = None
    pipeline_action_offsets: list[float] | None = None
    weights_bytes: int = 0
    #: Image feature keys declared in ``config.json``'s ``input_features``. For
    #: π0.5 this is where the camera names live; MolmoAct2 pins them in the
    #: pipeline instead.
    input_image_keys: list[str] = field(default_factory=list)
    #: π0.5's padded internal widths. ``None`` on a checkpoint that has no such
    #: notion, which is every MolmoAct2 one.
    max_state_dim: int | None = None
    max_action_dim: int | None = None
    #: The saved rename step's map, which is empty on both base checkpoints and
    #: is what :mod:`dk1lab.pi05` overrides.
    pipeline_rename_map: dict[str, str] = field(default_factory=dict)
    #: Feature keys the saved normalizer covers. Empty means it normalises
    #: nothing, which is π0.5's published state and a problem to be closed.
    pipeline_norm_features: list[str] = field(default_factory=list)

    @property
    def gripper_inversion_baked_in(self) -> bool:
        """True if the saved pipelines already invert something themselves."""
        return self.pipeline_state_signs is not None or self.pipeline_action_signs is not None


def parse(
    config_raw: dict[str, Any],
    preprocessor_raw: dict[str, Any],
    postprocessor_raw: dict[str, Any],
    *,
    path: Path = Path("."),
    weights_bytes: int = 0,
) -> CheckpointInfo:
    """Build a :class:`CheckpointInfo` from already-decoded JSON.

    Split from :func:`read` so the checks can be tested without a 11 GB
    directory on disk.
    """
    features = config_raw.get("input_features") or {}
    outputs = config_raw.get("output_features") or {}
    pre_steps = _steps(preprocessor_raw)
    post_steps = _steps(postprocessor_raw)
    pack = pre_steps.get(PACK_INPUTS_STEP, {})
    state_transform = pre_steps.get(STATE_TRANSFORM_STEP, {})
    action_transform = post_steps.get(ACTION_TRANSFORM_STEP, {})
    rename = pre_steps.get(RENAME_STEP, {})
    normalizer = pre_steps.get(NORMALIZER_STEP, {})

    return CheckpointInfo(
        path=path,
        policy_type=config_raw.get("type"),
        norm_tag=config_raw.get("norm_tag"),
        setup_type=config_raw.get("setup_type"),
        control_mode=config_raw.get("control_mode"),
        action_mode=config_raw.get("action_mode"),
        inference_action_mode=config_raw.get("inference_action_mode"),
        model_dtype=config_raw.get("model_dtype") or config_raw.get("dtype"),
        device=config_raw.get("device"),
        chunk_size=config_raw.get("chunk_size"),
        n_action_steps=config_raw.get("n_action_steps"),
        state_dim=_dim(features.get("observation.state")),
        action_dim=_dim(outputs.get("action")),
        pretrained_path=config_raw.get("pretrained_path"),
        config_image_keys=list(config_raw.get("image_keys") or []),
        config_joint_signs=config_raw.get("joint_signs"),
        pipeline_image_keys=list(pack.get("image_keys") or []),
        pipeline_state_signs=state_transform.get("joint_signs"),
        pipeline_state_offsets=state_transform.get("joint_offsets"),
        pipeline_action_signs=action_transform.get("joint_signs"),
        pipeline_action_offsets=action_transform.get("joint_offsets"),
        weights_bytes=weights_bytes,
        input_image_keys=[
            key
            for key, feature in features.items()
            if isinstance(feature, dict) and feature.get("type") == "VISUAL"
        ],
        max_state_dim=config_raw.get("max_state_dim"),
        max_action_dim=config_raw.get("max_action_dim"),
        pipeline_rename_map=dict(rename.get("rename_map") or {}),
        pipeline_norm_features=list((normalizer.get("features") or {}).keys()),
    )


def _steps(pipeline_raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """``{registry_name: config}`` for one saved pipeline."""
    return {
        step.get("registry_name", ""): step.get("config") or {}
        for step in pipeline_raw.get("steps") or []
    }


def _dim(feature: Any) -> int | None:
    if not isinstance(feature, dict):
        return None
    shape = feature.get("shape")
    if isinstance(shape, list) and shape:
        return int(shape[-1])
    return None


def read(spec: str | Path) -> CheckpointInfo:
    """Read a **local** checkpoint directory.

    Args:
        spec: directory path; ``~`` is expanded.

    Raises:
        CheckpointError: if the directory or any required file is missing, or if
            a file is not valid JSON. A Hugging Face repo id cannot be read this
            way — download it first, or skip the check.
    """
    path = Path(str(spec)).expanduser()
    if not path.is_dir():
        raise CheckpointError(
            f"{path} is not a directory. `dk1 policy check` reads a local checkpoint; "
            f"a Hugging Face repo id has to be downloaded first."
        )
    missing = [name for name in REQUIRED_FILES if not (path / name).exists()]
    if missing:
        raise CheckpointError(
            f"{path}: missing {missing}. A LeRobot policy directory carries its config and "
            f"both processor pipelines; without them the checkpoint loads with no prompt "
            f"metadata and no normalisation statistics."
        )

    raws = []
    for name in REQUIRED_FILES:
        try:
            raws.append(json.loads((path / name).read_text()))
        except json.JSONDecodeError as exc:
            raise CheckpointError(f"{path / name}: invalid JSON — {exc}") from exc

    weights = path / "model.safetensors"
    return parse(
        *raws,
        path=path,
        weights_bytes=weights.stat().st_size if weights.exists() else 0,
    )


# --------------------------------------------------------------------------- #
# Judging
# --------------------------------------------------------------------------- #


def problems(info: CheckpointInfo) -> list[str]:
    """Reasons not to deploy this checkpoint on this cell. Empty is good.

    Dispatches on the checkpoint's own ``type``: what makes a checkpoint usable
    is a fact about the policy, not about the directory. An unrecognised type is
    itself the problem — this fork deploys exactly two.
    """
    if info.policy_type == "pi05":
        return _pi05_problems(info)
    if info.policy_type != EXPECTED_TYPE:
        return [
            f"policy type is {info.policy_type!r}; this cell deploys {list(KNOWN_TYPES)}"
        ]
    return _molmoact2_problems(info)


def notes(info: CheckpointInfo) -> list[str]:
    """Things worth saying out loud that are not reasons to stop."""
    if info.policy_type == "pi05":
        return _pi05_notes(info)
    return _molmoact2_notes(info)


# --------------------------------------------------------------------------- #
# MolmoAct2
# --------------------------------------------------------------------------- #


def _molmoact2_problems(info: CheckpointInfo) -> list[str]:
    found: list[str] = []

    if info.policy_type != EXPECTED_TYPE:
        found.append(f"policy type is {info.policy_type!r}, expected {EXPECTED_TYPE!r}")
    if info.norm_tag != EXPECTED_NORM_TAG:
        found.append(
            f"norm_tag is {info.norm_tag!r}, expected {EXPECTED_NORM_TAG!r} — the wrong "
            f"statistics denormalise actions into the wrong units, which looks like a "
            f"working policy moving to the wrong places"
        )
    if info.state_dim != DOF:
        found.append(f"observation.state is {info.state_dim}-D, expected {DOF}")
    if info.action_dim != DOF:
        found.append(f"action is {info.action_dim}-D, expected {DOF}")

    keys = info.pipeline_image_keys
    if keys and list(keys) != list(IMAGE_KEYS):
        found.append(
            f"the saved preprocessor pins image keys {keys}, but this cell provides "
            f"{list(IMAGE_KEYS)} — the views would be fed to the model in the wrong order"
        )
    if not keys:
        found.append(
            "the saved preprocessor pins no image keys, so it falls back to whatever "
            "order the observation arrives in; the trained order is "
            f"{list(IMAGE_KEYS)} and 'left' < 'right' < 'top' sorted is not it"
        )

    # A checkpoint that already inverts, plus our own inversion applied on top,
    # is two inversions and therefore none.
    for what, signs, offsets in (
        ("preprocessor", info.pipeline_state_signs, info.pipeline_state_offsets),
        ("postprocessor", info.pipeline_action_signs, info.pipeline_action_offsets),
    ):
        if signs is None:
            continue
        expected_signs, expected_offsets = _expected_inversion()
        if list(signs) != expected_signs or list(offsets or []) != expected_offsets:
            found.append(
                f"the saved {what} already applies its own joint transform "
                f"(signs={signs}, offsets={offsets}) which is not the gripper inversion "
                f"this cell applies — deploying would combine the two"
            )

    return found


def _molmoact2_notes(info: CheckpointInfo) -> list[str]:
    said: list[str] = []

    if info.device and info.device != "cuda":
        said.append(
            f'config.json says "device": {info.device!r}; overridden to cuda at load. '
            f"Left alone it silently runs the whole policy on the CPU."
        )
    if info.model_dtype != "bfloat16":
        said.append(
            f"model_dtype is {info.model_dtype!r}; this cell loads bfloat16, so a float32 "
            f"checkpoint pays a ~20 s cast on every process start"
        )
    if info.pretrained_path and not str(info.path) == info.pretrained_path:
        said.append(
            f'config.json carries a stale absolute "pretrained_path" '
            f"({info.pretrained_path}); overridden at load"
        )
    if info.inference_action_mode != "continuous":
        said.append(
            f"inference_action_mode is {info.inference_action_mode!r}; set to 'continuous' "
            f"at load, which is what RTC requires"
        )
    if info.setup_type != EXPECTED_SETUP_TYPE:
        said.append(f"setup_type is {info.setup_type!r}, expected {EXPECTED_SETUP_TYPE!r}")
    if info.control_mode != EXPECTED_CONTROL_MODE:
        said.append(f"control_mode is {info.control_mode!r}, expected {EXPECTED_CONTROL_MODE!r}")

    if not info.gripper_inversion_baked_in:
        said.append(
            "the saved pipelines do not invert the gripper channel (joint_signs: null). "
            "That is expected, and it is why `dk1 policy` patches the two frame-transform "
            "steps after loading: --policy.joint_signs on a lerobot-rollout command line "
            "is parsed and then ignored, because the pipelines are rebuilt from these "
            "JSON files rather than from the policy config."
        )
    if info.config_joint_signs:
        said.append(
            f"config.json carries joint_signs={info.config_joint_signs}, which LeRobot does "
            f"NOT apply when the pipelines are loaded from a path — do not rely on it"
        )
    if info.config_image_keys and list(info.config_image_keys) != list(IMAGE_KEYS):
        said.append(
            f"config.json image_keys {info.config_image_keys} disagree with the saved "
            f"pipeline's {info.pipeline_image_keys}; the pipeline wins"
        )

    return said


def _expected_inversion() -> tuple[list[float], list[float]]:
    from .layout import yam_joint_offsets, yam_joint_signs

    return yam_joint_signs(), yam_joint_offsets()


# --------------------------------------------------------------------------- #
# π0.5
# --------------------------------------------------------------------------- #
#
# π0.5's base checkpoint is judged against a different bar than MolmoAct2's, and
# deliberately a lower one. MolmoAct2-BimanualYAM is *supposed* to arrive fitting
# this cell — 14-D, the right norm tag, the right image order — so any departure
# is a fault. π0.5-base is supposed to arrive fitting nothing in particular: 32-D
# padded, three camera names of its own, and a normalizer covering no features at
# all. None of that is wrong with the checkpoint; it is what a general-purpose
# base model looks like, and :mod:`dk1lab.pi05` is what closes each gap.
#
# So the problems below are only the ones that would mean the adaptation cannot
# be applied, and everything the adaptation *does* apply is reported as a note —
# loudly, because "borrowed normalisation" has to travel with every number this
# policy produces.


def _pi05_problems(info: CheckpointInfo) -> list[str]:
    """Reasons π0.5 cannot be adapted to this cell. Padding is not one of them."""
    from .pi05 import PI05_IMAGE_NAMES

    found: list[str] = []

    expected_images = [f"observation.images.{name}" for name in PI05_IMAGE_NAMES]
    if info.input_image_keys and list(info.input_image_keys) != expected_images:
        found.append(
            f"config.json declares image features {info.input_image_keys}, but the "
            f"rename in dk1lab/pi05.py targets {expected_images}. The model embeds the "
            f"views positionally, so a mismatch feeds the wrong camera to each slot"
        )

    for what, dim, width in (
        ("state", info.state_dim, info.max_state_dim),
        ("action", info.action_dim, info.max_action_dim),
    ):
        if width is None:
            found.append(
                f"config.json has no max_{what}_dim, so there is no padded width to "
                f"narrow to {DOF}-D. This is not the pi05 checkpoint this was written for"
            )
        elif dim is not None and DOF > width:
            found.append(
                f"this cell's {what} is {DOF}-D but the checkpoint pads to {width}-D; "
                f"a {DOF}-D vector does not fit"
            )

    return found


def _pi05_notes(info: CheckpointInfo) -> list[str]:
    """What the adaptation will change, said out loud before anything runs."""
    from .pi05 import IMAGE_RENAME, NORM_STATS_REPO, TOKENIZER_REPO

    said: list[str] = []

    if not info.pipeline_norm_features:
        said.append(
            f"the saved normalizer covers NO features, so a literal zero-shot load "
            f"normalises nothing — the policy would be handed raw radians where it "
            f"expects roughly [-1, 1]. {DOF}-D statistics are borrowed from "
            f"{NORM_STATS_REPO} instead. π0.5 zero-shot on this cell is ZERO-SHOT "
            f"WEIGHTS, BORROWED NORMALISATION, and must be labelled that way."
        )
    if info.state_dim != DOF or info.action_dim != DOF:
        said.append(
            f"config.json declares state {info.state_dim}-D and action "
            f"{info.action_dim}-D — the model's padded width, not a robot's. Both are "
            f"narrowed to {DOF} at load, which is what trims the action chunk to this "
            f"cell's {DOF} channels rather than 32 with 18 meaningless ones"
        )
    if not info.pipeline_rename_map:
        said.append(
            "the saved rename step is empty; this cell's "
            + ", ".join(f"{ours.rsplit('.', 1)[-1]} -> {theirs.rsplit('.', 1)[-1]}"
                        for ours, theirs in IMAGE_RENAME.items())
            + " is applied as an override at load"
        )
    if info.device and info.device != "cuda":
        said.append(
            f'config.json says "device": {info.device!r}; overridden to cuda at load'
        )
    if info.model_dtype not in (None, "bfloat16"):
        said.append(
            f"dtype is {info.model_dtype!r}; this cell loads bfloat16 (8.8 GiB measured)"
        )
    said.append(
        f"the prompt tokenizer comes from {TOKENIZER_REPO}, a GATED Hugging Face "
        f"repository. `dk1 policy smoke` checks it is reachable before loading weights."
    )
    said.append(
        "the gripper inversion is OFF for π0.5, always: its normalisation comes from "
        "DK1 data in DK1 convention, so there is nothing to flip"
    )
    return said
