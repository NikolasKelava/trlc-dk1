"""π0.5 on this cell — the second policy in the two-policy comparison.

``lerobot/pi05_base`` is a flow-matching VLA on PaliGemma with the broadest open
cross-embodiment pretraining there is. It is also, unlike MolmoAct2's BimanualYAM
checkpoint, **not** trained on anything shaped like this robot, and three
mismatches have to be closed before it can be handed a DK1 observation. Each is
closed here, once, so that ``dk1 policy check`` / ``smoke`` and every later
rollout close them the same way.

**1. It speaks 32-D, we speak 14-D.** ``max_state_dim`` and ``max_action_dim``
are both 32; states are zero-padded up to that width inside the model and action
chunks come back 32 wide. What decides where the padding is cut off is
``config.output_features["action"].shape`` — ``predict_action_chunk`` trims the
chunk to it — and ``config.input_features["observation.state"].shape`` likewise.
Setting both to :data:`dk1lab.layout.DOF` is the whole of it: the chunk comes
back ``(1, 50, 14)``, in this cell's key order.

**2. Its camera names are not ours.** The checkpoint's image features are
``base_0_rgb`` / ``left_wrist_0_rgb`` / ``right_wrist_0_rgb``, and the model
embeds the views **positionally** — the first is the exterior view, the other two
are the wrists. Our cameras are ``top`` / ``left`` / ``right``, which is a
constraint MolmoAct2 imposes and which nothing here may change. So the *keys are
renamed*, in the pipeline's own ``rename_observations_processor`` step, rather
than the features being rewritten: :data:`IMAGE_RENAME` is the map, and it is the
one place the correspondence is written down.

**3. It has no normalisation statistics for this robot, and cannot make any.**
The saved normalizer has ``features: {}`` and ``stats`` for nothing at all: the
base checkpoint has never seen a 14-D DK1 vector, so a literal zero-shot load
normalises nothing and the policy is handed raw radians where it expects roughly
[-1, 1]. The mitigation ``STUDY.md`` fixes in advance is to borrow
:data:`NORM_STATS_REPO`'s ``meta/stats.json`` — 1823 episodes of this exact
``bi_dk1_follower`` layout, whose ``observation.state`` and ``action`` names are
:data:`dk1lab.layout.ACTION_KEYS` verbatim, and which carries the ``q01`` / ``q99``
quantiles the checkpoint's ``QUANTILES`` norm map asks for.

    **This makes π0.5 zero-shot "zero-shot weights, borrowed normalisation", and
    it must be labelled that way every time it is quoted.** Every command that
    builds this policy says so on stdout, which is why :func:`describe_stats`
    exists and why nothing here is silent about it.

**The gripper inversion is off, always.** MolmoAct2's weights genuinely speak YAM
(1 = open) and this cell is 0 = open, which is why that checkpoint is inverted.
π0.5's normalisation now comes from DK1 data recorded in DK1 convention, so there
is nothing to flip — inverting would introduce the error, not remove it. See
``STUDY.md`` § *The gripper convention*.

**One thing here is not in our hands.** The pipeline's tokenizer step loads
``google/paligemma-3b-pt-224``, which is a **gated** repository on the Hugging
Face Hub. Nothing about it can be worked around from this side, and substituting
some third-party mirror of a tokenizer would silently change what the prompt
means. :func:`tokenizer_available` checks for it up front so the failure is a
sentence about accepting a licence rather than a traceback out of
``transformers``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .checkpoint import resolve
from .layout import ACTION_KEYS, CAMERA_NAMES, DOF, IMAGE_KEYS

logger = logging.getLogger(__name__)

#: The policy type string in ``config.json``.
POLICY_TYPE = "pi05"

#: The base checkpoint, and where this machine keeps it.
CHECKPOINT_REPO = "lerobot/pi05_base"
DEFAULT_CHECKPOINT = "~/Documents/RobotLearning/policies/pi05/base"

#: The gated repository the pipeline's tokenizer step loads.
TOKENIZER_REPO = "google/paligemma-3b-pt-224"

#: Where the borrowed 14-D normalisation statistics come from.
#:
#: 1823 episodes, 2.2 M frames, ``robot_type: bi_dk1_follower``, and
#: ``meta/info.json`` names its ``observation.state`` and ``action`` channels with
#: exactly :data:`dk1lab.layout.ACTION_KEYS`. That last fact is checked rather
#: than assumed — see :func:`verify_layout`.
NORM_STATS_REPO = "andreaskoepf/dk1-merge-2026-03"
NORM_STATS_FILE = "meta/stats.json"
NORM_STATS_INFO_FILE = "meta/info.json"

#: This cell's camera keys, mapped onto the ones π0.5 was pretrained with.
#:
#: The order is what carries the meaning: the model embeds the views positionally,
#: so ``top`` has to become the exterior view and the two wrists have to keep
#: their sides. Written as a map from *our* key so that the left-hand column is
#: always :data:`dk1lab.layout.IMAGE_KEYS` and a change to the camera names
#: cannot leave this behind — :func:`image_rename_map` derives it and
#: ``tests/test_pi05.py`` asserts the correspondence.
PI05_IMAGE_NAMES: tuple[str, ...] = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")

#: Statistic names the ``QUANTILES`` normalisation mode needs.
REQUIRED_STATS = ("q01", "q99")

#: The two vector features the borrowed statistics have to cover.
STATS_FEATURES = ("observation.state", "action")


class Pi05Error(RuntimeError):
    """Raised when π0.5 cannot be set up for this cell, with the reason."""


def image_rename_map() -> dict[str, str]:
    """``{our image key: the key π0.5 was pretrained with}``, in camera order.

    Derived from :data:`dk1lab.layout.CAMERA_NAMES` rather than written out, so
    the two lists cannot drift and the positional correspondence stays visible.
    """
    if len(CAMERA_NAMES) != len(PI05_IMAGE_NAMES):
        raise Pi05Error(
            f"this cell has {len(CAMERA_NAMES)} cameras and π0.5 was pretrained with "
            f"{len(PI05_IMAGE_NAMES)}; the correspondence in dk1lab/pi05.py has to be "
            f"rewritten rather than guessed"
        )
    return {
        ours: f"observation.images.{theirs}"
        for ours, theirs in zip(IMAGE_KEYS, PI05_IMAGE_NAMES, strict=True)
    }


#: The rename applied to every observation before it reaches the model.
IMAGE_RENAME: dict[str, str] = image_rename_map()


# --------------------------------------------------------------------------- #
# The borrowed normalisation statistics
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class NormStats:
    """14-D statistics for this cell, and where they came from.

    ``source`` is carried so that every command can say it out loud. A π0.5
    number quoted without "borrowed normalisation" beside it is a number about
    something else.
    """

    stats: dict[str, dict[str, list[float]]]
    source: str
    #: ``True`` when ``meta/info.json`` was read and its channel names matched
    #: this cell's layout exactly. ``False`` means the names could not be checked.
    layout_verified: bool = False

    def describe(self) -> str:
        """One line naming what was borrowed and from where."""
        checked = "layout verified" if self.layout_verified else "layout NOT verified"
        return f"borrowed {DOF}-D normalisation from {self.source} ({checked})"


def load_norm_stats(source: str | Path | None = None) -> NormStats:
    """The 14-D statistics π0.5 needs, from a dataset directory or from the Hub.

    Args:
        source: a local LeRobot dataset directory (or a path straight to a
            ``stats.json``), or ``None`` to fetch :data:`NORM_STATS_REPO` from the
            Hub. The download is one small JSON file and is cached.

    Raises:
        Pi05Error: if the statistics are missing, are not 14-D, or lack the
            quantiles the checkpoint's ``QUANTILES`` norm map needs. Every one of
            those would otherwise surface as a policy that moves confidently to
            the wrong places.
    """
    if source is None:
        raw, info_raw, name = _from_hub()
    else:
        raw, info_raw, name = _from_path(Path(str(source)).expanduser())

    validate(raw, name)
    return NormStats(stats=raw, source=name, layout_verified=verify_layout(info_raw))


def _from_hub() -> tuple[dict, dict | None, str]:
    from huggingface_hub import hf_hub_download

    def fetch(filename: str) -> dict | None:
        try:
            path = hf_hub_download(NORM_STATS_REPO, filename, repo_type="dataset")
        except Exception as exc:  # noqa: BLE001 - the reason is what the operator needs
            if filename == NORM_STATS_FILE:
                raise Pi05Error(
                    f"could not fetch {NORM_STATS_REPO}/{filename}: {exc}. π0.5 has no "
                    f"normalisation statistics of its own for this robot, so there is "
                    f"nothing to fall back to — pass --norm-stats with a local LeRobot "
                    f"dataset instead."
                ) from exc
            logger.debug("could not fetch %s: %s", filename, exc)
            return None
        return json.loads(Path(path).read_text())

    stats = fetch(NORM_STATS_FILE)
    assert stats is not None  # fetch raises for the stats file
    return stats, fetch(NORM_STATS_INFO_FILE), f"{NORM_STATS_REPO}/{NORM_STATS_FILE}"


def _from_path(path: Path) -> tuple[dict, dict | None, str]:
    stats_path = path if path.is_file() else path / "meta" / NORM_STATS_FILE.split("/")[-1]
    if not stats_path.is_file():
        raise Pi05Error(
            f"{stats_path} does not exist. --norm-stats wants a LeRobot dataset "
            f"directory (the one holding meta/stats.json) or that file itself."
        )
    info_path = stats_path.parent / "info.json"
    info = json.loads(info_path.read_text()) if info_path.is_file() else None
    return json.loads(stats_path.read_text()), info, str(stats_path)


def validate(raw: Any, source: str) -> None:
    """Check statistics cover this cell's 14-D vectors. Raises, never warns."""
    if not isinstance(raw, dict):
        raise Pi05Error(f"{source}: expected a JSON object of per-feature statistics")
    for feature in STATS_FEATURES:
        entry = raw.get(feature)
        if not isinstance(entry, dict):
            raise Pi05Error(
                f"{source}: no statistics for {feature!r}. π0.5's saved normalizer covers "
                f"nothing at all, so these are the only ones there would be."
            )
        for name in REQUIRED_STATS:
            values = entry.get(name)
            if not isinstance(values, list):
                raise Pi05Error(
                    f"{source}: {feature}.{name} is missing. The checkpoint normalises "
                    f"state and action with QUANTILES, which needs {list(REQUIRED_STATS)}."
                )
            if len(values) != DOF:
                raise Pi05Error(
                    f"{source}: {feature}.{name} is {len(values)}-D, expected {DOF}. "
                    f"These statistics are for a different robot."
                )


def verify_layout(info_raw: Any) -> bool:
    """Whether a dataset's channel names are this cell's layout, exactly.

    Returns ``False`` — rather than raising — when there is no ``info.json`` to
    read, because a statistics file without one is still usable and the caller
    says so out loud instead. A file that *does* name its channels and names them
    differently is a genuine error: same width, different meaning per slot, which
    is the failure that looks like a working policy.
    """
    if not isinstance(info_raw, dict):
        return False
    features = info_raw.get("features")
    if not isinstance(features, dict):
        return False
    for feature in STATS_FEATURES:
        names = (features.get(feature) or {}).get("names")
        if not isinstance(names, list):
            return False
        if list(names) != list(ACTION_KEYS):
            raise Pi05Error(
                f"the borrowed statistics are for {feature} channels {names}, but this "
                f"cell's layout is {list(ACTION_KEYS)}. Same width, different meaning per "
                f"slot — normalising with these would move every joint by the wrong amount."
            )
    return True


def describe_stats(stats: NormStats) -> list[str]:
    """The lines every π0.5 command prints about where its normalisation came from."""
    return [
        stats.describe(),
        "π0.5 zero-shot on this cell is ZERO-SHOT WEIGHTS, BORROWED NORMALISATION.",
        "Label it that way wherever the number is quoted.",
    ]


# --------------------------------------------------------------------------- #
# The policy
# --------------------------------------------------------------------------- #


def tokenizer_available() -> tuple[bool, str]:
    """Whether the gated PaliGemma tokenizer can be reached. ``(ok, explanation)``.

    Checked before the weights are loaded, because loading them costs a minute
    and the failure is a licence acceptance rather than anything about this code.
    """
    try:
        from huggingface_hub import hf_hub_download

        hf_hub_download(TOKENIZER_REPO, "tokenizer_config.json")
    except Exception as exc:  # noqa: BLE001 - the message is the product here
        return False, (
            f"π0.5's prompt tokenizer comes from {TOKENIZER_REPO}, which is a GATED "
            f"repository on the Hugging Face Hub and is not reachable from here:\n"
            f"    {type(exc).__name__}: {exc}\n"
            f"Accept the licence at https://huggingface.co/{TOKENIZER_REPO} with the "
            f"account you want to use, then `hf auth login` (or set HF_TOKEN).\n"
            f"Do not substitute a third-party mirror: the tokenizer decides what the "
            f"prompt means, and the study compares two policies on one prompt."
        )
    return True, f"{TOKENIZER_REPO} is reachable"


def require_tokenizer() -> None:
    """:func:`tokenizer_available`, as an exception."""
    ok, explanation = tokenizer_available()
    if not ok:
        raise Pi05Error(explanation)


def policy_config(
    checkpoint: str = DEFAULT_CHECKPOINT,
    *,
    device: str = "cuda",
    dtype: str = "bfloat16",
) -> Any:
    """π0.5's config with this cell's 14-D vectors and this machine's device.

    Three fields are overridden and each of them changes behaviour:

    ``device``
        the published checkpoint says ``"mps"``, which on this machine LeRobot
        warns about and reroutes — but stating it is cheaper than relying on that.
    ``dtype``
        ``float32`` as published; 4.14 B parameters in bf16 is 8.8 GiB of the
        5090's 32, measured.
    ``input_features`` / ``output_features``
        the published 32-D shapes are the model's *padded* width, not a robot's.
        Narrowing them to 14 is what makes ``predict_action_chunk`` return a
        ``(1, 50, 14)`` chunk in this cell's key order instead of 32 columns with
        18 of them meaningless.
    """
    import lerobot.policies  # noqa: F401 - registers "pi05" in the config registry
    from lerobot.configs import PreTrainedConfig
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.utils.constants import ACTION, OBS_STATE

    path = resolve(checkpoint)
    config = PreTrainedConfig.from_pretrained(path)
    if config.type != POLICY_TYPE:
        raise Pi05Error(
            f"{path} is a {config.type!r} checkpoint, not {POLICY_TYPE!r}. "
            f"dk1lab.pi05 builds π0.5 only."
        )
    config.pretrained_path = path
    config.device = device
    config.dtype = dtype
    config.input_features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(DOF,))
    config.output_features[ACTION] = PolicyFeature(type=FeatureType.ACTION, shape=(DOF,))
    return config


def _norm_features() -> dict[str, dict[str, Any]]:
    """The two features the normalizer steps are told to cover, as saved JSON shape."""
    from lerobot.utils.constants import ACTION, OBS_STATE

    return {
        OBS_STATE: {"type": "STATE", "shape": [DOF]},
        ACTION: {"type": "ACTION", "shape": [DOF]},
    }


def processor_overrides(
    stats: NormStats, *, device: str = "cuda"
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The pre/post-processor overrides that make π0.5 usable here.

    Returned rather than applied so that a caller can print them: these three
    overrides *are* the adaptation, and a run that could not say what it changed
    about the pipeline would not be evidence about π0.5.

    The rename comes first in the saved pipeline, so the normalizer's features
    are named in post-rename terms — which for the two vector features means they
    are unchanged, since only the image keys move.
    """
    features = _norm_features()
    preprocessor = {
        "rename_observations_processor": {"rename_map": dict(IMAGE_RENAME)},
        "normalizer_processor": {"features": features, "stats": stats.stats},
        "device_processor": {"device": device},
    }
    postprocessor = {
        "unnormalizer_processor": {"features": features, "stats": stats.stats},
    }
    return preprocessor, postprocessor


def load(
    checkpoint: str = DEFAULT_CHECKPOINT,
    *,
    stats: NormStats | None = None,
    device: str = "cuda",
    dtype: str = "bfloat16",
) -> tuple[Any, Any, Any, NormStats]:
    """Load π0.5 and build both pipelines. Returns ``(policy, pre, post, stats)``.

    Checks the gated tokenizer **before** loading 14 GB of weights, so an
    unaccepted licence costs a second rather than a minute.
    """
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    require_tokenizer()
    stats = stats if stats is not None else load_norm_stats()
    config = policy_config(checkpoint, device=device, dtype=dtype)

    logger.info("loading %s ...", config.pretrained_path)
    policy = get_policy_class(config.type).from_pretrained(
        config.pretrained_path, config=config
    )
    policy = policy.to(device)
    policy.eval()
    policy.requires_grad_(False)

    pre_over, post_over = processor_overrides(stats, device=device)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=config.pretrained_path,
        preprocessor_overrides=pre_over,
        postprocessor_overrides=post_over,
    )
    return policy, preprocessor, postprocessor, stats


# --------------------------------------------------------------------------- #
# The smoke test
# --------------------------------------------------------------------------- #


def smoke(
    checkpoint: str = DEFAULT_CHECKPOINT,
    *,
    task: str,
    steps: int = 5,
    device: str = "cuda",
    dtype: str = "bfloat16",
    width: int = 1280,
    height: int = 720,
    stats: NormStats | None = None,
) -> Any:
    """Run π0.5's deployment inference path on a synthetic observation. No robot.

    The same shape of check :func:`dk1lab.policy.smoke` runs for MolmoAct2, and
    it returns the same :class:`~dk1lab.policy.SmokeResult` so one printer serves
    both — with ``rtc_ms`` empty, because RTC is not the path either policy is
    deployed through here and timing it would be measuring something nothing
    runs.

    What it proves: the weights load, the three overrides take, the borrowed
    statistics are accepted, and inference returns a 14-D action in this cell's
    key order. What it cannot prove is that the actions are any *good* — the
    images are noise.
    """
    import time

    import numpy as np
    import torch
    from lerobot.rollout.inference.sync import SyncInferenceEngine
    from lerobot.utils.constants import OBS_STR
    from lerobot.utils.feature_utils import build_dataset_frame

    from .layout import STATE_KEYS
    from .policy import ROBOT_TYPE, SmokeResult, dataset_features

    policy, preprocessor, postprocessor, stats = load(
        checkpoint, stats=stats, device=device, dtype=dtype
    )

    features = dataset_features(width=width, height=height)
    engine = SyncInferenceEngine(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        dataset_features=features,
        ordered_action_keys=list(ACTION_KEYS),
        task=task,
        device=device,
        robot_type=ROBOT_TYPE,
    )

    rng = np.random.default_rng(0)
    values: dict[str, Any] = dict.fromkeys(STATE_KEYS, 0.0)
    values.update(
        {
            name: rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
            for name in CAMERA_NAMES
        }
    )
    frame = build_dataset_frame(features, values, OBS_STR)

    start = time.perf_counter()
    action = engine.get_action(frame)
    warmup_ms = (time.perf_counter() - start) * 1000.0

    chunk_ms: list[float] = []
    pop_ms: list[float] = []
    for _ in range(max(1, steps)):
        # Resetting empties the action queue, so this call has to run the model
        # rather than answer from the 50-step chunk it already holds.
        engine.reset()
        start = time.perf_counter()
        action = engine.get_action(frame)
        chunk_ms.append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        engine.get_action(frame)
        pop_ms.append((time.perf_counter() - start) * 1000.0)

    if action is None or len(action) != DOF:
        raise Pi05Error(
            f"expected a {DOF}-D action, got {None if action is None else len(action)}. "
            f"The 32-D padding is trimmed by config.output_features['action'], which "
            f"dk1lab.pi05.policy_config narrows — see its docstring."
        )

    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    return SmokeResult(
        action_keys=tuple(ACTION_KEYS),
        action=tuple(float(v) for v in action),
        chunk_ms=tuple(chunk_ms),
        pop_ms=tuple(pop_ms),
        warmup_ms=warmup_ms,
        rtc_ms=(),
        peak_gpu_gib=peak,
        inversion=None,
    )


# --------------------------------------------------------------------------- #
# The borrowed statistics, applied where they actually take effect
# --------------------------------------------------------------------------- #


def _normalisation_steps(pipeline: Any) -> list[Any]:
    """The steps of ``pipeline`` that normalise, found by what they carry.

    Duck-typed on ``features`` / ``norm_map`` / ``stats`` rather than on a class,
    for the same reason :func:`dk1lab.policy._transform_steps` is: the
    preprocessor holds a ``NormalizerProcessorStep`` and the postprocessor an
    ``UnnormalizerProcessorStep``, and both carry the same three attributes.
    """
    return [
        step
        for step in getattr(pipeline, "steps", ())
        if all(hasattr(step, name) for name in ("features", "norm_map", "stats"))
    ]


def apply_norm_stats(preprocessor: Any, postprocessor: Any, stats: NormStats) -> int:
    """Give the loaded pipelines this cell's statistics. Returns how many steps changed.

    **Why this is not done through the config.** Every rollout loads the policy
    from a path, and ``build_rollout_context`` rebuilds both pipelines from the
    checkpoint's saved JSON — where π0.5's normalizer covers ``features: {}`` and
    has no statistics at all. It does pass a ``dataset_stats`` argument through,
    but only when a LeRobot dataset is attached to the rollout, which is not how
    this cell runs one. So the loaded step objects are patched instead, in place,
    exactly as :func:`dk1lab.policy.apply_gripper_inversion` patches the gripper.

    Without this π0.5 is handed raw radians where it expects roughly [-1, 1] and
    its actions come back in the same wrong units. That failure is silent: the
    policy moves confidently to the wrong places.

    Raises:
        Pi05Error: if either pipeline has no normalisation step to patch, or more
            than one. Both mean the pipeline is not the shape this was written
            against, and continuing would deploy an unnormalised policy.
    """
    import torch

    features = _norm_features()
    patched = 0
    for what, pipeline in (("preprocessor", preprocessor), ("postprocessor", postprocessor)):
        steps = _normalisation_steps(pipeline)
        if len(steps) != 1:
            raise Pi05Error(
                f"expected exactly one normalisation step in the {what}, found "
                f"{len(steps)}. π0.5's borrowed statistics could not be applied, and "
                f"this checkpoint would be handed raw radians — see "
                f"dk1lab/pi05.py:apply_norm_stats."
            )
        step = steps[0]
        step.features = dict(features)
        step.stats = dict(stats.stats)
        # __post_init__ is what turns `features`/`stats` into the tensors the step
        # actually reads; assigning the fields alone changes nothing it looks at.
        step.__post_init__()
        device = getattr(step, "device", None)
        if device is not None:
            step.to(torch.device(device)) if hasattr(step, "to") else None
        patched += 1

    logger.info("%s applied to %d pipeline steps", stats.describe(), patched)
    return patched
