"""Run MolmoAct2 on the DK1 — the Phase 3 deployment path.

One module, three escalating entry points, each of which does strictly more than
the one before it:

``smoke``    GPU only. Loads the checkpoint, builds the real pre/post-processor
             pipelines, and runs inference on a synthetic observation. No robot
             is opened, nothing is connected, and no ``/dev`` node is touched.
``dryrun``   The full deployment path with the arms attached: cameras, robot
             state, processors, inference, action decoding — and then the action
             is **printed instead of sent**. ``send_action`` is never called.
``run``      The real thing: LeRobot's ``BaseStrategy`` control loop, driving the
             followers through :class:`dk1lab.robot.SafeBiDK1Follower`.

All three go through the same config builders, so what the dry run exercises is
what the rollout runs.

Three decisions are made here rather than left to a command line:

**The gripper channel is inverted, always.** The DK1 is 0 = open, 1 = closed;
the checkpoint is 1 = open, 0 = closed. That is not an experiment to opt into —
without it the policy closes the gripper whenever it means to open it. See
:mod:`dk1lab.layout` for the two independent sources.

**The inversion is applied to the loaded pipeline objects**, not through the
policy config, because through the policy config it does not work. See
:func:`apply_gripper_inversion`.

**The speed limit is on.** Teleoperation runs uncapped because a human hand is
on the leader; a policy is exactly the case the limiter was written for, so
rollout reads ``[limits.policy]`` and :data:`POLICY_LIMITS` caps it if the file
says nothing. ``--no-limit`` exists, and is a thing to do knowingly.

Stopping never moves the arms: ``return_to_initial_position`` is forced to
``False`` unless explicitly asked for. LeRobot defaults it to ``True``, which
means pressing stop sweeps both arms back to their startup pose over ~3 s — the
exact opposite of what stopping is for.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from threading import Event
from typing import Any

from .checkpoint import resolve
from .config import DK1Config, LimitProfile
from .layout import (
    ACTION_KEYS,
    CAMERA_NAMES,
    DOF,
    GRIPPER_INDICES,
    IMAGE_KEYS,
    STATE_KEYS,
    yam_joint_offsets,
    yam_joint_signs,
)
from .robot import SafeBiDK1FollowerConfig
from .teleop import follower_config as _follower_config

logger = logging.getLogger(__name__)

#: The speed limit rollout runs under when ``dk1.toml`` says nothing.
#:
#: **Capped**, unlike teleoperation. 0.3 rad/s is about 17 deg/s: slow enough to
#: watch a mistake develop and reach the e-stop. The gripper is not slowed to
#: match — a gripper that takes three seconds to close fails every grasp, and it
#: is not the hazard here.
POLICY_LIMITS = LimitProfile(
    max_joint_rate=0.3,
    max_gripper_rate=1.0,
    max_lag=0.1,
    max_dt=0.1,
)

#: MolmoAct2 BimanualYAM was trained at 30 Hz with a 30-step chunk. This is a
#: property of the checkpoint, not a preference.
DEFAULT_FPS: int = 30

#: How many actions of each chunk RTC executes before it needs the next one.
#: Inference was measured at ~172 ms in the previous iteration of this project,
#: which is ~5 control periods at 30 Hz — 10 leaves margin. Unverified here.
DEFAULT_EXECUTION_HORIZON: int = 10

#: Registered name of the rate-limited follower, for the log line and the report.
ROBOT_TYPE = "bi_dk1_follower_safe"


# --------------------------------------------------------------------------- #
# Configs
# --------------------------------------------------------------------------- #


def policy_config(
    checkpoint: str,
    *,
    device: str = "cuda",
    dtype: str = "bfloat16",
    use_amp: bool = True,
) -> Any:
    """The MolmoAct2 policy config, with this cell's deployment settings applied.

    Args:
        checkpoint: local directory or Hugging Face repo id. ``~`` is expanded.
        device: overridden explicitly because the converted bf16 checkpoint's
            ``config.json`` says ``"device": "cpu"``, and left alone that loads
            7B parameters onto the CPU and runs there — slowly, silently, and
            correctly enough that nothing complains.
        dtype: ``bfloat16`` matches the converted checkpoint; ``float32`` costs a
            long cast at every process start.
        use_amp: autocast during inference. The sync engine only autocasts when
            this is set.

    Note:
        ``joint_signs`` / ``joint_offsets`` are set here too, but they are *not*
        what makes the gripper inversion happen — see
        :func:`apply_gripper_inversion`. They are set so that anything which does
        rebuild the pipelines from this config (training, a future LeRobot that
        reconciles them) gets the same answer, and so that the value is visible
        where someone would look for it.
    """
    # Importing the policy package is what registers "molmoact2" as a
    # PreTrainedConfig subclass; without it from_pretrained reports an empty
    # registry. Same shape of problem as bi_dk1_follower_safe in dk1lab.robot.
    import lerobot.policies  # noqa: F401
    from lerobot.configs import PreTrainedConfig

    path = resolve(checkpoint)
    config = PreTrainedConfig.from_pretrained(path)
    config.pretrained_path = path

    config.device = device
    config.model_dtype = dtype
    config.use_amp = use_amp
    # Continuous actions: the flow-matching head rather than the discrete FAST
    # tokenizer. RTC requires it, and it is what the checkpoint was evaluated on.
    config.inference_action_mode = "continuous"
    config.image_keys = list(IMAGE_KEYS)
    config.joint_signs = yam_joint_signs()
    config.joint_offsets = yam_joint_offsets()
    return config


def follower_config(
    config: DK1Config,
    *,
    limits: LimitProfile | None = None,
    control_mode: str = "impedance",
    profile: str = "policy",
) -> SafeBiDK1FollowerConfig:
    """The rate-limited follower, at the capture profile the policy was trained on.

    Deliberately the same builder teleoperation uses: the camera names, the
    limiter wiring and the port lookup cannot drift between the two paths,
    because there is only one of each.
    """
    limits = limits if limits is not None else config.limit("policy", POLICY_LIMITS)
    return _follower_config(
        config,
        cameras=True,
        profile=profile,
        control_mode=control_mode,
        limits=limits,
    )


def rollout_config(
    config: DK1Config,
    *,
    checkpoint: str,
    task: str,
    limits: LimitProfile | None = None,
    control_mode: str = "impedance",
    fps: int = DEFAULT_FPS,
    duration_s: float = 0.0,
    interpolation: int = 1,
    rtc: bool = False,
    execution_horizon: int = DEFAULT_EXECUTION_HORIZON,
    device: str = "cuda",
    dtype: str = "bfloat16",
    display: bool = False,
    return_home: bool = False,
) -> Any:
    """Assemble LeRobot's :class:`RolloutConfig` for this cell.

    Args:
        task: the language instruction. MolmoAct2 is conditioned on it; there is
            no sensible default, so callers must pass one.
        duration_s: ``0`` means run until interrupted.
        interpolation: split each policy action into N commands and run the loop
            N times faster. The limiter already ramps between commands, so this
            is about loop granularity rather than smoothing; the cameras only
            deliver 30 fps, so large values buy little.
        rtc: real-time chunking — inference runs in a background thread instead
            of blocking the control loop. The right choice for a 7B model at
            30 Hz, and it needs ``inference_action_mode='continuous'``.
        return_home: sweep the arms back to their startup pose at shutdown.
            **Off**, because stopping is what you do when something is wrong.
    """
    from lerobot.rollout import RolloutConfig
    from lerobot.rollout.configs import BaseStrategyConfig
    from lerobot.rollout.inference import RTCInferenceConfig, SyncInferenceConfig

    inference = SyncInferenceConfig()
    if rtc:
        inference = RTCInferenceConfig()
        inference.rtc.execution_horizon = execution_horizon

    return RolloutConfig(
        robot=follower_config(config, limits=limits, control_mode=control_mode),
        policy=policy_config(checkpoint, device=device, dtype=dtype),
        strategy=BaseStrategyConfig(),
        inference=inference,
        fps=fps,
        duration=duration_s,
        interpolation_multiplier=interpolation,
        task=task,
        device=device,
        display_data=display,
        return_to_initial_position=return_home,
    )


# --------------------------------------------------------------------------- #
# The gripper inversion, applied where it actually takes effect
# --------------------------------------------------------------------------- #


class InversionError(RuntimeError):
    """Raised when the gripper inversion could not be applied."""


@dataclass(frozen=True)
class Inversion:
    """What :func:`apply_gripper_inversion` did, for reporting."""

    steps: tuple[str, ...]
    signs: tuple[float, ...]
    offsets: tuple[float, ...]

    def describe(self) -> str:
        channels = ", ".join(ACTION_KEYS[i] for i in GRIPPER_INDICES)
        return f"inverted {channels} in {len(self.steps)} pipeline steps"


def _transform_steps(pipeline: Any) -> list[Any]:
    """The steps of ``pipeline`` that carry a joint sign/offset transform."""
    return [
        step
        for step in getattr(pipeline, "steps", ())
        if hasattr(step, "joint_signs") and hasattr(step, "joint_offsets")
    ]


def apply_gripper_inversion(preprocessor: Any, postprocessor: Any) -> Inversion:
    """Turn on the gripper inversion, on the loaded pipeline objects.

    **Why this is not done through the policy config.** When a policy is loaded
    from a path — which is every rollout — ``make_pre_post_processors`` rebuilds
    both pipelines from the checkpoint's saved ``policy_preprocessor.json`` and
    ``policy_postprocessor.json``. The policy config's ``joint_signs`` and
    ``joint_offsets`` are read only on the *other* branch, the one that builds
    processors from scratch. So ``--policy.joint_signs=...`` on a
    ``lerobot-rollout`` command line parses, validates, is stored on the config,
    and then has no effect whatsoever — and the BimanualYAM checkpoint ships both
    pipelines with ``joint_signs: null``.

    That failure is silent and symmetric: the policy simply opens the gripper
    when it meant to close it. Hence this function, and hence it raising rather
    than warning.

    MolmoAct2 applies the transform in both directions, which is what makes one
    sign/offset pair enough::

        state_model = signs * state_robot + offsets
        action_robot = signs * (action_model - offsets)

    With ``sign = -1, offset = 1`` on the two gripper channels, both are
    ``x -> 1 - x``.

    Args:
        preprocessor: the policy's input pipeline (robot state → model).
        postprocessor: the policy's output pipeline (model action → robot).

    Returns:
        An :class:`Inversion` naming the steps that were patched.

    Raises:
        InversionError: if either pipeline has no transform step to patch, or
            more than one. Both mean the pipeline is not the shape this was
            written against, and continuing would deploy an uninverted gripper.
    """
    signs, offsets = yam_joint_signs(), yam_joint_offsets()
    patched: list[str] = []

    for what, pipeline in (("preprocessor", preprocessor), ("postprocessor", postprocessor)):
        steps = _transform_steps(pipeline)
        if len(steps) != 1:
            raise InversionError(
                f"expected exactly one joint-transform step in the {what}, found "
                f"{len(steps)}. The gripper inversion could not be applied, and this "
                f"checkpoint would drive the grippers backwards — see "
                f"dk1lab/policy.py:apply_gripper_inversion."
            )
        step = steps[0]
        step.joint_signs = list(signs)
        step.joint_offsets = list(offsets)
        patched.append(type(step).__name__)

    logger.info("gripper inversion applied to %s", ", ".join(patched))
    return Inversion(tuple(patched), tuple(signs), tuple(offsets))


# --------------------------------------------------------------------------- #
# Building the live context
# --------------------------------------------------------------------------- #


def build_context(cfg: Any, shutdown_event: Event | None = None) -> tuple[Any, Inversion]:
    """Load the policy, connect the robot, and switch the gripper inversion on.

    **Connects the arms.** ``build_rollout_context`` loads the policy first and
    the hardware last, so a bad checkpoint path fails before anything is
    energised — but by the time this returns, the followers are live and holding
    position.

    Returns:
        The rollout context and the applied inversion.
    """
    from lerobot.rollout import build_rollout_context

    ctx = build_rollout_context(cfg, shutdown_event or Event())
    # Before any strategy runs, and therefore before the RTC thread starts: the
    # inference engine holds references to these pipeline objects, so patching
    # the steps in place reaches it.
    inversion = apply_gripper_inversion(ctx.policy.preprocessor, ctx.policy.postprocessor)
    return ctx, inversion


# --------------------------------------------------------------------------- #
# 1. Smoke test — GPU only, no robot
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SmokeResult:
    """What the GPU-only check found."""

    action_keys: tuple[str, ...]
    action: tuple[float, ...]
    latencies_ms: tuple[float, ...]
    peak_gpu_gib: float
    inversion: Inversion

    @property
    def steady_ms(self) -> float:
        """Median latency after warmup, which is the number that matters."""
        tail = sorted(self.latencies_ms[1:]) or list(self.latencies_ms)
        return tail[len(tail) // 2]


def smoke(
    checkpoint: str,
    *,
    task: str,
    steps: int = 5,
    device: str = "cuda",
    dtype: str = "bfloat16",
    width: int = 640,
    height: int = 360,
) -> SmokeResult:
    """Run the deployment inference path on a synthetic observation. No robot.

    This is the first step of Phase 3 and the only one that can be run with the
    cell powered down. It checks the things that are cheap to get wrong: that the
    checkpoint loads at all, that the processor pipelines build, that the
    inversion can be applied, that inference returns a 14-D action in this cell's
    key order, and how long a call takes.

    What it cannot tell you is whether the actions are any *good* — the images
    are noise. A plausible-looking action vector here means the plumbing works,
    nothing more.

    Args:
        steps: inference calls. The first pays warmup and CUDA-graph capture and
            is excluded from the reported latency.
        width, height: synthetic image size, defaulting to the ``[capture.policy]``
            resolution so the resize path matches deployment.
    """
    import numpy as np
    import torch
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from lerobot.rollout.inference.sync import SyncInferenceEngine
    from lerobot.utils.constants import ACTION, OBS_STR
    from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features

    config = policy_config(checkpoint, device=device, dtype=dtype)
    logger.info("loading %s ...", config.pretrained_path)
    policy = get_policy_class(config.type).from_pretrained(
        config.pretrained_path, config=config
    )
    policy = policy.to(device)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=config.pretrained_path,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    inversion = apply_gripper_inversion(preprocessor, postprocessor)

    # The same feature plumbing the rollout builds, from the same layout — so a
    # key-order mistake shows up here rather than on the arms.
    observation_hw: dict[str, Any] = {key: float for key in STATE_KEYS}
    observation_hw.update({name: (height, width, 3) for name in CAMERA_NAMES})
    features = hw_to_dataset_features(observation_hw, OBS_STR, use_video=False)
    features.update(hw_to_dataset_features({key: float for key in ACTION_KEYS}, ACTION))

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

    latencies: list[float] = []
    action = None
    for _ in range(max(1, steps)):
        start = time.perf_counter()
        action = engine.get_action(frame)
        latencies.append((time.perf_counter() - start) * 1000.0)

    if action is None or len(action) != DOF:
        raise RuntimeError(
            f"expected a {DOF}-D action, got {None if action is None else len(action)}"
        )

    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    return SmokeResult(
        action_keys=tuple(ACTION_KEYS),
        action=tuple(float(v) for v in action),
        latencies_ms=tuple(latencies),
        peak_gpu_gib=peak,
        inversion=inversion,
    )


# --------------------------------------------------------------------------- #
# 2. Dry run — everything except send_action
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DryRunStep:
    """One dry-run tick: where the arms are, and where the policy wants them."""

    index: int
    measured: dict[str, float]
    commanded: dict[str, float]

    @property
    def deltas(self) -> dict[str, float]:
        return {key: self.commanded[key] - self.measured[key] for key in self.commanded}

    @property
    def worst(self) -> tuple[str, float]:
        """The joint the policy disagrees with most, and by how much."""
        key = max(self.deltas, key=lambda k: abs(self.deltas[k]))
        return key, self.deltas[key]


def dryrun(
    cfg: Any,
    *,
    steps: int = 10,
    on_step: Any = None,
) -> list[DryRunStep]:
    """Run the whole deployment path with the arms attached, and send nothing.

    **Energises the arms.** Connecting a DK1 follower is not passive: every motor
    is energised and both grippers self-zero by driving closed until they stall.
    Nothing is ever passed to ``send_action`` — the actions are returned and
    printed — but the arms are live and holding position throughout.

    This is what tells you whether the policy agrees with your start pose. A
    large delta on the first tick means it wants to be somewhere else entirely,
    and the rollout that follows would begin by driving there.

    Args:
        steps: how many observations to run inference on.
        on_step: optional callback per step, for progressive output.
    """
    import torch
    from lerobot.rollout.strategies import BaseStrategy
    from lerobot.utils.constants import OBS_STR
    from lerobot.utils.feature_utils import build_dataset_frame

    ctx, _ = build_context(cfg)
    strategy = BaseStrategy(cfg.strategy)
    collected: list[DryRunStep] = []

    try:
        strategy.setup(ctx)  # builds and starts the inference engine
        engine = ctx.policy.inference
        engine.resume()

        keys = ctx.data.ordered_action_keys
        for index in range(steps):
            observation = ctx.hardware.robot_wrapper.get_observation()
            processed = ctx.processors.robot_observation_processor(observation)
            frame = build_dataset_frame(ctx.data.dataset_features, processed, prefix=OBS_STR)

            action = engine.get_action(frame)
            if action is None:
                # RTC serves from a background thread and may not have a chunk
                # ready on the first ticks.
                continue

            step = DryRunStep(
                index=index,
                measured={key: float(observation[key]) for key in keys},
                commanded={key: float(value) for key, value in zip(keys, action, strict=True)},
            )
            collected.append(step)
            if on_step is not None:
                on_step(step)
    finally:
        # Disconnects. cfg.return_to_initial_position is False, so nothing moves.
        strategy.teardown(ctx)
        if torch.cuda.is_available():
            logger.info("peak GPU memory: %.1f GiB", torch.cuda.max_memory_allocated() / 2**30)

    return collected


# --------------------------------------------------------------------------- #
# 3. The rollout itself
# --------------------------------------------------------------------------- #


def run(cfg: Any, *, display: bool = False) -> None:
    """Drive the followers with the policy. **Moves the arms.**

    LeRobot's ``BaseStrategy`` control loop, unmodified — the same loop
    teleoperation and recording use. Ctrl-C sets the shutdown event and the loop
    leaves; teardown disconnects without moving anything, because
    ``return_to_initial_position`` is off.
    """
    from lerobot.rollout.strategies import BaseStrategy
    from lerobot.utils.process import ProcessSignalHandler
    from lerobot.utils.visualization_utils import init_visualization, shutdown_visualization

    handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    shutdown_event = handler.shutdown_event

    if display:
        init_visualization("rerun", session_name="dk1-policy")

    ctx, _ = build_context(cfg, shutdown_event)
    strategy = BaseStrategy(cfg.strategy)
    try:
        strategy.setup(ctx)
        strategy.run(ctx)
    except KeyboardInterrupt:
        shutdown_event.set()
    finally:
        strategy.teardown(ctx)
        if display:
            shutdown_visualization("rerun")
