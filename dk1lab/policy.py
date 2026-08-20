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

**The gripper inversion is a flag, and it is OFF by default.** The DK1 is
0 = open, 1 = closed; the checkpoint is 1 = open, 0 = closed, and
:mod:`dk1lab.layout` gives two independent sources for that. It was on by
default until the first rollouts, and it is now opt-in (``--invert-gripper``)
because the argument for it, however good, is still an argument: no run has yet
been watched to open or close a gripper on this cell, so the sign is a
hypothesis to test rather than a setting to bake in. Run it both ways and watch
the grippers; :mod:`dk1lab.trace` records the policy's own gripper channel
either way, which is the measurement that settles it.

**When it is on, the inversion is applied to the loaded pipeline objects**, not
through the policy config, because through the policy config it does not work.
See :func:`apply_gripper_inversion`.

**The speed limit is on.** Teleoperation runs uncapped because a human hand is
on the leader; a policy is exactly the case the limiter was written for, so
rollout reads ``[limits.policy]`` and :data:`POLICY_LIMITS` caps it if the file
says nothing. ``--no-limit`` exists, and is a thing to do knowingly.

**Stopping never moves the arms unless homing was asked for.**
``return_to_initial_position`` is forced to ``False`` — always, whatever
``--home`` says. LeRobot defaults it to ``True``, which means every exit path
including a crash sweeps both arms toward their startup pose for a fixed 3 s and
disconnects whether or not they arrived. ``--home`` is the deliberate version of
the same idea and is handled here instead: it runs on a clean end only, targets
the pose in ``[home]``, ramps at the cap the run drove under, and stops when the
arms actually arrive. See :mod:`dk1lab.home`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from threading import Event
from typing import Any

from .checkpoint import resolve
from .config import DK1Config, LimitProfile
from .home import (
    DEFAULT_HOME_RATE,
    HomeError,
    HomeReport,
    go_home,
    interrupt_aborts,
)
from .home import validate_target as validate_home_target
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
#: **Capped**, unlike teleoperation, but no longer timid. Both numbers were
#: raised on 2026-08-20 from measurement rather than from caution:
#:
#: ``max_joint_rate`` 0.3 -> 1.0 rad/s. A recorded 120 s sim episode of this
#: checkpoint demands a median of 0.036 rad/s but a p95 of 0.31 and peaks of
#: 4.56 — bursty, not fast. Replayed through :class:`~dk1lab.limiter.SlewLimiter`,
#: a 0.3 cap leaves the worst joint 0.98 rad (56 deg) behind the policy's intent
#: on 26% of ticks; 1.0 cuts that to 0.40 rad on 3.3%; 2.0 is transparent.
#:
#: ``max_lag`` 0.1 -> 0.4 rad. This is a **torque** limit wearing a position
#: clamp's costume: impedance torque is ``arm_kp * (q_des - q)`` with
#: ``arm_kp = [100, 100, 100, 20, 20, 10]``, so a 0.1 rad lead cap held the PD
#: torque to 10 Nm on j1-j3 (motor limit 28) and 1 Nm on j6 (limit 10). Since
#: the limiter stores the *clamped* value as the new previous command, a joint
#: that cannot break stiction inside 0.1 rad stalls there silently forever —
#: the very deadlock :mod:`dk1lab.limiter` was designed to avoid.
#:
#: The gripper is not slowed to match — a gripper that takes three seconds to
#: close fails every grasp, and it is not the hazard here.
POLICY_LIMITS = LimitProfile(
    max_joint_rate=1.0,
    max_gripper_rate=1.0,
    max_lag=0.4,
    max_dt=0.1,
)

#: MolmoAct2 BimanualYAM was trained at 30 Hz with a 30-step chunk. This is a
#: property of the checkpoint, not a preference.
DEFAULT_FPS: int = 30

#: How many actions of each chunk RTC blends against the previous one.
#:
#: **This must be strictly greater than the inference delay in ticks**, and the
#: old value of 10 was not. RTC builds its prefix weights as
#: ``get_prefix_weights(delay, execution_horizon, chunk)``: ones for the first
#: ``delay`` steps (already committed while the GPU was thinking), then a linear
#: ramp down to zero at ``execution_horizon``. When ``delay >= execution_horizon``
#: the ramp has zero width and the schedule collapses to a hard binary mask —
#: the first steps pinned rigidly to the previous chunk, the rest unconstrained,
#: with a discontinuity at the seam. That is the judder.
#:
#: Measured on this machine (RTX 5090, bf16, the RTC code path, 3x 640x360):
#: 324 ms per chunk, 272 ms with :func:`freeze_for_inference`. At 30 Hz that is
#: a delay of 9-10 ticks, so the old default of 10 was exactly degenerate.
#:
#: 20 leaves a 10-step ramp and matches the steady-state leftover length
#: (``chunk - delay`` = 30 - 10 = 20), so the previous chunk is used whole and
#: never zero-padded. Raising it to 30 is worse, not better: every step gets
#: weight 1.0, the new chunk is pinned to the old one for its whole length, and
#: the policy stops reacting to what it sees.
#:
#: The ~172 ms this used to cite was the *non-RTC* path. RTC costs about twice
#: that, because its denoise step runs under ``torch.enable_grad``.
DEFAULT_EXECUTION_HORIZON: int = 20

#: How many steps of linear blend RTC needs between the inference delay and the
#: execution horizon for the seam between two chunks to be smooth. A one- or
#: two-step ramp is a step function with a rounded corner, so "delay < horizon"
#: is not a sufficient test on its own.
MIN_RTC_BLEND_STEPS: int = 5

#: Inference latency measured on this machine for the RTC path, seconds. Used
#: only to sanity-check the execution horizon before a run; the real number is
#: measured at startup by :func:`prewarm`.
MEASURED_RTC_LATENCY_S: float = 0.27

#: Registered name of the rate-limited follower, for the log line and the report.
ROBOT_TYPE = "bi_dk1_follower_safe"

#: Pass as ``run(home=...)`` to home to the pose captured at connect rather than
#: to a configured one. A sentinel rather than ``True`` so that "home to the start
#: pose" and "home to this pose" are the same parameter and cannot both be given.
HOME_AT_START_POSE = "start-pose"


# --------------------------------------------------------------------------- #
# Configs
# --------------------------------------------------------------------------- #


def policy_config(
    checkpoint: str,
    *,
    device: str = "cuda",
    dtype: str = "bfloat16",
    use_amp: bool = True,
    invert_gripper: bool = False,
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
    if invert_gripper:
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
    invert_gripper: bool = False,
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
        invert_gripper: whether this run inverts the two gripper channels. Set on
            the config so that ``_report`` and the banner can state what the run
            will actually do; the effect itself comes from
            :func:`apply_gripper_inversion` at :func:`build_context`.
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
        policy=policy_config(
            checkpoint, device=device, dtype=dtype, invert_gripper=invert_gripper
        ),
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


def build_context(
    cfg: Any,
    shutdown_event: Event | None = None,
    *,
    prewarm_engine: bool = True,
    invert_gripper: bool = False,
) -> tuple[Any, Inversion | None]:
    """Load the policy, connect the robot, and switch the gripper inversion on.

    **Connects the arms.** ``build_rollout_context`` loads the policy first and
    the hardware last, so a bad checkpoint path fails before anything is
    energised — but by the time this returns, the followers are live and holding
    position.

    Also does the two things that have to happen between loading the policy and
    starting the control loop: :func:`freeze_for_inference` and :func:`prewarm`.
    Both exist because of how RTC behaves, and both are no-ops for correctness —
    they change only speed, and what RTC believes about speed.

    Args:
        prewarm_engine: run one inference before returning, and report the
            resulting RTC headroom. Off only for tests.
        invert_gripper: apply the gripper inversion. **Off by default** — see
            :func:`apply_gripper_inversion` for what it does and
            :mod:`dk1lab.layout` for the argument that it is the right thing to
            do; it is a flag rather than a fact because nothing has yet watched
            the policy open or close a gripper on this cell.

    Returns:
        The rollout context, and the applied inversion or ``None`` if it was off.
    """
    from lerobot.rollout import build_rollout_context

    ctx = build_rollout_context(cfg, shutdown_event or Event())
    # Before any strategy runs, and therefore before the RTC thread starts: the
    # inference engine holds references to these pipeline objects, so patching
    # the steps in place reaches it.
    inversion = (
        apply_gripper_inversion(ctx.policy.preprocessor, ctx.policy.postprocessor)
        if invert_gripper
        else None
    )
    freeze_for_inference(ctx.policy.policy)
    if prewarm_engine:
        latency = prewarm(ctx)
        report_rtc_headroom(latency, fps=cfg.fps, execution_horizon=_execution_horizon(cfg))
    return ctx, inversion


# --------------------------------------------------------------------------- #
# Making RTC behave: the two startup steps that were missing
# --------------------------------------------------------------------------- #


def freeze_for_inference(policy: Any) -> int:
    """``requires_grad_(False)`` on every parameter. Returns how many changed.

    Nothing here trains, so this ought to be a no-op — and for the sync engine it
    is. It is not a no-op for RTC, because RTC's guidance step runs the action
    expert under ``torch.enable_grad()`` (``policies/rtc/modeling_rtc.py``,
    ``denoise_step``). With the parameters still flagged trainable, each of the
    eight flow steps builds a full autograd graph over the action expert and
    keeps the activations alive.

    That graph is never used. ``denoise_step`` computes ``v_t`` *before* it calls
    ``x_t.requires_grad_(True)``, so the only path ``torch.autograd.grad`` can
    follow is ``x1_t = x_t - time * v_t`` — an identity in ``x_t``. The
    correction it recovers is exactly the ``grad_outputs`` it passed in. The
    graph over the weights is pure overhead.

    Measured here: 324 ms -> 272 ms per chunk, with the produced action chunk
    **bit-identical** under a fixed generator seed (max abs difference 0.0).
    """
    changed = sum(1 for p in policy.parameters() if p.requires_grad)
    policy.requires_grad_(False)
    logger.info("froze %d parameter tensors for inference", changed)
    return changed


def prewarm(ctx: Any) -> float:
    """Run one full inference before the control loop starts. Returns its seconds.

    **Why this is not just impatience.** RTC decides how much of each new chunk
    to pin to the previous one from ``latency_tracker.max()``, and
    :class:`~lerobot.policies.rtc.LatencyTracker` keeps a running maximum that
    never decays. The exclusion for warmup samples is guarded by
    ``use_torch_compile``, which is ``False`` here — so the first call, which
    pays model warmup and CUDA-graph capture, is fed straight into the tracker
    and sets the delay for the rest of the run. Measured here that first call is
    511 ms against a steady state of 324: a delay of 16 ticks instead of 10,
    permanently, from one sample taken before the model was warm.

    Doing it here, on the real observation, walks exactly the path the RTC thread
    walks — robot observation, observation processor, dataset frame, policy
    preprocessor, ``predict_action_chunk`` — so the caches it fills are the ones
    that matter.

    Reads the robot. Sends nothing, and does not touch the action queue:
    ``predict_action_chunk`` has no queue side effect, unlike ``select_action``.
    """
    import torch
    from lerobot.policies.utils import prepare_observation_for_inference
    from lerobot.utils.constants import OBS_STR
    from lerobot.utils.feature_utils import build_dataset_frame

    observation = ctx.hardware.robot_wrapper.get_observation()
    processed = ctx.processors.robot_observation_processor(observation)
    frame = build_dataset_frame(ctx.data.hw_features, processed, prefix=OBS_STR)

    task = ctx.runtime.cfg.task
    device = torch.device(ctx.runtime.cfg.device or "cpu")
    batch = prepare_observation_for_inference(
        frame, device, task, ctx.hardware.robot_wrapper.robot_type
    )
    batch["task"] = [task]

    start = time.perf_counter()
    ctx.policy.policy.predict_action_chunk(
        ctx.policy.preprocessor(batch), inference_delay=0, prev_chunk_left_over=None
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    logger.info("prewarm inference: %.0f ms (kept out of the RTC latency tracker)", elapsed * 1000)
    return elapsed


def _execution_horizon(cfg: Any) -> int | None:
    """The configured RTC execution horizon, or ``None`` when not running RTC."""
    rtc = getattr(getattr(cfg, "inference", None), "rtc", None)
    return getattr(rtc, "execution_horizon", None)


def rtc_headroom(
    latency_s: float,
    *,
    fps: float,
    execution_horizon: int,
    min_blend: int = MIN_RTC_BLEND_STEPS,
) -> tuple[int, bool]:
    """The inference delay in ticks, and whether RTC's prefix ramp survives it.

    RTC weights the first ``delay`` steps of a new chunk at 1.0 — those describe
    time that passed while the GPU was thinking — then ramps linearly to 0 at
    ``execution_horizon``. The ramp is what makes consecutive chunks agree at the
    seam. When ``delay >= execution_horizon`` the ramp is empty and the schedule
    degenerates into a step function, which is the shape that judders; a ramp of
    one or two steps is that same step function with a rounded corner, so the
    test is against ``min_blend`` rather than against zero.
    """
    delay = int(-(-latency_s * fps // 1))  # ceil, matching the RTC thread
    return delay, (execution_horizon - delay) >= min_blend


def report_rtc_headroom(latency_s: float, *, fps: float, execution_horizon: int | None) -> None:
    """Log the delay-vs-horizon relationship, loudly when there is no room."""
    if execution_horizon is None:
        return
    delay, ok = rtc_headroom(latency_s, fps=fps, execution_horizon=execution_horizon)
    blend = execution_horizon - delay
    if ok:
        logger.info(
            "RTC: %.0f ms inference = %d ticks delay, ramping to 0 at %d — %d steps of blend",
            latency_s * 1000, delay, execution_horizon, blend,
        )
        return
    logger.warning(
        "RTC prefix blending has no room: %.0f ms inference is %d ticks at %g Hz against an "
        "--execution-horizon of %d, leaving a %d-step blend. The weights collapse towards a "
        "step function, consecutive chunks meet with a discontinuity, and the arms judder. "
        "Raise --execution-horizon to at least %d (and keep it below the chunk size, 30).",
        latency_s * 1000, delay, fps, execution_horizon, blend, delay + MIN_RTC_BLEND_STEPS,
    )


def dataset_features(*, width: int, height: int) -> dict:
    """The LeRobot feature dict this cell presents to the policy.

    Derived from :mod:`dk1lab.layout` rather than restated, so the 14 state and
    action slots and the three camera names cannot drift between the paths that
    build one of these — the smoke test, and the ``/act`` server in
    :mod:`dk1lab.serve`. The rollout gets its own from the connected robot, and
    ``tests/test_layout.py`` asserts the two agree.

    The image ``shape`` is nominal: ``build_dataset_frame`` passes image arrays
    through untouched, so a frame of a different size still works. It is set
    from ``[capture.policy]`` anyway, because a wrong number here would be read
    as a claim.
    """
    from lerobot.utils.constants import ACTION, OBS_STR
    from lerobot.utils.feature_utils import hw_to_dataset_features

    observation_hw: dict[str, Any] = {key: float for key in STATE_KEYS}
    observation_hw.update({name: (height, width, 3) for name in CAMERA_NAMES})
    features = hw_to_dataset_features(observation_hw, OBS_STR, use_video=False)
    features.update(hw_to_dataset_features({key: float for key in ACTION_KEYS}, ACTION))
    return features


# --------------------------------------------------------------------------- #
# 1. Smoke test — GPU only, no robot
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SmokeResult:
    """What the GPU-only check found.

    The two latencies are measured separately on purpose. ``select_action``
    serves from a cached 30-step chunk, so all but one call in thirty is a queue
    pop costing a millisecond or two — timing a run of consecutive calls reports
    that, and it is not the number that decides anything.
    """

    action_keys: tuple[str, ...]
    action: tuple[float, ...]
    #: Cost of a call that has to run the model. This is the one that matters.
    chunk_ms: tuple[float, ...]
    #: Cost of a call served from the cached chunk.
    pop_ms: tuple[float, ...]
    #: The very first call, which also pays warmup and CUDA-graph capture.
    warmup_ms: float
    #: Cost of a chunk through the **RTC** code path — the one a rollout runs.
    #: Measured separately because it is roughly twice ``chunk_ms``: RTC's
    #: guidance step runs under ``torch.enable_grad`` and cannot use the CUDA
    #: graph. Reporting only ``chunk_ms`` is what hid the judder.
    rtc_ms: tuple[float, ...]
    peak_gpu_gib: float
    #: ``None`` when the run did not invert the gripper channels.
    inversion: Inversion | None

    @staticmethod
    def _median(values: tuple[float, ...]) -> float:
        ordered = sorted(values)
        return ordered[len(ordered) // 2] if ordered else 0.0

    @property
    def inference_ms(self) -> float:
        """Median cost of an actual forward pass."""
        return self._median(self.chunk_ms)

    @property
    def cached_ms(self) -> float:
        """Median cost of a call served from the chunk already computed."""
        return self._median(self.pop_ms)

    @property
    def rtc_inference_ms(self) -> float:
        """Median cost of one chunk through the RTC path. The deployment number."""
        return self._median(self.rtc_ms)


def smoke(
    checkpoint: str,
    *,
    task: str,
    steps: int = 5,
    device: str = "cuda",
    dtype: str = "bfloat16",
    width: int = 640,
    height: int = 360,
    invert_gripper: bool = False,
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
        steps: how many timed samples of each kind to take. Each chunk sample
            resets the policy first, so it has to run the model rather than
            answer from the chunk it already has.
        width, height: synthetic image size, defaulting to the ``[capture.policy]``
            resolution so the resize path matches deployment.
    """
    import numpy as np
    import torch
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from lerobot.rollout.inference.sync import SyncInferenceEngine
    from lerobot.utils.constants import OBS_STR
    from lerobot.utils.feature_utils import build_dataset_frame

    config = policy_config(
        checkpoint, device=device, dtype=dtype, invert_gripper=invert_gripper
    )
    logger.info("loading %s ...", config.pretrained_path)
    policy = get_policy_class(config.type).from_pretrained(
        config.pretrained_path, config=config
    )
    policy = policy.to(device)
    policy.eval()
    freeze_for_inference(policy)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=config.pretrained_path,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    inversion = apply_gripper_inversion(preprocessor, postprocessor) if invert_gripper else None

    # The same feature plumbing the rollout builds, from the same layout — so a
    # key-order mistake shows up here rather than on the arms.
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

    # The first call pays model warmup and CUDA-graph capture and says nothing
    # about deployment; it is reported separately because it is what you wait
    # through at startup.
    start = time.perf_counter()
    action = engine.get_action(frame)
    warmup_ms = (time.perf_counter() - start) * 1000.0

    chunk_ms: list[float] = []
    pop_ms: list[float] = []
    for _ in range(max(1, steps)):
        # Resetting empties the policy's action queue, so this call must run the
        # model. Without the reset, select_action answers from the 30-step chunk
        # it already holds and the measurement is of a dict lookup.
        engine.reset()
        start = time.perf_counter()
        action = engine.get_action(frame)
        chunk_ms.append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        engine.get_action(frame)
        pop_ms.append((time.perf_counter() - start) * 1000.0)

    if action is None or len(action) != DOF:
        raise RuntimeError(
            f"expected a {DOF}-D action, got {None if action is None else len(action)}"
        )

    rtc_ms = _time_rtc_path(policy, preprocessor, frame, task=task, device=device, steps=steps)

    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    return SmokeResult(
        action_keys=tuple(ACTION_KEYS),
        action=tuple(float(v) for v in action),
        chunk_ms=tuple(chunk_ms),
        pop_ms=tuple(pop_ms),
        warmup_ms=warmup_ms,
        rtc_ms=rtc_ms,
        peak_gpu_gib=peak,
        inversion=inversion,
    )


def _time_rtc_path(
    policy: Any,
    preprocessor: Any,
    frame: dict,
    *,
    task: str,
    device: str,
    steps: int,
) -> tuple[float, ...]:
    """Time ``predict_action_chunk`` the way the RTC thread calls it.

    This is the number that sets the inference delay, and therefore whether RTC's
    prefix blending has any room — see :func:`rtc_headroom`. It is not the same
    as the ``select_action`` cost: the RTC path runs its denoise steps under
    ``torch.enable_grad`` for a guidance term, which rules out the CUDA graph and
    costs roughly twice as much. Measuring only ``select_action`` is exactly how
    a 172 ms figure came to justify an execution horizon of 10 for a 324 ms call.

    Leaves the policy's RTC state as it found it.
    """
    import torch
    from lerobot.policies.rtc.configuration_rtc import RTCConfig
    from lerobot.policies.utils import prepare_observation_for_inference

    previous = policy.config.rtc_config
    policy.config.rtc_config = RTCConfig()
    policy.init_rtc_processor()
    try:
        def chunk(prev: Any, delay: int) -> Any:
            batch = prepare_observation_for_inference(
                dict(frame), torch.device(device), task, ROBOT_TYPE
            )
            batch["task"] = [task]
            return policy.predict_action_chunk(
                preprocessor(batch), inference_delay=delay, prev_chunk_left_over=prev
            )

        # A leftover prefix has to exist for the guidance term to be exercised;
        # without one RTC short-circuits and the measurement is of the cheap path.
        leftover = chunk(None, 0).squeeze(0)[:DEFAULT_EXECUTION_HORIZON].clone()
        timings: list[float] = []
        for _ in range(max(1, steps)):
            start = time.perf_counter()
            out = chunk(leftover, DEFAULT_EXECUTION_HORIZON // 2)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000.0)
            leftover = out.squeeze(0)[:DEFAULT_EXECUTION_HORIZON].clone()
        return tuple(timings)
    finally:
        policy.config.rtc_config = previous
        policy.init_rtc_processor()


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
    invert_gripper: bool = False,
    trace: Any = None,
) -> list[DryRunStep]:
    """Run the whole deployment path with the arms attached, and send nothing.

    **Energises the arms.** Connecting a DK1 follower is not passive: every motor
    is energised and both grippers self-zero by driving open against their stop.
    Nothing is ever passed to ``send_action`` — the actions are returned and
    printed — but the arms are live and holding position throughout.

    This is what tells you whether the policy agrees with your start pose. A
    large delta on the first tick means it wants to be somewhere else entirely,
    and the rollout that follows would begin by driving there.

    Args:
        steps: how many observations to run inference on.
        on_step: optional callback per step, for progressive output.
        invert_gripper: apply the gripper inversion. Off by default.
        trace: an optional :class:`~dk1lab.trace.RolloutTrace`. With one attached
            this is the cheapest way to see the *model's* own action next to the
            one the postprocessor produced, and — with ``display`` set on it —
            the images as the model receives them. Nothing is ever sent, so it is
            also the safe place to check both.
    """
    import torch
    from lerobot.rollout.strategies import BaseStrategy
    from lerobot.utils.constants import OBS_STR
    from lerobot.utils.feature_utils import build_dataset_frame

    ctx, _ = build_context(cfg, invert_gripper=invert_gripper)
    strategy = BaseStrategy(cfg.strategy)
    collected: list[DryRunStep] = []
    if trace is not None:
        trace.attach(ctx)

    try:
        strategy.setup(ctx)  # builds and starts the inference engine
        if trace is not None:
            trace.attach_queue(strategy)
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


def measure_fn(robot_wrapper: Any) -> Any:
    """A positions-only reader for ``robot_wrapper``, avoiding the cameras.

    :meth:`SafeBiDK1Follower.measured_positions` reads the 12 motors and nothing
    else; ``get_observation`` would also grab three camera frames, on every tick
    of a sweep that does not look at them. The fallback keeps this working for any
    robot, and keeps the lock, at the cost of the frames.
    """
    inner = getattr(robot_wrapper, "inner", robot_wrapper)
    if hasattr(inner, "measured_positions"):
        return inner.measured_positions
    return lambda: {
        key: value for key, value in robot_wrapper.get_observation().items() if key.endswith(".pos")
    }


def home_target(ctx: Any, pose: Any | None) -> dict[str, float]:
    """The pose a home sweep should drive to: the configured one, or the start pose.

    ``ctx.hardware.initial_position`` is what LeRobot captured immediately after
    ``connect()``, i.e. wherever the arms were standing when the run began. It is
    the fallback and not the default, because "wherever you left it last time" is
    a pose, not a home — see :class:`dk1lab.config.HomePose`.
    """
    if pose is not None:
        return validate_home_target(pose.as_action_dict())
    initial = getattr(ctx.hardware, "initial_position", None)
    if not initial:
        raise HomeError(
            "no [home] in dk1.toml and no start pose was captured, so there is "
            "nothing to home to. Run `dk1 policy home --capture` first."
        )
    return validate_home_target({key: float(value) for key, value in initial.items()})


def go_home_before_teardown(
    ctx: Any,
    strategy: Any,
    *,
    target: dict[str, float],
    rate: float,
    fps: float,
) -> HomeReport:
    """Stop inference, sweep the arms home, and report. **Moves the arms.**

    Called from :func:`run`'s ``finally`` and *before* ``strategy.teardown``,
    because teardown disconnects and disconnecting de-energises every motor.

    The inference engine is stopped first. Nothing would send its actions once
    the control loop has left, but RTC's background thread would otherwise keep
    running 270 ms forward passes on the GPU while this sweep is trying to hold a
    30 Hz command rate. ``stop()`` is idempotent, so teardown's own call is fine.
    """
    engine = getattr(strategy, "_engine", None)
    if engine is not None:
        engine.stop()

    robot_wrapper = ctx.hardware.robot_wrapper
    with interrupt_aborts() as aborted:
        report = go_home(
            measure=measure_fn(robot_wrapper),
            send=robot_wrapper.send_action,
            target=target,
            rate=rate,
            fps=fps,
            should_abort=aborted,
        )
    logger.info("%s", report.summary())
    if not report.reached:
        logger.warning(
            "the arms did not reach home; disconnecting now disables every motor, "
            "so support anything that is holding itself up"
        )
    return report


def ended_cleanly(error: BaseException | None) -> bool:
    """Whether a finished run earns a home sweep.

    Nothing faulted: the loop hit its duration limit, or the operator stopped it.
    An exception means something went wrong that nobody has looked at yet, and
    commanding more motion into that is the opposite of stopping. Ctrl-C is a
    clean end — the operator asked, and they can ask again during the sweep to
    stop it where the arms are.
    """
    return error is None or isinstance(error, KeyboardInterrupt)


def _home_rate(cfg: Any) -> float:
    """The sweep speed: the same cap this run drove the policy under.

    A home sweep is commanded motion with nobody's hand on a leader arm, so it
    has no business being faster than the policy was allowed to be. When the run
    was uncapped (``--no-limit``) there is no number to inherit and
    :data:`~dk1lab.home.DEFAULT_HOME_RATE` applies — uncapping the policy is a
    deliberate act, letting an automatic shutdown sweep inherit it is not.
    """
    rate = getattr(cfg.robot, "max_joint_rate", None)
    return float(rate) if rate else DEFAULT_HOME_RATE


def run(
    cfg: Any,
    *,
    display: bool = False,
    home: Any | None = None,
    invert_gripper: bool = False,
    trace: Any = None,
) -> HomeReport | None:
    """Drive the followers with the policy. **Moves the arms.**

    LeRobot's ``BaseStrategy`` control loop, unmodified — the same loop
    teleoperation and recording use. Ctrl-C sets the shutdown event and the loop
    leaves.

    Args:
        home: when given, sweep the arms to this pose once the loop has ended —
            on the duration limit *and* on Ctrl-C, but never after an exception,
            because a run that ended in a fault is not one to command more motion
            in. A :class:`~dk1lab.config.HomePose`, or
            :data:`HOME_AT_START_POSE` to use the pose captured at connect. A
            second Ctrl-C during the sweep stops it where the arms are.
        invert_gripper: apply the gripper inversion. Off by default.
        trace: an optional :class:`~dk1lab.trace.RolloutTrace`, attached around
            the engine so the run records per-chunk latency, queue depth and the
            policy's own actions. It never changes what is sent.

    Returns:
        The :class:`~dk1lab.home.HomeReport`, or ``None`` if homing was not asked
        for or the run ended in an exception.
    """
    from lerobot.rollout.strategies import BaseStrategy
    from lerobot.utils.process import ProcessSignalHandler
    from lerobot.utils.visualization_utils import init_visualization, shutdown_visualization

    handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    shutdown_event = handler.shutdown_event

    watching = display or (trace is not None and trace.display)
    if watching:
        init_visualization("rerun", session_name="dk1-policy")

    ctx, _ = build_context(cfg, shutdown_event, invert_gripper=invert_gripper)
    strategy = BaseStrategy(cfg.strategy)
    # After build_context, so prewarm's cold call is not counted as a chunk.
    if trace is not None:
        trace.attach(ctx)
    report: HomeReport | None = None
    ended: BaseException | None = None
    try:
        strategy.setup(ctx)
        # After setup: the RTC action queue does not exist until engine.start().
        if trace is not None:
            trace.attach_queue(strategy)
        strategy.run(ctx)
    except KeyboardInterrupt as exc:
        # Ctrl-C during the loop. LeRobot's signal handler normally absorbs it
        # into the shutdown event, so this is the belt-and-braces path — and it
        # is still a clean end: the operator asked to stop, nothing faulted.
        ended = exc
        shutdown_event.set()
    except BaseException as exc:
        ended = exc
        raise
    finally:
        if home is not None and ended_cleanly(ended):
            try:
                report = go_home_before_teardown(
                    ctx,
                    strategy,
                    target=home_target(ctx, None if home is HOME_AT_START_POSE else home),
                    rate=_home_rate(cfg),
                    fps=cfg.fps,
                )
            except Exception:
                # Never let homing keep the arms connected and energised.
                logger.exception("home sweep failed; disconnecting anyway")
        strategy.teardown(ctx)
        if watching:
            shutdown_visualization("rerun")
    return report
