"""``SimRobot`` — the MuJoCo cell, behind the same interface the real one has.

``STUDY.md``'s design for the simulator is one sentence: *a ``SimRobot``
implementing the same robot interface ``dk1lab/policy.py`` already calls, so only
the robot object swaps and rollout, FIFO engine, limiter and home sweep are
untouched code.* That is what this is. It is a LeRobot :class:`~lerobot.robots.Robot`
like :class:`~dk1lab.robot.SafeBiDK1Follower` is, registered as
``dk1_sim``, and every command that drives the arms drives it instead by naming
that type — nothing in :mod:`dk1lab.policy`, :mod:`dk1lab.fifo`,
:mod:`dk1lab.trace` or :mod:`dk1lab.home` changes or knows about it.

**What it is for, and what it is not.** ``STUDY.md``: *the sim produces no
episodes and no scores. It exists to confirm each policy drives the pipeline
before it drives the arms.* π0.5 has never commanded this cell and MolmoAct2 has
only ever commanded it in one configuration; a rollout that fails here fails
without anything moving in the room. Read a sim rollout as "the pipeline ran",
never as "the policy can do the task".

**The clock is the sim's, not the wall's.** One :meth:`send_action` advances the
model by exactly one control period — :data:`DEFAULT_FPS` steps of
``1/30 s`` worth of physics — so the policy always sees the 30 Hz cadence it was
trained at however long inference takes. ``--realtime`` additionally sleeps to
keep the sim in step with the wall clock, which is what makes the viewer watchable;
``--free-run`` lets it go as fast as the model and the GPU allow, which is what
makes an unattended check quick. Neither changes a single number the policy sees.

**The gripper.** This cell has one normalised gripper channel per arm (0 = open)
and the model has two prismatic finger joints, so :func:`dk1lab.scene.gripper_position`
maps the one onto the two and the measurement is mapped back. That conversion
lives in :mod:`dk1lab.scene` beside the travel it depends on, not here.

**Connecting is passive, and that is the whole point.** The real follower's
``connect()`` energises every motor and drives both grippers open against their
stop; this one builds a model. Every safety notice in the CLI is about the real
one, and a command pointed at this robot says so rather than repeating them.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from lerobot.robots import Robot, RobotConfig

from . import scene as scene_module
from .layout import ACTION_KEYS, ARM_KEYS, ARMS, CAMERA_NAMES, is_gripper
from .limiter import (
    DEFAULT_MAX_DT,
    DEFAULT_MAX_GRIPPER_RATE,
    DEFAULT_MAX_JOINT_RATE,
    DEFAULT_MAX_LAG,
    SlewLimiter,
)

logger = logging.getLogger(__name__)

#: The control rate the policy expects. A property of the checkpoint, matched
#: here so that ``sim dt`` and the loop period are the same number.
DEFAULT_FPS: int = 30

#: Rendered frame size. 16:9, matching ``[capture.policy]``'s aspect ratio,
#: because the model input is a *stretch* of the frame and a 4:3 sim picture
#: would stretch differently than the cell does.
DEFAULT_WIDTH: int = 640
DEFAULT_HEIGHT: int = 360

#: Physics steps per control tick. 10 puts the integrator at 300 Hz, which the
#: DK1's masses and this scene's contacts are comfortable at, and it divides the
#: control period exactly — which the scene's own 2 ms does not.
DEFAULT_SUBSTEPS: int = 10

#: Registered name, for ``--robot.type``.
ROBOT_TYPE = "dk1_sim"


@RobotConfig.register_subclass(ROBOT_TYPE)
@dataclass
class SimRobotConfig(RobotConfig):
    """The MuJoCo cell's configuration.

    Args:
        fps: the control rate. One :meth:`SimRobot.send_action` advances the model
            by one period of this, whatever the wall clock did.
        width, height: rendered camera size.
        realtime: sleep so the sim keeps pace with the wall clock. On for
            watching, off for an unattended check — and it changes nothing the
            policy sees either way.
        view: open ``mujoco.viewer``. Passive, so the control loop keeps driving
            while the window is open.
        objects: put the die and the bowl in the scene. Off is the kinematic
            check ``STUDY.md`` names as the acceptable fallback.
        urdf: this repo's follower URDF, which the scene is generated from.
        substeps: physics steps per control tick. The model's timestep is set to
            ``1 / (fps * substeps)`` so that one tick is *exactly* one control
            period — see :meth:`SimRobot.connect`.
        max_joint_rate, max_gripper_rate, max_lag, max_dt: the same speed limit
            :class:`~dk1lab.robot.SafeBiDK1FollowerConfig` takes, and the same
            :class:`~dk1lab.limiter.SlewLimiter` behind it. The sim carries it
            because ``STUDY.md`` asks for the *pipeline* to be unchanged, and
            because "did the 0.6 rad/s cap bound either policy" is one of the
            questions the study ends on — a sim where the cap was quietly absent
            could not begin to answer it. ``None`` disables it.
        cameras: unused, and present because ``RobotConfig`` consumers ask for it.
    """

    fps: int = DEFAULT_FPS
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    realtime: bool = True
    view: bool = True
    objects: bool = True
    urdf: str | None = None
    substeps: int = DEFAULT_SUBSTEPS
    max_joint_rate: float | None = DEFAULT_MAX_JOINT_RATE
    max_gripper_rate: float = DEFAULT_MAX_GRIPPER_RATE
    max_lag: float = DEFAULT_MAX_LAG
    max_dt: float = DEFAULT_MAX_DT
    cameras: dict[str, Any] = field(default_factory=dict)


class SimRobot(Robot):
    """The bimanual DK1 in MuJoCo, driven exactly as the real one is.

    Implements the interface :mod:`dk1lab.policy` calls — ``connect``,
    ``get_observation``, ``send_action``, ``disconnect``, the two feature
    dictionaries — plus :meth:`measured_positions`, which
    :func:`dk1lab.policy.measure_fn` prefers so a home sweep does not pay for
    three renders on every tick of a sweep that does not look at them.

    The 14-D contract is :mod:`dk1lab.layout`'s, unchanged and underived from
    anything here: :data:`~dk1lab.layout.ACTION_KEYS` in, the same keys out.
    """

    config_class = SimRobotConfig
    name = ROBOT_TYPE

    def __init__(self, config: SimRobotConfig):
        super().__init__(config)
        self.config = config
        self.fps = int(config.fps)
        self._model: Any = None
        self._data: Any = None
        self._renderer: Any = None
        self._viewer: Any = None
        self._connected = False
        #: MuJoCo qpos addresses for the 14 channels, in ACTION_KEYS order.
        self._qpos: list[int] = []
        #: Actuator ids for the 14 channels; the gripper channel carries two.
        self._actuators: list[tuple[int, ...]] = []
        self._next_frame: float = 0.0
        self._steps_per_tick: int = 1
        #: The same limiter the real follower runs, for the same reason: a policy
        #: is what it was written for, and the sim is where a policy is pointed
        #: first.
        #:
        #: **On the sim's clock, not the wall's.** The limiter ramps by
        #: ``rate * dt``, and with ``time.monotonic`` that ``dt`` would be however
        #: long the last tick took in the room — which under ``--free-run`` is
        #: half a millisecond, so a 1.0 rad/s cap would behave like 0.03. The
        #: motion would then depend on how fast the computer is, and "did the
        #: 0.6 rad/s cap bound either policy" would be unanswerable. ``mjData.time``
        #: advances by exactly one control period per tick, whatever the wall did.
        self.limiter = SlewLimiter(
            max_joint_rate=config.max_joint_rate,
            max_gripper_rate=config.max_gripper_rate,
            max_lag=config.max_lag,
            max_dt=config.max_dt,
            clock=self._sim_time,
        )

    def _sim_time(self) -> float:
        """Simulated seconds since connect. The limiter's clock — see above."""
        return float(self._data.time) if self._data is not None else 0.0

    # ------------------------------------------------------------------ #
    # The interface every consumer reads
    # ------------------------------------------------------------------ #

    @property
    def observation_features(self) -> dict:
        """14 joint positions and three images — the same shape the cell reports."""
        features: dict[str, Any] = dict.fromkeys(ACTION_KEYS, float)
        features.update(
            {name: (self.config.height, self.config.width, 3) for name in CAMERA_NAMES}
        )
        return features

    @property
    def action_features(self) -> dict:
        """The 14 channels this robot is driven by."""
        return dict.fromkeys(ACTION_KEYS, float)

    @property
    def cameras(self) -> dict:
        """Named for the rollout context, which counts them. There are three."""
        return dict.fromkeys(CAMERA_NAMES, None)

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        """Always. A model has no encoders to zero."""
        return True

    def calibrate(self) -> None:
        """Nothing to do. Kept because the interface has it."""

    def configure(self) -> None:
        """Nothing to do. The gains are in the scene, at the cell's own values."""

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def connect(self, calibrate: bool = True) -> None:
        """Build the model, the renderer and (optionally) the viewer. **Moves nothing.**

        Deliberately loud about being harmless: every other ``connect()`` in this
        project energises thirteen motors and drives two grippers open against
        their stop, and a reader who has learned to treat the word as dangerous
        should be told once that here it is not.
        """
        import mujoco

        del calibrate
        if self._connected:
            return
        logger.info("building the MuJoCo scene (nothing is energised; no /dev node is opened)")
        built = scene_module.build(
            self.config.urdf or scene_module.DEFAULT_URDF, objects=self.config.objects
        )
        self._model = mujoco.MjModel.from_xml_string(built.xml)
        self._data = mujoco.MjData(self._model)
        self._resolve_indices(mujoco)

        # One control period of physics per send_action, so the policy sees its
        # trained cadence whatever the wall clock and the GPU are doing — and
        # *exactly* one, which is why the model's timestep is set from the control
        # period rather than the substep count being rounded to whatever the
        # scene's own timestep happens to divide into. At 30 Hz and the scene's
        # 2 ms that rounding was 17 steps = 34 ms, a 2% clock error the policy
        # would never be told about.
        period = 1.0 / self.fps
        self._steps_per_tick = max(1, int(self.config.substeps))
        self._model.opt.timestep = period / self._steps_per_tick
        self._renderer = mujoco.Renderer(
            self._model, height=self.config.height, width=self.config.width
        )
        if self.config.view:
            self._viewer = self._open_viewer(mujoco)
        mujoco.mj_forward(self._model, self._data)
        self._next_frame = time.perf_counter()
        self._connected = True
        # Seed the ramp where the model actually is, so the first commanded action
        # cannot be a step away from it — the real follower does the same.
        self.limiter.reset()
        self.limiter.limit(self.measured_positions())
        logger.info(
            "sim connected: %d physics steps per %.1f ms tick, %d cameras at %dx%d",
            self._steps_per_tick, period * 1000, len(CAMERA_NAMES),
            self.config.width, self.config.height,
        )

    def _open_viewer(self, mujoco: Any) -> Any:
        """A **passive** viewer: it draws, the control loop drives.

        ``launch_passive`` rather than ``launch``, because ``launch`` takes the
        thread and runs its own stepping loop — which would leave the policy
        talking to a model somebody else was integrating.
        """
        try:
            from mujoco import viewer
        except Exception as exc:  # noqa: BLE001 - a missing display must not stop a run
            logger.warning("no mujoco.viewer (%s); running without a window", exc)
            return None
        try:
            return viewer.launch_passive(self._model, self._data, show_left_ui=False,
                                         show_right_ui=False)
        except Exception as exc:  # noqa: BLE001 - headless is a normal way to run this
            logger.warning("could not open a viewer (%s); running headless", exc)
            return None

    def disconnect(self) -> None:
        """Close the renderer and the window. **Moves nothing, disables nothing.**"""
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:  # noqa: BLE001 - closing must not raise
                logger.debug("the viewer was already gone")
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self._model = self._data = None
        self._connected = False
        self.limiter.reset()

    # ------------------------------------------------------------------ #
    # Indices — resolved once, so a tick is array lookups
    # ------------------------------------------------------------------ #

    def _resolve_indices(self, mujoco: Any) -> None:
        """Map the 14 channels onto qpos addresses and actuator ids.

        Done once at connect and against :data:`dk1lab.layout.ACTION_KEYS`, so
        the vector contract is read from the one place that owns it rather than
        re-derived from the model's own ordering — which is MuJoCo's business and
        not this cell's.
        """
        self._qpos, self._actuators = [], []
        for arm in ARMS:
            joints = scene_module.joint_names(arm)
            joint_acts, finger_acts = scene_module.actuator_names(arm)
            for index, key in enumerate(ARM_KEYS):
                joint_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, joints[index])
                if joint_id < 0:
                    raise RuntimeError(
                        f"the scene has no joint {joints[index]!r} for {arm}_{key}; "
                        f"dk1lab.scene and dk1lab.layout disagree"
                    )
                self._qpos.append(int(self._model.jnt_qposadr[joint_id]))
                # The gripper channel drives both fingers; every other channel
                # drives one actuator.
                names = finger_acts if is_gripper(key) else [joint_acts[index]]
                ids = tuple(
                    int(mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, name))
                    for name in names
                )
                if any(actuator < 0 for actuator in ids):
                    raise RuntimeError(f"the scene is missing actuators {names}")
                self._actuators.append(ids)

    # ------------------------------------------------------------------ #
    # Observation
    # ------------------------------------------------------------------ #

    def measured_positions(self) -> dict[str, float]:
        """The 14 joint positions, without rendering anything.

        The counterpart of :meth:`SafeBiDK1Follower.measured_positions` and for
        the same reason: :func:`dk1lab.policy.measure_fn` uses it for the home
        sweep, where three renders per tick would be paid for pictures nobody
        looks at.
        """
        self._require_connection()
        return {
            key: self._read(key, address)
            for key, address in zip(ACTION_KEYS, self._qpos, strict=True)
        }

    def _read(self, key: str, address: int) -> float:
        """One channel, in the units this cell reports them in."""
        value = float(self._data.qpos[address])
        return scene_module.gripper_normalised(value) if is_gripper(key) else value

    def get_observation(self) -> dict[str, Any]:
        """The positions and the three camera images, as the cell reports them.

        The image keys are the camera *names* — ``top`` / ``left`` / ``right`` —
        because that is what LeRobot's feature builders turn into
        ``observation.images.{name}``, and what the checkpoint pins.
        """
        self._require_connection()
        observation: dict[str, Any] = self.measured_positions()
        for name in CAMERA_NAMES:
            observation[name] = self._render(name)
        return observation

    def _render(self, camera: str) -> Any:
        """One camera, as an RGB ``(h, w, 3)`` array — the format the cell delivers."""
        self._renderer.update_scene(self._data, camera=camera)
        return self._renderer.render()

    # ------------------------------------------------------------------ #
    # Action
    # ------------------------------------------------------------------ #

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Drive the model for exactly one control period. **Returns what was sent.**

        Returning the action is not decoration: the recorders write what
        ``send_action`` returned, on the argument that a dataset must hold what
        the arms were *given*. So what comes back is the **limited** action when
        the limiter is on, exactly as :meth:`SafeBiDK1Follower.send_action` does.

        A channel the caller left out holds its last target rather than falling
        to zero, which is what the real chain does between commands.
        """
        import mujoco

        self._require_connection()
        if self.limiter.enabled:
            action = self.limiter.limit(action, self.measured_positions())
        for index, key in enumerate(ACTION_KEYS):
            if key not in action:
                continue
            value = float(action[key])
            target = scene_module.gripper_position(value) if is_gripper(key) else value
            for actuator in self._actuators[index]:
                self._data.ctrl[actuator] = target

        for _ in range(self._steps_per_tick):
            mujoco.mj_step(self._model, self._data)
        if self._viewer is not None:
            self._sync_viewer()
        if self.config.realtime:
            self._wait_for_the_wall_clock()
        return dict(action)

    def _sync_viewer(self) -> None:
        """Draw, and drop the window if it was closed rather than dying with it."""
        try:
            if not self._viewer.is_running():
                logger.info("the viewer window was closed; the rollout continues")
                self._viewer = None
                return
            self._viewer.sync()
        except Exception as exc:  # noqa: BLE001 - a window must not stop a rollout
            logger.warning("the viewer stopped updating (%s); continuing without it", exc)
            self._viewer = None

    def _wait_for_the_wall_clock(self) -> None:
        """Sleep until this tick's share of real time has passed.

        ``--realtime``'s whole job. The deadline advances by one period each tick
        rather than being measured from now, so a slow tick is absorbed by the
        next fast one instead of making the sim drift permanently behind — and a
        run that has fallen far behind (a long model call, a paused debugger)
        resets rather than trying to catch up in a burst.
        """
        period = 1.0 / self.fps
        self._next_frame += period
        remaining = self._next_frame - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)
        elif remaining < -period:
            self._next_frame = time.perf_counter()

    def _require_connection(self) -> None:
        if not self._connected:
            raise RuntimeError("the sim is not connected; call connect() first")
