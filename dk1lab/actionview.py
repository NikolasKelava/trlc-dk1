"""Three lines per joint, live: what the policy planned, what the arms were told, where they went.

``dk1 policy run --display`` already streams the cameras and the robot state to
Rerun — that is LeRobot's own ``log_rerun_data``. What it does **not** show is
the chain between the model and the motor, and that chain is where a rough
rollout has to be diagnosed. Four things happen to a number between
``predict_action_chunk`` and a joint moving, and only the first and the last are
currently visible:

1. the postprocessor turns the model's normalised output into radians;
2. :mod:`dk1lab.fifo` cross-fades the first rows of a new chunk into the plan it
   replaces, so the seam is a ramp;
3. :class:`~dk1lab.robot.SafeBiDK1Follower` rate-limits the command;
4. the impedance controller tracks it, imperfectly and with real compliance.

So this logs three series per joint, on the same axes and the same timeline:

===============  ============================================================
``policy/*``     the model's own row for this tick, in robot units, **before**
                 the cross-fade and before the speed limiter
``command/*``    what ``send_action`` actually sent — after both
``observation.`` where the joint really is, logged by LeRobot as always
===============  ============================================================

Read them together and the roughness is attributable without another rollout.
``policy`` rough and ``command`` following it means the plans themselves are
rough — the policy, or the replan interval. ``policy`` smooth and ``command``
lagging or stepping means it is ours: the limiter, or the splice. ``command``
smooth and ``observation`` rough means the arm, not the software.

**It attaches by wrapping, never by replacing** — the rule :mod:`dk1lab.trace`
and :mod:`dk1lab.modelview` both follow. Two wrappers, both of which forward
their call untouched and log as a side effect, and neither of which the control
loop knows about.

**It pins the Rerun layout, and that is not optional.** ``log_rerun_data``
builds a blueprint from the first observation it sees and caches it on itself;
that blueprint is an explicit grid, so ``policy/*`` and ``command/*`` — which
never pass through it — would get no view and be invisible until the operator
built one by hand. Filling the cache first is what stops that. Same coupling to
the same upstream implementation detail as :func:`dk1lab.modelview.pin_blueprint`,
written down in both places.

**A display must never take a rollout down.** Every call is guarded, and a
second failure switches the view off for the rest of the run rather than
printing a warning thirty times a second.
"""

from __future__ import annotations

import logging
from typing import Any

from .layout import ACTION_KEYS, CAMERA_NAMES, IMAGE_KEYS

logger = logging.getLogger(__name__)

#: Entity path prefixes. Deliberately *not* under ``action.`` or ``observation.``:
#: those namespaces belong to ``log_rerun_data``, which sweeps every key it is
#: handed into one shared view, and the whole point here is a view per joint.
POLICY_PREFIX = "policy"
COMMAND_PREFIX = "command"


def _scalar(value: Any) -> float | None:
    """``value`` as a float, or ``None`` if it is not one number."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class ActionView:
    """Logs the policy's plan and the robot's command to Rerun, once per tick.

    Args:
        keys: the action keys, in this cell's order — from
            :data:`dk1lab.layout.ACTION_KEYS`. Used both to name the entity
            paths and to turn the engine's 14-vector into named series.
    """

    def __init__(self, keys: tuple[str, ...] = ACTION_KEYS) -> None:
        self.keys = tuple(keys)
        self.ticks = 0
        #: Set when logging raises. A display that fails once fails every tick.
        self.failed: str | None = None
        self._engine: Any = None
        self._restore: list[tuple[Any, str, Any]] = []

    # -- attach ------------------------------------------------------------- #

    def attach(self, ctx: Any) -> None:
        """Wrap the engine's ``get_action`` and the robot's ``send_action``.

        Both are instance attributes afterwards, shadowing the bound methods, so
        anything else that already wrapped them — :class:`dk1lab.trace.RolloutTrace`
        does — stays in the chain rather than being replaced.
        """
        engine = ctx.policy.inference
        self._engine = engine
        robot = getattr(getattr(ctx, "hardware", None), "robot_wrapper", None)

        inner_get = engine.get_action

        def get_action(obs_frame: Any) -> Any:
            action = inner_get(obs_frame)
            self.ticks += 1
            self._log_policy()
            return action

        engine.get_action = get_action
        self._restore.append((engine, "get_action", inner_get))

        if robot is None:
            logger.debug("no robot wrapper to trace commands from")
            return
        inner_send = robot.send_action

        def send_action(action: Any) -> Any:
            sent = inner_send(action)
            self._log_command(sent if sent is not None else action)
            return sent

        robot.send_action = send_action
        self._restore.append((robot, "send_action", inner_send))

    def detach(self) -> None:
        """Put the wrapped methods back. For tests, and for a clean teardown."""
        for owner, name, inner in reversed(self._restore):
            try:
                setattr(owner, name, inner)
            except Exception:  # noqa: BLE001 - teardown must not raise
                logger.debug("could not restore %s.%s", type(owner).__name__, name)
        self._restore.clear()

    # -- logging ------------------------------------------------------------ #

    def _log_policy(self) -> None:
        """The model's own row for this tick, before the fade and the limiter.

        ``planned`` exists only on :mod:`dk1lab.fifo`'s engines. Under ``--rtc``
        or ``--no-fifo`` there is no such row to show — LeRobot's engines do not
        keep one — so the panel carries ``command`` and ``observation`` alone
        rather than inventing a third line.
        """
        row = getattr(self._engine, "planned", None)
        if row is None:
            return
        self._log(POLICY_PREFIX, dict(zip(self.keys, row, strict=False)))

    def _log_command(self, action: Any) -> None:
        """What ``send_action`` returned: the action the arms were actually given.

        ``SafeBiDK1Follower.send_action`` returns the **limited** action rather
        than the requested one, which is exactly the number wanted here — the
        difference between this and ``policy/`` is the speed cap doing its job,
        or failing to.
        """
        if not isinstance(action, dict):
            return
        self._log(COMMAND_PREFIX, action)

    def _log(self, prefix: str, values: dict) -> None:
        """One ``rr.log`` per named scalar, under ``prefix/<key>``.

        Per key rather than one batched ``Scalars`` because the blueprint below
        overlays this series with two others on a per-joint axis, and a batch
        under a single path cannot be split across views.
        """
        if self.failed is not None:
            return
        try:
            import rerun as rr

            for key, value in values.items():
                if key not in self.keys:
                    continue
                if (number := _scalar(value)) is not None:
                    rr.log(f"{prefix}/{key}", rr.Scalars(number))
        except Exception as exc:  # noqa: BLE001 - a display must never stop a rollout
            self.failed = str(exc)
            logger.warning("action view disabled after an error: %s", exc)


def build_blueprint(
    keys: tuple[str, ...] = ACTION_KEYS,
    camera_names: tuple[str, ...] = CAMERA_NAMES,
    *,
    model_input: bool = False,
) -> Any:
    """The layout itself: the cameras across the top, then one view per joint.

    Split out from :func:`pin_blueprint` because it has two callers that send it
    to different places — the live viewer, and the ``.rrd`` an episode recording
    writes (:mod:`dk1lab.record`). A recording that laid its three series out
    differently from the panel they were watched on would be a second thing to
    learn rather than the same thing to replay.
    """
    import rerun.blueprint as rrb

    joints = [
        rrb.TimeSeriesView(
            name=key.removesuffix(".pos"),
            contents=[
                f"{POLICY_PREFIX}/{key}",
                f"{COMMAND_PREFIX}/{key}",
                f"observation.{key}",
            ],
        )
        for key in keys
    ]
    images = [
        rrb.Spatial2DView(origin=f"observation.{name}", name=name) for name in camera_names
    ]
    if model_input:
        names = [key.rsplit(".", 1)[-1] for key in IMAGE_KEYS]
        images += [
            rrb.Spatial2DView(origin=f"policy_input/{name}", name=f"model {name}")
            for name in names
        ]

    # Seven columns puts one arm per row: joints 1-6 then the gripper.
    columns = max(1, len(keys) // 2)
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(*images),
            rrb.Grid(*joints, grid_columns=columns),
            row_shares=[1, 2],
        )
    )


def pin_blueprint(
    keys: tuple[str, ...] = ACTION_KEYS,
    camera_names: tuple[str, ...] = CAMERA_NAMES,
    *,
    model_input: bool = False,
) -> bool:
    """Lay Rerun out one view per joint, and stop LeRobot replacing it.

    The layout is the argument: **one time-series view per joint**, carrying
    ``policy``, ``command`` and ``observation`` for that joint on shared axes,
    arranged seven across so the top row is the left arm and the bottom row the
    right. Anything less — LeRobot's default of one view for all fourteen
    observations and another for all fourteen actions — puts the three numbers
    that have to be compared in three different panels at three different
    scales, which is why the comparison has never been made.

    ``_ensure_blueprint`` returns early when the cache attribute is already set,
    so a layout sent from here is the one that survives. A coupling to an
    upstream implementation detail, and the alternative is a flag that silently
    shows nothing.

    Args:
        model_input: also lay out the ``policy_input/*`` panels that
            ``--display-policy-input`` logs, so the two flags compose instead of
            the second one's views vanishing.

    Returns:
        Whether the blueprint was sent. ``False`` if Rerun is missing or not
        initialised — in which case everything still logs and Rerun falls back
        to its own automatic layout, which is a worse arrangement of the right
        data rather than a failure.
    """
    try:
        import rerun as rr
        import rerun.blueprint as rrb
        from lerobot.utils import rerun_visualization
    except ImportError:  # pragma: no cover - display is opt-in
        return False

    log_rerun_data = getattr(rerun_visualization, "log_rerun_data", None)
    if log_rerun_data is None or not hasattr(log_rerun_data, "blueprint"):
        return False

    blueprint = build_blueprint(keys, camera_names, model_input=model_input)
    try:
        rr.send_blueprint(blueprint)
    except Exception as exc:  # noqa: BLE001 - a layout must never stop a rollout
        logger.warning("could not send the Rerun layout: %s", exc)
        return False
    log_rerun_data.blueprint = blueprint
    return True
