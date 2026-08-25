"""``dk1 sim`` — the MuJoCo cell. Nothing here touches the arms.

    dk1 sim scene    build the scene and write the MJCF. No window, no physics.
    dk1 sim view     open it and hold the home pose. The scene, looked at.
    dk1 sim sweep    drive both arms through a canned motion, so the clock,
                     the cameras and the actuators are all exercised.

None of these opens a ``/dev`` node, energises a motor or moves anything in the
room. That is worth saying plainly, because every other command in this CLI that
uses the words *connect*, *send* and *sweep* does all three.

What the sim is **for** is one thing, from ``STUDY.md``: confirming a policy
drives the pipeline before it drives the arms. It produces no episodes and no
scores, and a sim rollout that goes well is evidence that the plumbing works and
nothing else.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(no_args_is_help=True, help=__doc__)

RealtimeOpt = Annotated[
    bool,
    typer.Option(
        "--realtime/--free-run",
        help=(
            "Keep the sim in step with the wall clock, or let it run as fast as it "
            "can. --realtime is for watching; --free-run is for an unattended check. "
            "Neither changes a single number the policy sees: one control tick is "
            "one 1/30 s of physics either way."
        ),
    ),
]
ViewOpt = Annotated[
    bool,
    typer.Option("--view/--no-view", help="Open mujoco.viewer. Passive: it draws, the loop drives."),
]
ObjectsOpt = Annotated[
    bool,
    typer.Option(
        "--objects/--no-objects",
        help=(
            "Put the die and the bowl in the scene. --no-objects is the kinematic "
            "check: the arms move, nothing is grasped."
        ),
    ),
]
UrdfOpt = Annotated[
    Path | None,
    typer.Option("--urdf", help="The follower URDF the scene is built from. Read, never written."),
]
ConfigOpt = Annotated[
    Path | None,
    typer.Option("--config", "-c", help="Path to dk1.toml, for the speed limit."),
]


def _sim(objects: bool, urdf, *, realtime: bool, view: bool, config):
    """A :class:`~dk1lab.sim.SimRobotConfig` at the cap a rollout would run under.

    The speed limit comes from ``[limits.<profile>]`` in dk1.toml, through the
    same :func:`dk1lab.policy.sim_config` a policy rollout goes through, so the
    sim is never quietly capped differently from the thing it stands in for.
    """
    from ..config import DEFAULT_CONFIG_PATH, load
    from ..policy import sim_config
    from ..runprofile import resolve

    settings = load(config or DEFAULT_CONFIG_PATH, require_devices=False)
    built = sim_config(
        settings,
        limits=resolve(None).limits(settings),
        realtime=realtime,
        view=view,
    )
    built.objects = objects
    built.urdf = str(urdf) if urdf else None
    return built


def _build(urdf, objects: bool):
    from ..scene import DEFAULT_URDF, SceneError, build

    try:
        return build(urdf or DEFAULT_URDF, objects=objects)
    except SceneError as exc:
        typer.secho(f"\n{exc}\n", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc


# --------------------------------------------------------------------------- #
# scene — the MJCF, and nothing else
# --------------------------------------------------------------------------- #


@app.command("scene")
def scene_cmd(
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Write the MJCF here, e.g. to open it in the viewer yourself."),
    ] = None,
    objects: ObjectsOpt = True,
    urdf: UrdfOpt = None,
) -> None:
    """Build the scene from this repo's URDF and report what came out.

    No physics, no window, no robot. This is the cheapest check that the URDF is
    still importable and that the two arms, the three cameras and the actuators
    all came out of it.
    """
    import mujoco

    from ..layout import CAMERA_NAMES
    from ..scene import ARM_Y

    built = _build(urdf, objects)
    model = mujoco.MjModel.from_xml_string(built.xml)

    typer.secho("scene", bold=True)
    typer.echo(f"  from        {built.urdf}")
    typer.echo(f"  bodies      {model.nbody}")
    typer.echo(f"  joints      {model.njnt} ({model.nq} qpos)")
    typer.echo(f"  actuators   {model.nu}")
    typer.echo(f"  arms        " + ", ".join(f"{side} at y={y:+g}" for side, y in ARM_Y.items()))

    cameras = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, index)
        for index in range(model.ncam)
    ]
    typer.secho("\ncameras", bold=True)
    for name in cameras:
        typer.echo(f"  {name}")
    missing = [name for name in CAMERA_NAMES if name not in cameras]
    if missing:
        typer.secho(
            f"  MISSING {missing} — the policy's image keys would not resolve",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    typer.echo(f"  this cell's names, in order: {', '.join(CAMERA_NAMES)}")

    if out is not None:
        typer.secho(f"\nwrote {built.write(out)}", fg=typer.colors.GREEN)
        typer.echo(f"  look at it with:  python -m mujoco.viewer --mjcf={out}")
    typer.secho("\nthe scene builds. Nothing was connected and nothing moved.", fg=typer.colors.GREEN)


# --------------------------------------------------------------------------- #
# view — the scene, held at home
# --------------------------------------------------------------------------- #


@app.command("view")
def view_cmd(
    seconds: Annotated[
        float, typer.Option("--seconds", help="How long to hold it. 0 = until the window closes.")
    ] = 0.0,
    objects: ObjectsOpt = True,
    urdf: UrdfOpt = None,
    config: ConfigOpt = None,
) -> None:
    """Open the scene and hold the home pose. **The real arms are not involved.**

    The sim arms are commanded — to stand still at the pose ``[home]`` names,
    which in the model is its own zero. Ctrl-C leaves.
    """
    from ..layout import ACTION_KEYS
    from ..sim import SimRobot

    robot = SimRobot(_sim(objects, urdf, realtime=True, view=True, config=config))
    typer.secho("opening the MuJoCo cell — no /dev node, no motor, no motion in the room",
                fg=typer.colors.GREEN)
    robot.connect()
    hold = dict.fromkeys(ACTION_KEYS, 0.0)
    ticks = int(seconds * robot.fps) if seconds else 0
    try:
        index = 0
        while ticks == 0 or index < ticks:
            robot.send_action(hold)
            index += 1
            if robot._viewer is None and ticks == 0 and index > robot.fps:
                # The window is gone and nobody asked for a duration, so there is
                # nothing left to look at.
                break
    except KeyboardInterrupt:
        typer.echo()
    finally:
        robot.disconnect()
    typer.secho("closed.", fg=typer.colors.GREEN)


# --------------------------------------------------------------------------- #
# sweep — a canned motion, so every part of the loop is exercised
# --------------------------------------------------------------------------- #


@app.command("sweep")
def sweep_cmd(
    seconds: Annotated[float, typer.Option("--seconds", help="How long to move for.")] = 10.0,
    realtime: RealtimeOpt = True,
    view: ViewOpt = True,
    objects: ObjectsOpt = True,
    render: Annotated[
        bool,
        typer.Option(
            "--render/--no-render",
            help="Render all three cameras every tick, as a rollout does. On, because "
            "that is the cost a policy will actually pay.",
        ),
    ] = True,
    urdf: UrdfOpt = None,
    config: ConfigOpt = None,
) -> None:
    """Drive both sim arms through a canned reach, so the whole loop is exercised.

    Not a policy and not a task: a smooth sweep of every joint and both grippers,
    which is enough to show that the clock, the actuators, the cameras and the
    viewer all work together at 30 Hz. **The real arms are not involved.**

    The rate it reports is the useful number. Under ``--free-run`` it is how fast
    the sim *can* go, which is what says whether a policy will be waiting for the
    simulator or the simulator for the policy.
    """
    import math
    import time

    from ..layout import ACTION_KEYS, is_gripper
    from ..sim import SimRobot

    robot = SimRobot(_sim(objects, urdf, realtime=realtime, view=view, config=config))
    typer.secho("driving the MuJoCo cell — no /dev node, no motor, no motion in the room",
                fg=typer.colors.GREEN)
    typer.echo(f"  {'realtime' if realtime else 'free-run'}, "
               f"{'rendering' if render else 'no rendering'}, "
               f"{'viewer' if view else 'headless'}, "
               f"cap {robot.config.max_joint_rate} rad/s from dk1.toml")
    robot.connect()

    ticks = max(1, int(seconds * robot.fps))
    started = time.perf_counter()
    index = 0
    try:
        for index in range(ticks):
            phase = 2 * math.pi * index / (robot.fps * 4.0)
            action = {}
            for offset, key in enumerate(ACTION_KEYS):
                if is_gripper(key):
                    action[key] = 0.5 * (1 - math.cos(phase))
                else:
                    action[key] = 0.4 * math.sin(phase + offset * 0.4)
            if render:
                robot.get_observation()
            robot.send_action(action)
    except KeyboardInterrupt:
        typer.echo()
        ticks = index + 1
    finally:
        elapsed = time.perf_counter() - started
        robot.disconnect()

    rate = ticks / elapsed if elapsed > 0 else 0.0
    typer.secho(f"\n{ticks} ticks in {elapsed:.1f} s = {rate:.1f} Hz", bold=True)
    if realtime:
        typer.echo(f"  --realtime holds it at {robot.fps} Hz; --free-run says how fast it can go")
    elif rate < robot.fps:
        typer.secho(
            f"  slower than the {robot.fps} Hz the policy expects — under --realtime the "
            f"sim would fall behind. --no-render is the first thing to try.",
            fg=typer.colors.YELLOW,
        )
    typer.secho("nothing was connected and nothing in the room moved.", fg=typer.colors.GREEN)
