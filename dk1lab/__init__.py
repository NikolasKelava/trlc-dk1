"""Bimanual TRLC-DK1 operation and MolmoAct2 tooling.

Everything this fork adds lives here; upstream's packages are left untouched so
the fork stays rebaseable. The entry point is the ``dk1`` command
(:mod:`dk1lab.cli.main`).

Import order note: :mod:`dk1lab.layout`, :mod:`dk1lab.config`,
:mod:`dk1lab.limiter` and :mod:`dk1lab.discovery` are deliberately free of any
LeRobot or torch import, so config handling and the tests around it stay fast and
usable on a machine with no robot stack installed. :mod:`dk1lab.cameras` and
:mod:`dk1lab.robot` are where LeRobot enters.
"""

__all__ = ["CroppedOpenCVCamera", "SafeBiDK1Follower", "SimRobot", "__version__"]

__version__ = "0.1.0"


def __getattr__(name: str):
    """Expose ``SafeBiDK1Follower`` here, lazily, because LeRobot looks for it here.

    ``lerobot-rollout`` does not instantiate a robot the way ``dk1 teleop`` does.
    It calls ``make_robot_from_config``, which reconstructs the *class* name from
    the *config* class name — ``SafeBiDK1FollowerConfig`` minus ``Config`` — and
    then looks for it in the package containing the config's module, and in
    ``<package>.<classname.lower()>``. Since the config lives in
    ``dk1lab.robot``, that means ``dk1lab`` and ``dk1lab.safebidk1follower``, and
    neither carried the class:

        ImportError: Could not locate device class 'SafeBiDK1Follower' ...

    Registering the config subclass is not enough — registration decides which
    config a ``--robot.type`` string parses into, and this is the separate step
    that turns that config into an object. Teleoperation never hit it, because
    it constructs the follower directly.

    It is done through a module ``__getattr__`` rather than a plain import so
    that ``import dk1lab`` still costs nothing: :mod:`dk1lab.robot` pulls in
    LeRobot and torch, and the config-only half of this package is deliberately
    usable without them.
    """
    if name == "SafeBiDK1Follower":
        from .robot import SafeBiDK1Follower

        return SafeBiDK1Follower
    if name == "CroppedOpenCVCamera":
        # Same story, one layer down: LeRobot's `make_cameras_from_configs` has
        # a hardcoded branch per built-in camera type and falls through to
        # `make_device_from_device_class` for anything else, which looks for
        # `CroppedOpenCVCamera` in the package holding `CroppedOpenCVCameraConfig`
        # — that is, here. Registering the config subclass only decides what a
        # `type: opencv_cropped` string parses into.
        from .crop import CroppedOpenCVCamera

        return CroppedOpenCVCamera
    if name == "SimRobot":
        # And once more, for the MuJoCo cell: `make_robot_from_config` finds a
        # robot by stripping "Config" off its config class's name and looking for
        # the result in the package holding that config's module — here.
        from .sim import SimRobot

        return SimRobot
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
