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

__all__ = ["__version__"]

__version__ = "0.1.0"
