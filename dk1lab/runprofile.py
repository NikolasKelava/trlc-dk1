"""Run profiles: the tuned configuration, and the level playing field.

Two ways to run a policy on this cell, and the difference between them is the
whole of what ``STUDY.md`` calls "the price of the level playing field".

``optimized``
    What this cell runs today, and has run for every rollout so far: the wrist
    crop and its offset from ``[cameras.left]`` / ``[cameras.right]``, and the
    speed cap in ``[limits.policy]``. **The default, and frozen** — nothing about
    it changes, which is what keeps the existing evidence (the eight ``.rrd``
    episodes, the six debugging rollouts) about a configuration that still
    exists.

``common``
    The same cell with every advantage removed: **no wrist crop, no offset**, the
    full 105 degree frame from all three cameras, and the tighter cap in
    ``[limits.study]``. Both policies in the two-policy comparison see exactly
    the same observations, so neither is being run on a rig tuned for the other.

**The profile is a derived config, not an edit.** :func:`apply` returns a new
:class:`~dk1lab.config.DK1Config` with the crop stripped out of the camera
devices; ``dk1.toml`` is never written and its ``[cameras.*]`` tables are never
touched. Every consumer downstream — :func:`dk1lab.cameras.camera_configs`, the
crop summary in the banner, the recorder's notes — reads the derived config and
therefore agrees about what the policy is being shown, without any of them
having to know that profiles exist.

That matters more than it looks. The crop lives *in the camera*
(:mod:`dk1lab.crop`) rather than in a policy processor precisely so that it is
true of every image this cell produces. Removing it anywhere but at the camera
would leave teleop, recording and rollout disagreeing about what the lens does —
which is the failure ``STUDY.md`` names as the one that will happen if any does:

    The demonstrations must be recorded through exactly the observation path the
    fine-tuned policy is rolled out under.

Nothing here imports LeRobot or torch, for the same reason
:mod:`dk1lab.config` does not: choosing a profile is a decision about
configuration, and it should be testable on a machine with no robot stack.

The module is ``runprofile`` and not ``profile`` because this project already
has two other things called a profile — the ``[capture.*]`` profile a camera
runs at, and the ``[limits.*]`` profile it is capped by. This is the third, and
the one that *selects* the other two.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .config import CameraDevice, DK1Config, LimitProfile

#: The tuned configuration this cell already runs. Default everywhere.
OPTIMIZED = "optimized"

#: The level playing field: identical observations for both policies.
COMMON = "common"

#: The speed limit a policy rollout runs under when ``dk1.toml`` says nothing.
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
#:
#: Defined here rather than in :mod:`dk1lab.policy` so that the fallback and the
#: profile that selects it sit together, and so that reading it costs no torch
#: import. ``dk1lab.policy.POLICY_LIMITS`` re-exports it.
POLICY_LIMITS = LimitProfile(
    max_joint_rate=1.0,
    max_gripper_rate=1.0,
    max_lag=0.4,
    max_dt=0.1,
)

#: The speed limit the ``common`` profile runs under when ``[limits.study]`` is
#: absent. **Only the rate drops**, to 0.6 rad/s.
#:
#: 0.6 sits between two measured points: at 0.3 the worst joint ended 0.98 rad
#: behind the policy's intent on 26% of ticks; at 1.0, 0.40 rad on 3.3%. It is
#: expected to cost both policies something, and what it costs is to be recorded
#: rather than assumed — both wear the same handicap, so the comparison survives
#: and the absolute numbers carry the caveat.
#:
#: ``max_lag`` is deliberately **not** tightened with it. It is a torque clamp,
#: not a position one; lowering it would stall the arms rather than make them
#: safer. See :data:`POLICY_LIMITS` and ``DIAGNOSTICS §`` *What the caps were
#: doing*.
STUDY_LIMITS = replace(POLICY_LIMITS, max_joint_rate=0.6)


@dataclass(frozen=True)
class RunProfile:
    """One way of running a policy on this cell: what it sees, and how fast.

    Args:
        name: ``optimized`` or ``common``, as typed at ``--profile``.
        limits_table: which ``[limits.*]`` table in ``dk1.toml`` this profile
            reads.
        fallback: the limit used when that table is absent from the file.
        cropped: whether the wrist crop applies. ``False`` strips ``target_hfov``
            and the three tuning offsets out of every camera, so all three
            deliver the lens's own field of view.
        summary: one line for the banner, saying what was given up and why.
    """

    name: str
    limits_table: str
    fallback: LimitProfile
    cropped: bool
    summary: str

    def apply(self, config: DK1Config) -> DK1Config:
        """``config`` as this profile sees it. Never writes ``dk1.toml``."""
        return apply(config, self)

    def limits(self, config: DK1Config) -> LimitProfile:
        """This profile's speed limit, from the file or from :attr:`fallback`."""
        return config.limit(self.limits_table, self.fallback)


#: The profiles, keyed by the string ``--profile`` accepts.
PROFILES: dict[str, RunProfile] = {
    OPTIMIZED: RunProfile(
        name=OPTIMIZED,
        limits_table="policy",
        fallback=POLICY_LIMITS,
        cropped=True,
        summary=(
            "the tuned configuration this cell already runs — wrist crop on, "
            "[limits.policy]"
        ),
    ),
    COMMON: RunProfile(
        name=COMMON,
        limits_table="study",
        fallback=STUDY_LIMITS,
        cropped=False,
        summary=(
            "the level playing field — no wrist crop, no offset, the full lens "
            "on all three cameras, [limits.study]"
        ),
    ),
}

#: The default. Everything that worked before this flag existed still does.
DEFAULT_PROFILE = OPTIMIZED


class ProfileError(ValueError):
    """Raised for a profile name that does not exist."""


def resolve(name: str | None) -> RunProfile:
    """The :class:`RunProfile` called ``name``. ``None`` gives the default.

    Raises:
        ProfileError: naming what was asked for and what there is, because a
            misspelled profile that silently fell back to the default would run
            an experiment under a configuration nobody chose.
    """
    if name is None:
        return PROFILES[DEFAULT_PROFILE]
    try:
        return PROFILES[name]
    except KeyError:
        raise ProfileError(
            f"no such profile {name!r} — expected one of {', '.join(PROFILES)}"
        ) from None


def uncropped(device: CameraDevice) -> CameraDevice:
    """``device`` with the crop removed: the lens's own field of view, whole.

    ``hfov`` is kept — it is the lens's own angle and stays true — while
    ``target_hfov`` and the three hand-tuned offsets go, which is exactly what
    makes :func:`dk1lab.cameras.camera_configs` build a plain ``OpenCVCamera``
    for it. ``rotation`` is untouched: the cameras really are mounted upside
    down, and that is a fact about the cell rather than a tuning choice.
    """
    return replace(
        device,
        target_hfov=None,
        crop_inset=0.0,
        crop_shift_x=0.0,
        crop_shift_y=0.0,
    )


def apply(config: DK1Config, profile: RunProfile) -> DK1Config:
    """``config`` as ``profile`` sees it — a new object, in memory only.

    Under ``optimized`` this is the identity, so the default path is the same
    object it has always been. Under ``common`` every camera loses its crop.
    """
    if profile.cropped:
        return config
    return replace(
        config,
        cameras={name: uncropped(device) for name, device in config.cameras.items()},
    )
