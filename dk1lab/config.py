"""Load, validate and surgically update ``dk1.toml`` — the one device config.

``dk1.toml`` is the only place arm serial ports, camera device nodes, camera
rotations and capture profiles are written down. Nothing else in this repo
hardcodes a port or a ``/dev`` path.

Two properties matter enough to be enforced rather than documented:

**Validation is total and happens on load.** A malformed or incomplete config
fails immediately, naming the file and the offending key, instead of surfacing
later as a camera that opens the wrong device or an arm that does not respond.

**Writes are surgical.** :func:`write_arms` rewrites only ``[arms.*]`` and
:func:`write_cameras` rewrites only ``[cameras.*]``; each leaves every other
section, and every comment, byte-identical. The previous iteration of this
project lost its whole camera section to a port-discovery run that rewrote the
file wholesale, and every script that depended on it broke at once.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomlkit

#: Bumped when the file's shape changes incompatibly.
SCHEMA_VERSION = 1

#: Default location: ``dk1.toml`` beside the repo root.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "dk1.toml"

_VALID_ROTATIONS = (0, 90, 180, 270)
_ARM_ROLES = ("follower", "leader")
_ARM_SIDES = ("left", "right")


class ConfigError(Exception):
    """Raised for any unusable ``dk1.toml``, with a message naming the key."""


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ArmPorts:
    """The two serial ports of one arm pair."""

    left: str
    right: str

    def as_dict(self) -> dict[str, str]:
        return {"left": self.left, "right": self.right}


@dataclass(frozen=True)
class CameraDevice:
    """One camera's stable device node and mounting rotation."""

    path: str
    rotation: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {"path": self.path, "rotation": self.rotation}


@dataclass(frozen=True)
class CaptureProfile:
    """A capture mode. Resolution differs by use, device identity does not."""

    width: int
    height: int
    fps: int
    fourcc: str = "MJPG"

    def as_dict(self) -> dict[str, Any]:
        return {"width": self.width, "height": self.height, "fps": self.fps, "fourcc": self.fourcc}


@dataclass(frozen=True)
class LimitProfile:
    """A speed limit for one activity, from ``[limits.*]``.

    ``max_joint_rate`` of ``None`` means no limiting at all — the limiter passes
    commands straight through and the follower also skips the extra motor read it
    would otherwise need on every tick. In the file that is written ``false``,
    because TOML has no null.
    """

    max_joint_rate: float | None
    max_gripper_rate: float
    max_lag: float
    max_dt: float

    def unlimited(self) -> LimitProfile:
        """The same profile with the joint cap removed. Deliberately explicit."""
        return LimitProfile(None, self.max_gripper_rate, self.max_lag, self.max_dt)

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_joint_rate": False if self.max_joint_rate is None else self.max_joint_rate,
            "max_gripper_rate": self.max_gripper_rate,
            "max_lag": self.max_lag,
            "max_dt": self.max_dt,
        }


@dataclass(frozen=True)
class PolicySettings:
    """``[policy]`` — where the MolmoAct2 checkpoint lives on this machine.

    Not a device, but the same argument applies: the checkpoint is a machine-
    specific path, it is needed by more than one command, and the alternative is
    a constant baked into Python. The section is optional — every policy command
    also takes ``--checkpoint``.
    """

    checkpoint: str


@dataclass(frozen=True)
class HomePose:
    """``[home]`` — the pose the arms are sent to when a run ends.

    Stored as seven numbers per arm, in the order one arm contributes to the 14-D
    vector: ``joint_1 .. joint_6`` then the gripper (``layout.ARM_KEYS``). Written
    as arrays rather than named keys because TOML's dotted keys would turn
    ``left_joint_1.pos`` into a nested table, and because the order is the same
    contract everything else in this project derives from :mod:`dk1lab.layout`.

    Optional. Without it, homing falls back to the pose the arms were in when the
    run connected — which is a pose, but not a *chosen* one. Capture the current
    pose into the file with ``dk1 policy home --capture``.
    """

    left: tuple[float, ...]
    right: tuple[float, ...]

    def as_action_dict(self) -> dict[str, float]:
        """The pose as the 14 canonical action keys — the form the robot wants."""
        from .layout import ARM_KEYS

        return {
            f"{side}_{key}": float(value)
            for side, values in (("left", self.left), ("right", self.right))
            for key, value in zip(ARM_KEYS, values, strict=True)
        }

    @classmethod
    def from_action_dict(cls, action: dict[str, float]) -> HomePose:
        """Inverse of :meth:`as_action_dict`, for capturing a live pose."""
        from .layout import ARM_KEYS

        def arm(side: str) -> tuple[float, ...]:
            return tuple(float(action[f"{side}_{key}"]) for key in ARM_KEYS)

        return cls(left=arm("left"), right=arm("right"))

    def as_dict(self) -> dict[str, Any]:
        return {"left": list(self.left), "right": list(self.right)}


@dataclass(frozen=True)
class DK1Config:
    """A validated ``dk1.toml``."""

    follower: ArmPorts
    leader: ArmPorts
    cameras: dict[str, CameraDevice]
    capture: dict[str, CaptureProfile]
    limits: dict[str, LimitProfile] = field(default_factory=dict)
    policy: PolicySettings | None = None
    home: HomePose | None = None
    path: Path = field(default=DEFAULT_CONFIG_PATH, compare=False)

    def camera(self, name: str) -> CameraDevice:
        try:
            return self.cameras[name]
        except KeyError:
            raise ConfigError(f"{self.path}: no camera named {name!r}") from None

    def profile(self, name: str) -> CaptureProfile:
        try:
            return self.capture[name]
        except KeyError:
            raise ConfigError(
                f"{self.path}: no capture profile named {name!r} "
                f"(have: {', '.join(sorted(self.capture))})"
            ) from None

    def limit(self, name: str, default: LimitProfile) -> LimitProfile:
        """The ``[limits.<name>]`` profile, or ``default`` if the file omits it.

        The caller supplies the fallback rather than this module inventing one:
        what counts as a sane speed depends on what is driving the arms, which is
        a fact about teleoperation or about a policy, not about the config format.
        """
        return self.limits.get(name, default)

    def checkpoint(self) -> str:
        """The configured policy checkpoint, or a message saying how to set one."""
        if self.policy is None:
            raise ConfigError(
                f"{self.path}: no [policy] section, so there is no default checkpoint. "
                f"Pass --checkpoint, or add:\n\n    [policy]\n    checkpoint = \"...\"\n"
            )
        return self.policy.checkpoint

    def arm_ports(self) -> dict[str, str]:
        """All four ports, keyed ``<role>_<side>``, for reporting."""
        return {
            f"{role}_{side}": getattr(getattr(self, role), side)
            for role in _ARM_ROLES
            for side in _ARM_SIDES
        }


# --------------------------------------------------------------------------- #
# Loading and validation
# --------------------------------------------------------------------------- #


def _require_table(raw: dict[str, Any], key: str, where: Path) -> dict[str, Any]:
    if key not in raw:
        raise ConfigError(f"{where}: missing required section [{key}]")
    value = raw[key]
    if not isinstance(value, dict):
        raise ConfigError(f"{where}: [{key}] must be a table, got {type(value).__name__}")
    return value


def _require_port(table: dict[str, Any], role: str, side: str, where: Path) -> str:
    if side not in table:
        raise ConfigError(f"{where}: missing key [arms.{role}].{side}")
    value = table[side]
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{where}: [arms.{role}].{side} must be a non-empty string, got {value!r}")
    return value


def parse(raw: dict[str, Any], path: Path = DEFAULT_CONFIG_PATH) -> DK1Config:
    """Validate an already-decoded TOML mapping into a :class:`DK1Config`.

    Split out from :func:`load` so tests can exercise validation without a file.
    """
    version = raw.get("version")
    if version is None:
        raise ConfigError(f"{path}: missing top-level `version` (expected {SCHEMA_VERSION})")
    if version != SCHEMA_VERSION:
        raise ConfigError(
            f"{path}: version {version!r} is not supported by this checkout "
            f"(expected {SCHEMA_VERSION})"
        )

    # --- arms ---
    arms = _require_table(raw, "arms", path)
    ports: dict[str, ArmPorts] = {}
    for role in _ARM_ROLES:
        table = _require_table(arms, role, path) if role in arms else None
        if table is None:
            raise ConfigError(f"{path}: missing required section [arms.{role}]")
        ports[role] = ArmPorts(
            left=_require_port(table, role, "left", path),
            right=_require_port(table, role, "right", path),
        )

    seen: dict[str, str] = {}
    for role in _ARM_ROLES:
        for side in _ARM_SIDES:
            port = getattr(ports[role], side)
            label = f"[arms.{role}].{side}"
            if port in seen:
                raise ConfigError(
                    f"{path}: {label} and {seen[port]} are both {port!r}. "
                    f"Each of the four arms needs its own serial port — "
                    f"re-run `dk1 find arms`."
                )
            seen[port] = label

    # --- cameras ---
    from .layout import CAMERA_NAMES  # local import: keeps layout dependency-free

    cameras_raw = _require_table(raw, "cameras", path)
    missing = [name for name in CAMERA_NAMES if name not in cameras_raw]
    if missing:
        raise ConfigError(
            f"{path}: [cameras] is missing {missing}. The MolmoAct2 BimanualYAM "
            f"checkpoint requires exactly {list(CAMERA_NAMES)} — re-run `dk1 find cameras`."
        )
    unexpected = [name for name in cameras_raw if name not in CAMERA_NAMES]
    if unexpected:
        raise ConfigError(
            f"{path}: [cameras] has unexpected entries {unexpected}; "
            f"only {list(CAMERA_NAMES)} are used."
        )

    cameras: dict[str, CameraDevice] = {}
    for name in CAMERA_NAMES:
        table = cameras_raw[name]
        if not isinstance(table, dict):
            raise ConfigError(f"{path}: [cameras.{name}] must be a table")
        device_path = table.get("path")
        if not isinstance(device_path, str) or not device_path.strip():
            raise ConfigError(
                f"{path}: [cameras.{name}].path must be a non-empty string, got {device_path!r}"
            )
        rotation = table.get("rotation", 0)
        if rotation not in _VALID_ROTATIONS:
            raise ConfigError(
                f"{path}: [cameras.{name}].rotation must be one of {list(_VALID_ROTATIONS)}, "
                f"got {rotation!r}"
            )
        cameras[name] = CameraDevice(path=device_path, rotation=int(rotation))

    by_path: dict[str, str] = {}
    for name, device in cameras.items():
        if device.path in by_path:
            raise ConfigError(
                f"{path}: cameras {by_path[device.path]!r} and {name!r} both point at "
                f"{device.path!r}. All three DK1 cameras share serial 20010101, so they "
                f"must be addressed by distinct /dev/v4l/by-path nodes — re-run "
                f"`dk1 find cameras`."
            )
        by_path[device.path] = name

    # --- capture profiles ---
    capture_raw = _require_table(raw, "capture", path)
    if not capture_raw:
        raise ConfigError(f"{path}: [capture] must define at least one profile")
    capture: dict[str, CaptureProfile] = {}
    for name, table in capture_raw.items():
        if not isinstance(table, dict):
            raise ConfigError(f"{path}: [capture.{name}] must be a table")
        for key in ("width", "height", "fps"):
            value = table.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ConfigError(
                    f"{path}: [capture.{name}].{key} must be a positive integer, got {value!r}"
                )
        fourcc = table.get("fourcc", "MJPG")
        if not isinstance(fourcc, str) or len(fourcc) != 4:
            raise ConfigError(
                f"{path}: [capture.{name}].fourcc must be a 4-character code, got {fourcc!r}"
            )
        capture[name] = CaptureProfile(
            width=table["width"], height=table["height"], fps=table["fps"], fourcc=fourcc
        )

    # --- limits (optional) ---
    limits: dict[str, LimitProfile] = {}
    limits_raw = raw.get("limits", {})
    if not isinstance(limits_raw, dict):
        raise ConfigError(f"{path}: [limits] must be a table, got {type(limits_raw).__name__}")
    for name, table in limits_raw.items():
        if not isinstance(table, dict):
            raise ConfigError(f"{path}: [limits.{name}] must be a table")
        rate = table.get("max_joint_rate", False)
        # `false` is how the file spells "no limit"; TOML has no null. Anything
        # else must be a positive number, so a typo cannot silently disable the
        # cap — the one failure mode that matters here.
        if rate is False:
            max_joint_rate = None
        elif isinstance(rate, (int, float)) and not isinstance(rate, bool) and rate > 0:
            max_joint_rate = float(rate)
        else:
            raise ConfigError(
                f"{path}: [limits.{name}].max_joint_rate must be a positive number, or "
                f"`false` for no limit at all — got {rate!r}"
            )
        values: dict[str, float] = {}
        for key, fallback in (("max_gripper_rate", 1.0), ("max_lag", 0.15), ("max_dt", 0.1)):
            value = table.get(key, fallback)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
                raise ConfigError(
                    f"{path}: [limits.{name}].{key} must be a positive number, got {value!r}"
                )
            values[key] = float(value)
        unexpected = [
            key
            for key in table
            if key not in ("max_joint_rate", "max_gripper_rate", "max_lag", "max_dt")
        ]
        if unexpected:
            raise ConfigError(f"{path}: [limits.{name}] has unknown keys {unexpected}")
        limits[name] = LimitProfile(max_joint_rate=max_joint_rate, **values)

    # --- policy (optional) ---
    policy: PolicySettings | None = None
    policy_raw = raw.get("policy")
    if policy_raw is not None:
        if not isinstance(policy_raw, dict):
            raise ConfigError(f"{path}: [policy] must be a table, got {type(policy_raw).__name__}")
        checkpoint = policy_raw.get("checkpoint")
        if not isinstance(checkpoint, str) or not checkpoint.strip():
            raise ConfigError(
                f"{path}: [policy].checkpoint must be a non-empty string "
                f"(a local directory or a Hugging Face repo id), got {checkpoint!r}"
            )
        unexpected = [key for key in policy_raw if key != "checkpoint"]
        if unexpected:
            raise ConfigError(f"{path}: [policy] has unknown keys {unexpected}")
        policy = PolicySettings(checkpoint=checkpoint)

    # --- home pose (optional) ---
    home = _parse_home(raw.get("home"), path)

    return DK1Config(
        follower=ports["follower"],
        leader=ports["leader"],
        cameras=cameras,
        capture=capture,
        limits=limits,
        policy=policy,
        home=home,
        path=path,
    )


def _parse_home(raw: Any, path: Path) -> HomePose | None:
    """Validate ``[home]``: seven finite numbers per arm, or nothing at all.

    A half-written home pose is worse than none — the joints it omits stay
    wherever the run left them while the others move, which looks like homing
    and is not — so every departure from the shape is an error, not a default.
    """
    if raw is None:
        return None
    from .layout import ARM_KEYS

    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: [home] must be a table, got {type(raw).__name__}")
    unexpected = [key for key in raw if key not in _ARM_SIDES]
    if unexpected:
        raise ConfigError(f"{path}: [home] has unknown keys {unexpected}; expected left and right")

    values: dict[str, tuple[float, ...]] = {}
    for side in _ARM_SIDES:
        if side not in raw:
            raise ConfigError(
                f"{path}: [home] is missing `{side}`. Both arms must be given, or the "
                f"whole section left out — re-capture with `dk1 policy home --capture`."
            )
        entry = raw[side]
        if not isinstance(entry, list) or len(entry) != len(ARM_KEYS):
            raise ConfigError(
                f"{path}: [home].{side} must be a list of {len(ARM_KEYS)} numbers "
                f"({', '.join(ARM_KEYS)}), got {entry!r}"
            )
        for index, value in enumerate(entry):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ConfigError(
                    f"{path}: [home].{side}[{index}] ({ARM_KEYS[index]}) must be a number, "
                    f"got {value!r}"
                )
        values[side] = tuple(float(value) for value in entry)

    return HomePose(left=values["left"], right=values["right"])


def load(path: Path | str = DEFAULT_CONFIG_PATH, *, require_devices: bool = False) -> DK1Config:
    """Read and validate ``dk1.toml``.

    Args:
        path: config file to read.
        require_devices: additionally assert every configured ``/dev`` node
            currently exists. Off by default so the config can be inspected and
            tested on a machine with no robot attached; ``dk1 config check``
            turns it on.
    """
    path = Path(path)
    if not path.exists():
        raise ConfigError(
            f"{path} does not exist. It is tracked in this repo, so this usually means "
            f"you are running from the wrong directory; otherwise run `dk1 find arms` and "
            f"`dk1 find cameras` to generate it."
        )
    try:
        raw = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: invalid TOML — {exc}") from exc

    config = parse(raw, path=path)
    if require_devices:
        check_devices(config)
    return config


def check_devices(config: DK1Config) -> None:
    """Raise if any configured device node is absent right now."""
    problems: list[str] = []
    for label, port in config.arm_ports().items():
        if not Path(port).exists():
            problems.append(f"  arm {label}: {port} not present")
    for name, device in config.cameras.items():
        if not Path(device.path).exists():
            problems.append(f"  camera {name}: {device.path} not present")
    if problems:
        raise ConfigError(
            f"{config.path}: configured devices are missing:\n"
            + "\n".join(problems)
            + "\n\nIs the robot powered and plugged in? Device nodes can also move after "
            "a replug — re-run `dk1 find arms` / `dk1 find cameras`."
        )


# --------------------------------------------------------------------------- #
# Surgical writes
# --------------------------------------------------------------------------- #


def _subtable(parent: Any, key: str) -> Any:
    """The existing sub-table at ``key``, created empty if absent.

    Returning the existing object matters: tomlkit attaches a section's trailing
    whitespace and the comment introducing the next section to that section, so
    replacing a table drops both.
    """
    existing = parent.get(key)
    if isinstance(existing, dict):
        return existing
    table = tomlkit.table()
    parent[key] = table
    return table


def _load_document(path: Path) -> tomlkit.TOMLDocument:
    if path.exists():
        return tomlkit.parse(path.read_text())
    doc = tomlkit.document()
    doc["version"] = SCHEMA_VERSION
    return doc


def write_arms(ports: dict[str, ArmPorts], path: Path | str = DEFAULT_CONFIG_PATH) -> None:
    """Replace only ``[arms.follower]`` and ``[arms.leader]``.

    Every other section — crucially ``[cameras]`` — and every comment survive
    untouched. This is the single most important invariant in this module.
    """
    path = Path(path)
    missing = [role for role in _ARM_ROLES if role not in ports]
    if missing:
        raise ConfigError(f"write_arms: missing roles {missing}")

    doc = _load_document(path)
    arms = doc.get("arms")
    if not isinstance(arms, dict):
        arms = tomlkit.table(is_super_table=True)
        doc["arms"] = arms
    for role in _ARM_ROLES:
        table = _subtable(arms, role)
        # Assign into the existing table rather than replacing it. A table's
        # trailing blank lines and the comment introducing the *next* section are
        # stored as that table's trivia, so replacing it silently eats them.
        table["left"] = ports[role].left
        table["right"] = ports[role].right

    _atomic_write(path, tomlkit.dumps(doc))


def write_cameras(
    cameras: dict[str, CameraDevice], path: Path | str = DEFAULT_CONFIG_PATH
) -> None:
    """Replace only ``[cameras.*]``, leaving ``[arms]`` and comments untouched."""
    from .layout import CAMERA_NAMES

    path = Path(path)
    missing = [name for name in CAMERA_NAMES if name not in cameras]
    if missing:
        raise ConfigError(f"write_cameras: missing cameras {missing}")

    doc = _load_document(path)
    root = doc.get("cameras")
    if not isinstance(root, dict):
        root = tomlkit.table(is_super_table=True)
        doc["cameras"] = root
    for name in CAMERA_NAMES:
        device = cameras[name]
        table = _subtable(root, name)  # in place — see write_arms
        table["path"] = device.path
        table["rotation"] = device.rotation
    # Drop any camera name no longer in use, so a rename cannot leave a stale
    # entry that validation would then reject as "unexpected".
    for stale in [key for key in root if key not in CAMERA_NAMES]:
        del root[stale]

    _atomic_write(path, tomlkit.dumps(doc))


def write_home(home: HomePose, path: Path | str = DEFAULT_CONFIG_PATH) -> None:
    """Replace only ``[home]``, leaving every other section and comment untouched.

    Same rules as :func:`write_arms`: the tables are mutated in place, because a
    replaced table takes the comment introducing the next section with it, and
    the file is written atomically.
    """
    path = Path(path)
    doc = _load_document(path)
    table = _subtable(doc, "home")
    for side in _ARM_SIDES:
        values = tomlkit.array()
        values.extend(round(value, 6) for value in getattr(home, side))
        table[side] = values

    _atomic_write(path, tomlkit.dumps(doc))


def _atomic_write(path: Path, text: str) -> None:
    """Write via a temp file + rename, so an interrupted write cannot truncate."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)
