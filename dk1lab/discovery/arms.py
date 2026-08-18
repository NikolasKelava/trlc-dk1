"""Identify which serial port belongs to which arm, by unplugging it.

There is no way to tell the four arms apart from the port alone — the followers
are identical CAN-to-USB adapters and the leaders are identical Dynamixel
adapters — so identification is physical: list the ports, ask the operator to
unplug one arm, list again, and see which port vanished.

The comparison logic is separated from the prompting so it can be tested against
a scripted device list with no hardware present.
"""

from __future__ import annotations

import platform
import time
from collections.abc import Callable, Iterable
from pathlib import Path

from ..config import ArmPorts

#: (role, side) in the order the operator is asked, matching ``[arms.*]``.
ARM_SEQUENCE: tuple[tuple[str, str], ...] = (
    ("follower", "left"),
    ("follower", "right"),
    ("leader", "left"),
    ("leader", "right"),
)


class DiscoveryError(Exception):
    """Raised when a port cannot be identified unambiguously."""


def list_serial_ports() -> set[str]:
    """Every serial port currently present on this machine."""
    if platform.system() == "Windows":
        from serial.tools import list_ports

        return {port.device for port in list_ports.comports()}
    # The DK1 arms all enumerate as USB CDC-ACM devices. Restricting to ttyACM*
    # rather than every tty* keeps unrelated hardware (and the host's own serial
    # consoles) out of the comparison, so an incidental device appearing between
    # the two listings cannot be mistaken for the arm.
    return {str(p) for p in Path("/dev").glob("ttyACM*")}


def detect_removed(before: Iterable[str], after: Iterable[str], *, what: str) -> str:
    """The single port present in ``before`` but not ``after``.

    Raises:
        DiscoveryError: if no port disappeared, or more than one did — both of
            which would otherwise write a wrong port into the config.
    """
    removed = sorted(set(before) - set(after))
    if len(removed) == 1:
        return removed[0]
    if not removed:
        raise DiscoveryError(
            f"No serial port disappeared when unplugging the {what}. "
            f"Was the right cable unplugged? Nothing has been written."
        )
    raise DiscoveryError(
        f"{len(removed)} serial ports disappeared when unplugging the {what}: "
        f"{removed}. Unplug exactly one arm at a time. Nothing has been written."
    )


def find_arms(
    *,
    prompt: Callable[[str], str] = input,
    announce: Callable[[str], None] = print,
    lister: Callable[[], set[str]] = list_serial_ports,
    sleep: Callable[[float], None] = time.sleep,
    settle_s: float = 0.5,
    replug_s: float = 1.0,
) -> dict[str, ArmPorts]:
    """Walk the operator through identifying all four arms.

    Every dependency is injected so the whole flow can be driven by a test with a
    scripted device list.

    Returns:
        ``{"follower": ArmPorts(...), "leader": ArmPorts(...)}``, ready for
        :func:`dk1lab.config.write_arms`.
    """
    found: dict[str, dict[str, str]] = {"follower": {}, "leader": {}}

    for role, side in ARM_SEQUENCE:
        what = f"{side} {role} arm"
        before = lister()
        prompt(f"Unplug the USB cable of the {what}, then press Enter...")
        sleep(settle_s)
        after = lister()
        port = detect_removed(before, after, what=what)
        announce(f"  {what:24s} -> {port}")
        found[role][side] = port
        prompt("Plug it back in and press Enter to continue...")
        sleep(replug_s)

    assigned: dict[str, str] = {}
    for role, sides in found.items():
        for side, port in sides.items():
            label = f"{side} {role}"
            if port in assigned:
                raise DiscoveryError(
                    f"{label} and {assigned[port]} both resolved to {port}. "
                    f"The device probably had not finished re-enumerating. "
                    f"Nothing has been written — run `dk1 find arms` again."
                )
            assigned[port] = label

    return {role: ArmPorts(left=sides["left"], right=sides["right"]) for role, sides in found.items()}
