"""What each serial port *is*, from USB identity alone. Opens nothing.

``/dev/ttyACM*`` numbering is assigned in enumeration order and reshuffles across
reboots and replugs, so the number is not an identity. Two things about a port
are stable, and neither requires touching the device:

**The adapter family** says whether a port is a follower or a leader. The two
arm kinds are driven by different hardware — the followers are Damiao DM4310 /
DM4340 chains behind a USB-to-CAN adapter, opened by ``follower.py`` as a raw
921600-baud serial port; the leaders are Dynamixel, opened through
``DynamixelMotorsBus`` — and the adapters have different USB vendor and product
IDs. This is what settled the contradiction the previous project left behind
(see the comment in ``dk1.toml``).

**The USB hub port** says which socket it is plugged into, and is what
``/dev/serial/by-path`` is built from.

What USB identity *cannot* say is which arm of a pair is the left one. That is a
fact about the room, not about the bus, and only :mod:`dk1lab.discovery.arms`
can settle it — by asking someone to unplug a cable.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

SERIAL_BY_PATH_DIR = Path("/dev/serial/by-path")

#: USB (vendor, product) -> the arm role that adapter belongs to.
#:
#: ``2e88:4603`` is the HDSC CDC device on Damiao's USB-to-CAN adapter; both of
#: ours report serial ``00000000050C``, so they are indistinguishable by serial
#: and must be told apart by hub port. ``1a86:55d3`` is the WCH CH343 serial
#: adapter in front of the Dynamixel leaders; those do carry distinct serials.
ADAPTERS: dict[tuple[int, int], str] = {
    (0x2E88, 0x4603): "follower",
    (0x1A86, 0x55D3): "leader",
}


@dataclass(frozen=True)
class SerialPort:
    """One serial port and everything knowable about it without opening it."""

    device: str
    vid: int | None = None
    pid: int | None = None
    serial_number: str | None = None
    location: str | None = None
    product: str | None = None

    @property
    def usb_id(self) -> str:
        """``vvvv:pppp``, or ``-`` for a non-USB port."""
        return f"{self.vid:04x}:{self.pid:04x}" if self.vid is not None else "-"

    @property
    def role(self) -> str | None:
        """``"follower"``, ``"leader"``, or ``None`` for an unrecognised adapter."""
        if self.vid is None or self.pid is None:
            return None
        return ADAPTERS.get((self.vid, self.pid))

    @property
    def hub_port(self) -> str:
        """The USB hub port from pyserial's location, e.g. ``5.1``."""
        # pyserial reports "1-5.1:1.0", i.e. "<bus>-<port chain>:<config>.<iface>".
        if not self.location:
            return "?"
        chain = self.location.split(":", 1)[0]
        return chain.split("-", 1)[1] if "-" in chain else chain


def is_usb(port: SerialPort) -> bool:
    """Whether this is a USB port at all, as opposed to a motherboard ``ttyS*``."""
    return port.vid is not None


def list_ports() -> list[SerialPort]:
    """Every USB serial port present right now, sorted by device name.

    The host's dozens of legacy ``/dev/ttyS*`` nodes are dropped: they carry no
    USB identity and none of them is ever an arm.
    """
    from serial.tools import list_ports as pyserial_ports

    ports = [
        SerialPort(
            device=p.device,
            vid=p.vid,
            pid=p.pid,
            serial_number=p.serial_number,
            location=p.location,
            product=p.product,
        )
        for p in pyserial_ports.comports()
    ]
    return sorted((p for p in ports if is_usb(p)), key=lambda p: p.device)


def group_by_role(ports: Iterable[SerialPort]) -> dict[str | None, list[SerialPort]]:
    """Split ports into ``follower``, ``leader`` and ``None`` (unrecognised)."""
    grouped: dict[str | None, list[SerialPort]] = {"follower": [], "leader": [], None: []}
    for port in ports:
        grouped.setdefault(port.role, []).append(port)
    return grouped


def stable_paths() -> dict[str, str]:
    """``{/dev/ttyACM0: /dev/serial/by-path/...}`` for every port that has one.

    The by-path node survives a reboot as long as the cable stays in its socket,
    which the ``ttyACM`` number does not. ``by-id`` is deliberately not used: the
    two follower adapters share a serial number, so it collapses to one entry for
    the pair — the same failure the cameras have.
    """
    if not SERIAL_BY_PATH_DIR.exists():
        return {}
    resolved: dict[str, str] = {}
    for link in sorted(SERIAL_BY_PATH_DIR.iterdir()):
        # The kernel exposes the same device under a "usbv2-" spelling too; the
        # plain "usb-" one is preferred so a port is never listed twice.
        if "-usbv" in link.name and link.name.replace("usbv2-", "usb-") in resolved.values():
            continue
        try:
            target = str(link.resolve())
        except OSError:  # pragma: no cover - a link vanishing mid-scan
            continue
        resolved.setdefault(target, str(link))
    return resolved


def describe(ports: Iterable[SerialPort], expected: dict[str, str] | None = None) -> list[str]:
    """Render the identity table, one line per port.

    Args:
        expected: optional ``{"<role>_<side>": device}`` from ``dk1.toml``, used
            to flag a configured port whose adapter family says it cannot be
            what the config claims.
    """
    # A list per device, not a single label: a config where two arms share a port
    # is exactly the mistake worth showing, and inverting the map would hide it.
    by_device: dict[str, list[str]] = {}
    for label, device in (expected or {}).items():
        by_device.setdefault(device, []).append(label)

    paths = stable_paths()
    lines = []
    for port in ports:
        role = port.role or "unknown adapter"
        line = f"  {port.device:14s} {port.usb_id}  hub {port.hub_port:6s} -> {role}"
        if labels := by_device.get(port.device):
            mismatched = [label for label in labels if label.rsplit("_", 1)[0] != port.role]
            mark = "   << does not match this adapter" if mismatched else ""
            line += f"   [dk1.toml: {', '.join(labels)}]{mark}"
        lines.append(line)
        if stable := paths.get(port.device):
            lines.append(f"      stable node: {stable}")
    return lines


def role_conflicts(assignments: dict[str, str], ports: Iterable[SerialPort]) -> list[str]:
    """Assignments whose port's adapter family contradicts the role claimed.

    A cross-check for :func:`dk1lab.discovery.arms.find_arms`: unplugging says
    which port belongs to the arm the operator was asked about, but nothing stops
    them from unplugging the wrong cable. USB identity knows follower adapters
    from leader ones, so a follower role landing on a Dynamixel adapter is caught
    before it is written — and before something tries to drive a Damiao CAN chain
    down a port that has none.

    Args:
        assignments: ``{"<role>_<side>": device}``, as ``DK1Config.arm_ports``.
        ports: what :func:`list_ports` returned.

    Returns:
        Human-readable descriptions, one per conflict. Empty means consistent.
        A port whose adapter is unrecognised is not a conflict — an unfamiliar
        adapter should not block a discovery run that is otherwise fine.
    """
    roles = {port.device: port.role for port in ports}
    conflicts = []
    for label, device in assignments.items():
        expected = label.rsplit("_", 1)[0]
        actual = roles.get(device)
        if actual is not None and actual != expected:
            conflicts.append(
                f"{label} was assigned {device}, but that port is a {actual} adapter "
                f"({next(p.usb_id for p in ports if p.device == device)}), not a {expected} one"
            )
    return conflicts
