"""What the machine was doing in the second before it froze.

Twice — 2026-08-25 17:58 and 2026-08-26 11:23 — this machine stopped dead
mid-session. Both times the kernel journal simply ends: no OOM kill, no oops, no
shutdown, no thermal message. A hang that leaves nothing behind cannot be
diagnosed from logs that are written *after* it, so this writes them before:

one line of JSON per second, ``fsync``-ed, holding the numbers that distinguish
the candidate causes from each other. When the machine comes back, the **last
line** is the state it was in.

What is sampled, and what each one would show:

| source | reads | a freeze caused by |
| --- | --- | --- |
| ``corsairpsu`` hwmon | total watts, +12 V volts and amps, PSU temperature | a 5090 transient tripping the PSU — the 12 V rail sags or the draw spikes |
| ``coretemp`` hwmon | CPU package temperature | thermal |
| NVML / ``nvidia-smi`` | GPU temperature, power, utilisation, memory, throttle reasons | the GPU, and whether it was throttling |
| ``/proc`` | load, available memory, IO pressure | memory or IO |

Everything comes from sysfs and ``/proc`` — no root, no subprocess — except the
GPU, which goes through NVML when ``pynvml`` is importable and falls back to one
``nvidia-smi`` call per sample when it is not. A sample costs a fraction of a
millisecond that way and about 30 ms through the fallback, on a thread that is
not the control loop.

**This is an instrument, not a fix.** It records; it does not prevent. The
freezes it was written for turned out to be platform firmware and were fixed by
a BIOS update on 2026-08-27 (`docs/CRASH.md`, closed); it is kept for the next
unexplained fault, not for that one.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: One second. Fast enough to catch a thermal or power excursion, slow enough
#: that the file stays readable by eye for a session of a few hours.
DEFAULT_INTERVAL_S = 1.0

DEFAULT_TELEMETRY_DIR = Path("logs")

_HWMON = Path("/sys/class/hwmon")


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _read_number(path: Path, scale: float = 1.0) -> float | None:
    text = _read(path)
    if text is None:
        return None
    try:
        return int(text) * scale
    except ValueError:
        return None


def find_hwmon(name: str) -> Path | None:
    """The hwmon directory whose ``name`` file says ``name``, if there is one.

    Looked up by name rather than by number because ``hwmon4`` is the PSU today
    and could be the NIC after a reboot — the numbering follows probe order.
    """
    try:
        candidates = sorted(_HWMON.iterdir())
    except OSError:  # pragma: no cover - no hwmon at all
        return None
    for directory in candidates:
        if _read(directory / "name") == name:
            return directory
    return None


@dataclass
class Sources:
    """Where the numbers come from on this machine, resolved once.

    Resolved at start rather than per sample: the paths do not move while the
    machine is up, and a freeze is not the moment to be walking sysfs.
    """

    psu: Path | None = None
    cpu: Path | None = None
    nvml: Any = None
    gpu_handle: Any = None
    #: The `nvidia-smi` binary, used only when NVML is not importable. A fork a
    #: second is affordable on a sampling thread and beats having no GPU numbers
    #: at all, which is the reading that most needs to exist here.
    smi: str | None = None
    notes: list[str] = field(default_factory=list)

    @classmethod
    def detect(cls) -> Sources:
        found = cls(psu=find_hwmon("corsairpsu"), cpu=find_hwmon("coretemp"))
        if found.psu is None:
            found.notes.append("no corsairpsu hwmon: no PSU power or 12 V rail")
        if found.cpu is None:
            found.notes.append("no coretemp hwmon: no CPU package temperature")
        try:
            import pynvml

            pynvml.nvmlInit()
            found.nvml = pynvml
            found.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception as exc:  # noqa: BLE001 - NVML absent is a fact, not a fault
            found.notes.append(f"no NVML ({exc}); falling back to nvidia-smi")
            found.smi = shutil.which("nvidia-smi")
            if found.smi is None:
                found.notes.append("no nvidia-smi either: no GPU readings")
        return found

    def sample(self) -> dict[str, Any]:
        """One reading of everything available. Never raises."""
        now = time.time()
        row: dict[str, Any] = {
            "t": round(now, 3),
            "clock": datetime.fromtimestamp(now).strftime("%H:%M:%S.%f")[:-3],
        }
        row.update(self._system())
        if self.psu is not None:
            row.update(self._psu(self.psu))
        if self.cpu is not None:
            row["cpu_c"] = _read_number(self.cpu / "temp1_input", 1e-3)
        if self.gpu_handle is not None:
            row.update(self._gpu())
        elif self.smi is not None:
            row.update(self._gpu_smi())
        return row

    # -- the individual sources ---------------------------------------------- #

    @staticmethod
    def _system() -> dict[str, Any]:
        out: dict[str, Any] = {}
        try:
            out["load1"] = os.getloadavg()[0]
        except OSError:  # pragma: no cover
            pass
        meminfo = _read(Path("/proc/meminfo")) or ""
        for line in meminfo.splitlines():
            if line.startswith("MemAvailable:"):
                out["mem_avail_gb"] = round(int(line.split()[1]) / 1e6, 2)
                break
        pressure = _read(Path("/proc/pressure/io"))
        if pressure:
            # "some avg10=1.23 avg60=... " — avg10 is the share of the last ten
            # seconds in which at least one task was stalled on IO.
            first = pressure.splitlines()[0]
            for part in first.split():
                if part.startswith("avg10="):
                    out["io_stall10"] = float(part.removeprefix("avg10="))
        return out

    @staticmethod
    def _psu(psu: Path) -> dict[str, Any]:
        return {
            "psu_w": _read_number(psu / "power1_input", 1e-6),
            "psu_12v_w": _read_number(psu / "power2_input", 1e-6),
            "psu_12v_v": _read_number(psu / "in1_input", 1e-3),
            "psu_12v_a": _read_number(psu / "curr2_input", 1e-3),
            "psu_in_v": _read_number(psu / "in0_input", 1e-3),
            "psu_c": _read_number(psu / "temp1_input", 1e-3),
        }

    def _gpu(self) -> dict[str, Any]:
        nvml, handle = self.nvml, self.gpu_handle
        out: dict[str, Any] = {}
        for key, call in (
            ("gpu_c", lambda: nvml.nvmlDeviceGetTemperature(handle, 0)),
            ("gpu_w", lambda: nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0),
            ("gpu_util", lambda: nvml.nvmlDeviceGetUtilizationRates(handle).gpu),
            ("gpu_mem_gb", lambda: nvml.nvmlDeviceGetMemoryInfo(handle).used / 1e9),
            ("gpu_clock", lambda: nvml.nvmlDeviceGetClockInfo(handle, 0)),
            ("gpu_throttle", lambda: nvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)),
        ):
            try:
                value = call()
            except Exception:  # noqa: BLE001 - one unsupported query is not a failure
                continue
            out[key] = round(value, 2) if isinstance(value, float) else value
        return out


    _SMI_FIELDS = (
        ("gpu_c", "temperature.gpu"),
        ("gpu_w", "power.draw"),
        ("gpu_util", "utilization.gpu"),
        ("gpu_mem_gb", "memory.used"),
        ("gpu_clock", "clocks.sm"),
    )

    def _gpu_smi(self) -> dict[str, Any]:
        """The same readings through ``nvidia-smi``, for a venv without NVML."""
        query = ",".join(name for _key, name in self._SMI_FIELDS)
        try:
            result = subprocess.run(
                [self.smi, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, check=True,
            )
        except (OSError, subprocess.SubprocessError):
            return {}
        parts = [p.strip() for p in result.stdout.strip().splitlines()[0].split(",")]
        out: dict[str, Any] = {}
        for (key, _name), text in zip(self._SMI_FIELDS, parts, strict=False):
            try:
                value = float(text)
            except ValueError:
                continue
            # nvidia-smi reports memory in MiB; the NVML path reports bytes, and
            # the file has to mean one thing.
            out[key] = round(value / 1024, 2) if key == "gpu_mem_gb" else value
        return out


class Telemetry:
    """A thread writing one JSON line a second, each one on the disk.

    Started by the commands that drive the arms and by ``dk1 doctor watch``.
    Stopping is cooperative and immediate; the file is closed either way.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        interval_s: float = DEFAULT_INTERVAL_S,
        context: dict[str, Any] | None = None,
    ) -> None:
        self.path = Path(path)
        self.interval_s = float(interval_s)
        self.context = dict(context or {})
        self.sources = Sources()
        self.samples = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._handle: Any = None
        #: Set by whoever owns the control loop, so a sample says how long ago
        #: the last tick was. A freeze that starts in our loop and one that
        #: starts under it look different here.
        self.marker: Any = None

    def start(self) -> Path:
        if self._thread is not None:
            return self.path
        self.sources = Sources.detect()
        for note in self.sources.notes:
            logger.info("telemetry: %s", note)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")
        self._write({"t": round(time.time(), 3), "event": "start", **self.context})
        self._thread = threading.Thread(target=self._run, name="telemetry", daemon=True)
        self._thread.start()
        logger.info("telemetry -> %s every %.1f s", self.path, self.interval_s)
        return self.path

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2 * self.interval_s + 1.0)
            self._thread = None
        if self._handle is not None:
            self._write({"t": round(time.time(), 3), "event": "stop", "samples": self.samples})
            try:
                self._handle.close()
            except OSError:  # pragma: no cover
                pass
            self._handle = None

    def mark(self, **fields: Any) -> None:
        """Put an event on the timeline — an episode start, a keep, a fault.

        The point of the file is reading backwards from the freeze; an event
        line is what says *what the operator was doing* at that moment.
        """
        self._write({"t": round(time.time(), 3), **fields})

    # -- internals ----------------------------------------------------------- #

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                row = self.sources.sample()
                if self.marker is not None:
                    row["tick_age_s"] = round(time.perf_counter() - float(self.marker), 3)
                self._write(row)
                self.samples += 1
            except Exception as exc:  # noqa: BLE001 - an instrument must not kill a run
                logger.warning("telemetry sample failed: %s", exc)
            self._stop.wait(self.interval_s)

    def _write(self, row: dict[str, Any]) -> None:
        handle = self._handle
        if handle is None:
            return
        try:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        except (OSError, ValueError) as exc:  # pragma: no cover - a full or closed file
            logger.warning("telemetry write failed: %s", exc)


def read(path: Path | str) -> list[dict[str, Any]]:
    """Every sample in a telemetry file. A truncated last line is dropped.

    Truncation is expected: the file's whole purpose is to be interrupted.
    """
    rows: list[dict[str, Any]] = []
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return rows
    for line in text.splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def summary(rows: list[dict[str, Any]]) -> list[str]:
    """What the run looked like, and what the last second before it ended did.

    The extremes matter more than the averages here: a freeze is an excursion,
    and the summary exists so nobody has to read four thousand JSON lines.
    """
    if not rows:
        return ["no samples"]
    lines: list[str] = []
    samples = [r for r in rows if "clock" in r]
    if samples:
        lines.append(f"{len(samples)} samples, {samples[0]['clock']} -> {samples[-1]['clock']}")
    for key, label, unit in (
        ("psu_w", "PSU total", "W"),
        ("psu_12v_v", "+12 V rail", "V"),
        ("gpu_w", "GPU power", "W"),
        ("gpu_c", "GPU temp", "C"),
        ("cpu_c", "CPU temp", "C"),
        ("mem_avail_gb", "memory free", "GB"),
        ("io_stall10", "IO stall", ""),
    ):
        values = [r[key] for r in samples if isinstance(r.get(key), (int, float))]
        if not values:
            continue
        last = values[-1]
        lines.append(
            f"{label:12s} min {min(values):8.1f}  max {max(values):8.1f}  "
            f"last {last:8.1f} {unit}"
        )
    events = [r for r in rows if "event" in r]
    if events:
        lines.append(f"events: {', '.join(str(e['event']) for e in events[-6:])}")
    return lines


__all__ = [
    "DEFAULT_INTERVAL_S",
    "DEFAULT_TELEMETRY_DIR",
    "Sources",
    "Telemetry",
    "find_hwmon",
    "read",
    "summary",
]
