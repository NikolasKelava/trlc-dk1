# CRASH.md — the machine freezes, and it is not solved

**Status: OPEN.** Two scored sessions and at least one teleoperation run have
ended with the whole machine dead — not the process, the machine. Nothing in
this repository has been shown to cause it and nothing in it has been shown not
to. **Do not record the ~100 teleoperation demonstrations of `STUDY.md` Phase 3
until this is understood**: those episodes cannot be re-run cheaply, and a
freeze in the middle of them costs a day of somebody's hands.

This file is the standing account and the brief for whoever picks it up. The
first section is what a next session should read as its instructions; the rest
is evidence, so nothing gets re-measured.

---

## The brief, for the next session

You are picking up an unsolved hard-freeze on the machine that runs this cell.
Read `CLAUDE.md` for the project, then this file. Then:

1. **Do not run anything that moves the arms without asking Nikolas.** The
   investigation below is designed so that most of it needs no motor at all.
2. **Work by elimination, cheapest first** (§ *The plan*). The point is to find
   which subsystem is involved — GPU, USB cameras, CAN adapters, or none of
   them — not to guess at a fix.
3. **Make the next freeze leave evidence** before provoking one (§ *Make it
   leave a trace*). Everything in that section except `sysrq` and `kdump` is
   already implemented and on by default; the kernel-side parts need `sudo` and
   are Nikolas's to run — hand him the commands.
4. When a run does freeze, the first thing to read is
   `uv run dk1 doctor report`, which prints the last second the machine was
   alive, followed by `journalctl -b -1 -k -n 200`.
5. Write what you find **into this file**, dated, including the hypotheses that
   turned out wrong. Half of the value here is not re-testing them.

Two things Nikolas suspects and neither is ruled out: **something in LeRobot**,
and **a thread we share with the desktop (Plasma/KDE)**. Take both seriously,
but note the shape of the problem: ordinary user-space code — Python threads,
CPU load, a busy GUI — cannot normally take a Linux machine down. What can is a
*kernel driver*, and this workload leans on three of them at once: `nvidia`
(CUDA and NVENC), `uvcvideo` (three 1280x720 MJPG streams on one controller),
and the USB-CAN adapters. A user-space bug that *triggers* a driver bug looks
exactly like this. So the question to answer first is **which driver**, and the
way to answer it is to run each of them without the others.

---

## What happened

| when | what was running | how it ended |
| --- | --- | --- |
| 2026-08-25 ~17:58 | `dk1 policy session --study A0`, 8th attempt, dataset recording, SVT-AV1 on the CPU | machine dead, hard reset |
| 2026-08-26 ~11:23 | `dk1 policy session --study A0`, scene 3 attempt 1, dataset recording, NVENC | machine dead, hard reset |
| earlier, once or twice | `dk1 teleop`, "after a few minutes" | machine dead |

The teleoperation cases matter more than their number suggests: **teleop loads no
model and uses no CUDA**. If the freeze happens there too, the GPU compute path
is not necessary for it — cameras, CAN and the desktop are what the two
workloads share. (Confirm this: was `--cameras` on? Nikolas, if you remember
whether the teleop freezes had cameras attached, say so — it splits the search
in half.)

### What the logs say: nothing, and that is a finding

For both crashes the systemd journal for that boot **simply stops** mid-second,
with no shutdown records after it. Checked and absent:

- no OOM kill, no `Killed process`, no `Out of memory` (memory was 8% used);
- no kernel oops, panic, `BUG:` or `WARNING:`;
- no `NVRM`/`Xid` message, no thermal or throttling message;
- no USB disconnect, no `xhci` error;
- no `mce`, no EDAC, no hardware error report.

`sar` was sampling every ten minutes and shows nothing unusual either: CPU 4–8%
average, memory 8%, IO 106 MB/s while the PNG cache was being written and near
zero in the last interval.

A journal that ends mid-line means the kernel never got to write anything —
consistent with a **hard hang** (interrupts dead, or a CPU lock-up with no
watchdog) or with **power being cut** (PSU protection tripping). Those two are
distinguished by whether the machine's fans keep spinning; ask Nikolas, he
watched it happen twice.

### The rig, for whoever is reading this cold

- **RTX 5090**, driver 580.173.02, kernel **7.0.0-30-generic**, Plasma/KDE on X11.
- **Corsair HX1200i** PSU, and it is monitored: `corsairpsu` hwmon exposes total
  watts, the +12 V rail's volts and amps, and PSU temperature. This is the
  single most useful sensor on the machine for this problem, and it is now
  sampled every second.
- 3x Innomaker U30CAM 1280x720 MJPG cameras on one USB controller, 4x USB-CAN
  adapters (followers `2e88:4603`, leaders `1a86:55d3`).
- 62 GB RAM, `powersave` governor, no swap pressure.

---

## Make it leave a trace

**Already implemented and on by default** (2026-08-26), so the next freeze is
not another blank page:

| | |
| --- | --- |
| `logs/<time>-<what>.log` | every log record from `dk1lab` at DEBUG and `lerobot` at INFO, **fsynced line by line**, with tracebacks. `--no-log` turns it off. `dk1lab/logs.py` |
| `logs/<time>-<what>.jsonl` | PSU total power, +12 V volts and amps, PSU temperature, CPU package temperature, GPU temperature/power/utilisation/memory/clock, load, free memory, IO stall — **once a second, each line fsynced**. `--no-telemetry` turns it off. `dk1lab/telemetry.py` |
| `dk1 doctor watch` | the same sampler, standalone, for anything that is not ours: `uv run dk1 doctor watch --label teleop-test` |
| `dk1 doctor report` | reads the newest file back: extremes, whether it ends with a `stop` event, and the last samples. **A file with no `stop` event ends where the machine did** |

The telemetry also carries `episode_start` / `episode_end` events, so a freeze
can be placed against what the operator was doing.

**Still to do, and it needs `sudo` — hand these to Nikolas:**

```bash
# 1. Magic SysRq, so a frozen machine can still be asked what it is doing.
#    After a freeze: Alt+SysRq+w (blocked tasks), then Alt+SysRq+l (CPU stacks),
#    then Alt+SysRq+s (sync) and Alt+SysRq+b (reboot). If the keyboard responds
#    at all, the kernel is alive and it is a userspace/GPU hang, not a lock-up.
#    That single fact splits the search in half.
sudo sysctl -w kernel.sysrq=1

# 2. Turn a silent lock-up into a panic that gets recorded.
sudo sysctl -w kernel.hung_task_panic=1 kernel.hung_task_timeout_secs=60
sudo sysctl -w kernel.panic_on_oops=1 kernel.panic=30

# 3. Somewhere for a panic to be written. pstore first (needs no second machine):
ls /sys/fs/pstore                 # empty now; a panic would leave dmesg-* here
# and if pstore does not work on this board, netconsole to a laptop is the
# fallback:  sudo modprobe netconsole netconsole=6666@<this-ip>/,6666@<laptop-ip>/<mac>

# 4. Watch the GPU independently of anything we write, at 1 Hz, to a file:
nvidia-smi --query-gpu=timestamp,temperature.gpu,power.draw,clocks.sm,utilization.gpu \
  --format=csv -l 1 > ~/gpu.csv
```

---

## The plan

Ordered so the cheap, no-motor experiments come first. Each step is a *bisection*
of the suspects: GPU compute, NVENC, the three cameras, the CAN adapters, the
desktop. Run each until it either freezes or has survived comfortably longer than
the ~20 minutes both real freezes took.

1. **Cameras alone, no arms, no GPU.** Three 720p MJPG streams, read at 30 Hz,
   for an hour. `dk1 teleop --dry-run` will not do it; write a small script that
   opens the three `CroppedOpenCVCamera`s and reads them, or run
   `dk1 study photo` in a loop. Watch with `dk1 doctor watch --label cameras`.
   *If this freezes, it is `uvcvideo`/`xhci` and nothing to do with the policy.*
2. **The GPU alone.** `dk1 policy smoke --steps 500` in a loop, or the sim
   (`dk1 policy run --sim --no-view --duration 600`): full inference, no camera,
   no motor. *If this freezes, it is CUDA/NVENC or power.*
3. **NVENC alone.** Encode the leftover PNG cache repeatedly
   (`study/rollouts/A0-crashed/images/...`) with `h264_nvenc`. Ten minutes of
   that is far more NVENC than a session does.
4. **CAN alone.** Arms connected, energised, nothing commanded — this needs
   Nikolas's permission and his hand on the e-stop.
5. **Only then the whole thing**, with the telemetry running, and read
   `dk1 doctor report` afterwards whatever happens.

Cheap mitigations worth trying *as diagnostics*, one at a time:

- **Cap the GPU:** `sudo nvidia-smi -pl 400`. The 5090's transient spikes are a
  known cause of PSU protection trips on otherwise adequate supplies. If capping
  makes the freezes stop, the answer is power, and the fix is a cable/PSU matter
  rather than a software one.
- **Disable USB autosuspend** for the camera hub, in case a suspend/resume race
  in `uvcvideo` is involved: `sudo sysctl -w kernel.printk=7` and check
  `/sys/bus/usb/devices/*/power/control`.
- **Try a plain session on TTY without Plasma** (Ctrl+Alt+F3, log in, run it).
  This is the cheapest way to test Nikolas's KDE suspicion, and it costs one
  session.

Things to *record* while doing all this: how long each ran, and what
`dk1 doctor report` said afterwards. Add them to the table below.

| date | experiment | duration | outcome |
| --- | --- | --- | --- |
| | | | |

---

## The other fault of 2026-08-26, and it is ours

The same session **recorded nothing at all**, and that is separate from the
freeze. The dataset directory holds one episode's PNG cache (7.8 GB), an
unfinished data file, `total_episodes: 0`, and three empty temporary directories
timestamped 11:08:47 — the signature of `save_episode` raising inside the video
encode. Five more attempts were then run and scored against a dataset that stayed
empty.

Three things have been fixed since (all in `dk1lab/dataset.py` and
`dk1lab/cli/policy_cmds.py`):

- the traceback now goes to the session log, where it survives a reset;
- a failed write is announced **in red, after that attempt**, saying the score
  will have no frames behind it — it is not one line among many any more;
- the episode buffer is cleared after a failure, so one bad encode no longer
  poisons every episode after it.

**The cause of that encode failure is still unknown.** Re-running the exact same
encode afterwards, on the same 3 538 real frames with the same encoder settings,
**succeeded in 48.8 s**. So it is not the codec settings and not the frame count.
What differs in the live case: three cameras encoding at once, and a CUDA
context holding MolmoAct2 while NVENC starts. First suspicion for the next
session — **concurrent NVENC sessions while CUDA is resident** — and the log file
will now name the exception.

---

## The starved ticks

Every episode of the 2026-08-26 session reported **7–15 starved ticks** — ticks
where the chunk queue had nothing to serve and the arms held their last command.
Out of ~3 550 ticks that is 0.2–0.4%, and the motion looked fine, but the
comparable measurement in `DIAGNOSTICS.md` § *The fix* is **zero starved ticks
over 335 chunks** — measured **without dataset recording**.

The obvious suspect is the difference between those two runs: the async image
writer, compressing and writing ~90 PNGs a second (~40 MB/s) while the inference
worker wants the CPU. Testable without the arms in the sim, and worth an entry
in `DIAGNOSTICS.md` once measured:

```
dk1 policy run --sim --no-view --duration 300                     # baseline
dk1 policy run --sim --no-view --duration 300 --record-dataset --dataset-dir /tmp/ds1
dk1 policy run --sim --no-view --duration 300 --record-dataset --dataset-dir /tmp/ds2 --stream-video
```

If recording is what starves the queue, `--stream-video` (no PNG cache at all)
is the interesting comparison, and its own cost is measured in
`DIAGNOSTICS.md` § *Recording: the episode that took minutes to save*.

---

## What is already ruled out

Do not spend time on these again without a reason:

- **Memory.** 8% used, no OOM killer entry, 62 GB installed, no swap thrash.
- **Disk space.** 1.3 TB free at the time.
- **The recorder's own durability** as a *cause* — it is a victim. The data-loss
  side of the first crash is fixed and tested (`DIAGNOSTICS.md` § *Recording: the
  crash that ate seven episodes*).
- **The codec settings.** The exact failing encode succeeds when re-run.
- **CPU thermal throttling**, as far as ten-minute `sar` averages can say: 4–8%
  average CPU. Not conclusive at one-second resolution, which is why the
  telemetry now exists.
