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

1. **First message of the session: ask Nikolas to enable the kernel traces**
   in § *Make it leave a trace* — Magic SysRq above all. They need `sudo`, they
   are his to run, and he has asked to be prompted. Without SysRq the next
   freeze answers nothing; with it, one keystroke says whether the kernel was
   still alive.
2. **Do not run anything that moves the arms without asking Nikolas.** The
   investigation below is designed so that most of it needs no motor at all.
3. **Work by elimination, cheapest first** (§ *The plan*). The point is to find
   which subsystem is involved — GPU, USB cameras, CAN adapters, or none of
   them — not to guess at a fix.
4. **Make the next freeze leave evidence** before provoking one (§ *Make it
   leave a trace*). Everything in that section except `sysrq` and `kdump` is
   already implemented and on by default; the kernel-side parts need `sudo` and
   are Nikolas's to run — hand him the commands.
5. When a run does freeze, the first thing to read is
   `uv run dk1 doctor report`, which prints the last second the machine was
   alive, followed by `journalctl -b -1 -k -n 200`.
6. Write what you find **into this file**, dated, including the hypotheses that
   turned out wrong. Half of the value here is not re-testing them.

Two things Nikolas suspects and neither is ruled out: **something in LeRobot**,
and **a thread we share with the desktop (Plasma/KDE)**. Take both seriously,
but note the shape of the problem: ordinary user-space code — Python threads,
CPU load, a busy GUI — cannot normally take a Linux machine down. What can is a
*kernel driver*, and this workload leans on three at once: `nvidia`, `uvcvideo`
(three 1280x720 MJPG streams), and the USB-CAN adapters — the last two on one
xHCI controller. A user-space bug that *triggers* a driver bug looks exactly
like this. The question to answer first is **which driver**, and the way to
answer it is to run each without the others.

**And ask rather than infer.** This file has already had to be corrected once
because a session guessed at which conditions a crash happened under instead of
asking Nikolas. He is at the machine and he remembers; the table above is only
worth as much as its accuracy.

---

## What happened

| when | what was running | Rerun | model / CUDA | how it ended |
| --- | --- | --- | --- | --- |
| 2026-08-25 ~17:58 | `dk1 policy session --study A0`, 8th attempt, dataset recording, SVT-AV1 on the CPU | **no** | yes | machine dead, hard reset |
| 2026-08-26 ~11:23 | `dk1 policy session --study A0`, scene 3 attempt 1, dataset recording, NVENC | **no** | yes | machine dead, hard reset |
| earlier, once or twice | `dk1 teleop`, "after a few minutes" | **yes** | no | machine dead |
| 2026-08-26 | `dk1 teleop` with the Rerun viewer open | **yes** | no | froze within a few minutes: teleoperation never worked at all, everything down, the display showing a stale image |

**Roughly twenty minutes into a session; a few minutes into teleoperation.**
The teleop case is the cheaper reproducer of the two, and it is the one to use —
once it can be reproduced without energising anything (see § *The plan*).

### What is common to all four, and what is not

This is the part to reason from, and an earlier version of this file got it
wrong by guessing instead of asking.

| | in the session crashes | in the teleop crashes | common? |
| --- | --- | --- | --- |
| the three USB cameras streaming 1280x720 MJPG | yes | yes (teleop attaches them by default) | **yes** |
| the four USB-CAN adapters, arms energised | yes | yes | **yes** |
| one xHCI controller carrying both | yes | yes | **yes** |
| Plasma/KDE on X11, kernel 7.0.0-30, nvidia 580.173.02 | yes | yes | **yes** |
| the Rerun viewer (wgpu/Vulkan) | **no** | yes | no |
| CUDA, MolmoAct2 resident, NVENC | yes | **no** | no |
| the dataset recorder, PNG cache, video encoding | yes | **no** | no |

So **neither GPU consumer is common** — the sessions had CUDA and no Rerun, the
teleop runs had Rerun and no CUDA. Either the GPU is involved through something
both share (the display stack itself, the driver, the card), or it is not
involved at all and the common factor is elsewhere. What *is* common to every
freeze: **the USB tree** — three 720p MJPG streams and four CDC-ACM CAN adapters
on one xHCI controller — **the desktop**, and **this kernel and driver pair**.

A detail that points the same way: the cameras produce a steady trickle of
`Corrupt JPEG data: N extraneous bytes before marker 0xd7`. That is benign in
itself, and it also means the USB video stream is *marginal* rather than clean.

**Ask Nikolas before building on this:** were the cameras attached in the teleop
runs that froze (`--cameras` is on by default, so probably yes — but "probably"
is what produced the last wrong turn in this file)?

### What it looked like, and what that rules out (2026-08-26, from Nikolas)

**The fans kept spinning. The screen updated once more — one warning line — and
then nothing: mouse and keyboard moved nothing, and only the reset button
worked.**

That is not a power cut. Power was still being delivered and the machine was
still executing something long enough to paint a line of text. It is the
signature of a **GPU / display-driver wedge**: the last thing the console
manages to print, then a desktop that never repaints. Whether the *kernel* was
also dead is the next question, and Magic SysRq answers it — if Alt+SysRq+b
reboots the machine, the kernel was alive and it was the GPU stack; if nothing
happens at all, the kernel itself was gone.

So: **the PSU theory is demoted** (keep the +12 V column, it is free, but stop
leading with it) and **the GPU driver is promoted**.

### The USB tree, as it is wired today

```
Bus 002 (xhci, 5 Gbps)              Bus 001 (xhci, 480M/12M)
  hub  -> camera, camera              hub -> CAN adapter, CAN adapter
  hub  -> camera                      hub -> CAN adapter, CAN adapter
```

Three Innomaker U30CAM (`0bda:5883`) behind two USB-3 hubs on bus 2; four
CDC-ACM adapters (`1a86:55d3` leaders, `2e88:4603` followers) behind two hubs on
bus 1. Both root hubs belong to the same xHCI controller. Every freeze so far
had all seven devices active.

Worth trying as a diagnostic, in this order: **cameras on a different
controller** (a PCIe USB card, or the rear ports if these are front-panel),
**cameras at 640x360** instead of 720p (a quarter of the bandwidth — it is
`[capture.policy]` in `dk1.toml`, and it changes what the policy sees, so it is
a test and not a setting to leave), and **USB autosuspend off**.

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

A journal that ends mid-line means the kernel never got to write anything to
disk. Combined with what Nikolas saw — fans still running, one last line painted
— the reading is a **hard hang** rather than a power loss: something kept the
kernel from ever reaching the journal again.

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

Ordered so the cheap, no-motor experiments come first, and led by what every
freeze had in common rather than by what any one of them had. Each step is a
*bisection*: the cameras, the display stack, GPU compute, NVENC, the CAN
adapters.

Every step below runs until it either freezes or has survived **comfortably
longer than the workload it stands in for** — half an hour for a teleop
substitute, an hour for a session substitute. Run `dk1 doctor watch --label
<step>` alongside each one, in a second terminal, and read
`dk1 doctor report` afterwards either way.

1. **The cameras, alone, with Rerun.** The safest thing that resembles the
   fastest reproducer: three 720p MJPG streams read at 30 Hz and logged to a
   Rerun viewer, with **no arms, no model, no CUDA**. If teleop-with-Rerun
   freezes in minutes and this does too, the arms are out of it and the
   reproducer costs nothing to run again. Write it as a small script against
   `dk1lab.crop.CroppedOpenCVCamera` and `rr.log`; do not use `dk1 teleop`,
   which needs the arms.
2. **The cameras, alone, without Rerun.** Same script, no viewer. This is the
   one that separates "USB video" from "USB video plus the display stack".
3. **The cameras at 640x360**, if either of the above freezes: a quarter of the
   USB bandwidth, same code path.
4. **The GPU alone.** `dk1 policy run --sim --no-view --duration 1800`: full
   inference at 30 Hz, no camera, no motor, no viewer.
5. **NVENC alone.** Encode the leftover PNG cache in
   `study/rollouts/A0-crashed/images/` in a loop with `h264_nvenc` — far more
   NVENC in ten minutes than a session does in an hour.
6. **The CAN adapters alone.** Arms connected and energised, nothing commanded.
   **Needs Nikolas's permission and his hand near the e-stop.**
7. **Teleoperation with Rerun**, the known reproducer, only if 1–6 all survive.
8. **A short row on the real cell**, to prove the recorder end to end before
   another nine-attempt session — three attempts, one per scene:

   ```
   dk1 policy session --study TEST --attempts 1 --profile common --duration 60 \
     --record-dataset --dataset-dir study/rollouts/TEST --vcodec libsvtav1
   ```

   Then `dk1 doctor report`, and check the dataset opens:
   `python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset as D; d=D.resume('dk1/test', root='study/rollouts/TEST'); print(d.num_episodes, d.num_frames)"`.
   `--vcodec libsvtav1` is deliberate here: slow, and the only encoder that has
   ever produced readable video on this cell.

Cheap mitigations worth trying *as diagnostics*, one at a time:

- **Move the cameras to another USB controller**, or cut them to 640x360. The
  USB tree is the one thing every freeze has in common.
- **Cap the GPU:** `sudo nvidia-smi -pl 400`. Demoted by the fans-still-spinning
  observation, but still cheap: if capping makes the freezes stop, the answer is
  power after all.
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
| 2026-08-26 | teleoperation **with the Rerun viewer open** | a few minutes | **froze.** Teleoperation never worked at all; everything down, display stale. The fastest known reproducer |
| 2026-08-26 | 3 episodes through the recorder, three 1280x720 cameras, NVENC, **10 GB held on the GPU** by torch to imitate a resident policy | 300 frames each | **saved clean**: 900 frames readable, 3 episodes in the metadata, no leftover frames, no failures. So the encode failure needs something this does not have — real cameras, a real inference load, or 120 s episodes |

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
- **A power cut / PSU shutdown**, as the *whole* story: the fans kept running
  and the screen painted one more line. Something was still executing.
- **The dataset recorder as a cause of the freeze** — three episodes with NVENC
  and 10 GB of CUDA resident save cleanly (see the table).
- **CPU thermal throttling**, as far as ten-minute `sar` averages can say: 4–8%
  average CPU. Not conclusive at one-second resolution, which is why the
  telemetry now exists.
