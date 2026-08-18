# Operating the bimanual DK1

Everything this fork adds runs through one command: `dk1`.

Upstream's own README follows this file's pointer and covers the hardware, the
CAD and the URDF. This guide covers running the cell.

---

## 1. Install

```bash
git clone https://github.com/NikolasKelava/trlc-dk1.git
cd trlc-dk1
uv venv --python 3.12
uv pip install -e ".[dev]"
```

Check it:

```bash
uv run dk1 --help
uv run pytest -q          # 103 tests, none need hardware
```

---

## 2. Safety — read before connecting

**Connecting is not passive.** Connecting a follower arm energises every motor
and self-zeroes both grippers by driving them closed until they stall. That
calibration runs every time, before any of your code. Clear the workspace, keep
hands and cables away from the grippers, and expect the arms to stiffen and hold
position the moment a command connects.

Every command that connects says so in its `--help` and warns again before it
acts. Commands that only read `/dev` — everything in section 3 — connect to
nothing.

**Stopping never moves the arms.** LeRobot's rollout defaults to sweeping the
arms back to their startup pose during teardown. That is motion caused by
pressing stop, which is the opposite of what you want when you stopped because
something was wrong. It is off by default here. Return-to-home is always opt-in.

**Keep the hardware e-stop in reach** whenever a policy is driving. A keyboard
stop needs a focused terminal, a live key listener and a responsive loop — all
three can fail exactly when you need them.

### The joint speed limit

Use `--robot.type=bi_dk1_follower_safe`. It caps joint speed in **both** control
modes:

```
--robot.max_joint_rate 0.2      # rad/s, about 11 deg/s. The default.
--robot.max_gripper_rate 1.0    # normalised units/s
--robot.max_lag 0.15            # rad a command may lead the measurement
```

The rate is per **second**, not per command, so the same number means the same
speed whether a policy is commanding at 30 Hz or teleop at 200 Hz. The first
command after connecting holds position, so connecting cannot lurch.

> Upstream's `--robot.joint_velocity_scaling` does **not** do this. It only
> affects `pos_vel` mode; in `impedance` mode — which is what the bimanual
> follower runs — it does nothing at all. Do not rely on it.

---

## 3. Find the devices

Every device is recorded in one file, `dk1.toml`, tracked in the repo. Device
nodes move when things are replugged, so verify before trusting them:

```bash
uv run dk1 config check
```

Validates the file and confirms all four arm ports and all three cameras are
present right now. Opens nothing, energises nothing.

```bash
uv run dk1 config show          # what is configured
uv run dk1 find cameras         # what is attached, vs. what is configured
uv run dk1 find arms            # re-identify the four serial ports
```

`find arms` asks you to unplug one arm at a time and watches which
`/dev/ttyACM*` node disappears. It reads `/dev` and nothing else.

Each of these rewrites **only its own section** of `dk1.toml`. Running
`find arms` cannot disturb the camera settings, and vice versa; comments and
every other section survive untouched.

### What is in `dk1.toml`

**`[arms.follower]`, `[arms.leader]`** — one `/dev/ttyACM*` per arm. Two arms
sharing a port is rejected as a discovery mistake.

**`[cameras.top|left|right]`** — the names are not free choices. The MolmoAct2
checkpoint's image keys are `observation.images.{top,left,right}` and a mismatch
fails at startup. Addressed by `/dev/v4l/by-path`, which encodes the physical USB
hub port. The alternatives do not work: `/dev/videoN` moves between reboots, and
`/dev/v4l/by-id` is unusable because all three cameras report serial `20010101`,
so only one wins the symlink. `rotation` is per camera; all three are currently
mounted upside down (180).

**`[capture.policy|teleop]`** — resolution differs by use, device identity does
not. `fourcc` is `MJPG` everywhere and should stay that way: YUYV at 720p60 needs
~884 Mb/s and the uvc driver fails to allocate it, so reads die immediately.

A bad config fails on load with a message naming the offending key, and
`config check` reports *all* missing devices in one pass rather than stopping at
the first.

---

## 4. Teleoperate

Not built yet — Phase 2.

## 5. Record a dataset

Not built yet — Phase 4.

## 6. Evaluate MolmoAct2 zero-shot

Not built yet — Phase 3. This is the first real goal: find out how the
off-the-shelf `lerobot/MolmoAct2-BimanualYAM-LeRobot` checkpoint behaves on the
DK1 before recording anything or fine-tuning anything.

It will run in escalating order of risk — model loads and returns a chunk (GPU
only, no robot) → dry run of the full deployment path with actions printed and
never sent → a slow, hard-rate-limited rollout with a human on the e-stop.

## 7. Fine-tune and deploy

Not built yet — Phase 4, and gated on looking at the Phase 3 results together.

---

## What is actually verified

This fork inherits a body of reasoning from an earlier attempt at the same
project, in which confirmed results and untested plans were not distinguished.
They are here:

**Verified on hardware.** The LeRobot plugin classes and the DM4310/DM4340 motor,
impedance and gravity-compensation stack. Bimanual teleoperation, with and
without cameras. The three camera facts above (shared serial, MJPG requirement,
upside-down mounting).

**Not verified.** Everything about MolmoAct2 on this robot. No dataset has been
recorded, no fine-tune completed, and no policy has ever driven these arms. The
evidence that zero-shot is worth trying is a colleague's simulation work, which
is promising but carries no measured success rates.

One open assumption worth knowing about: `[capture.policy]` is set to 640×360 to
match the 16:9 aspect ratio the checkpoint was trained on. It is unconfirmed that
these cameras offer that mode at all — Phase 1 checks it.
