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
uv run dk1 config show               # what is configured
uv run dk1 config check --formats    # ...and do the cameras offer the capture profiles?
uv run dk1 find arms --inspect       # what each serial port is, by USB identity
uv run dk1 find cameras --list       # what is attached, vs. what is configured
```

None of those four touches a device: they read `/dev` and ask the kernel what it
already knows. The two that do write are below.

**`dk1 find arms`** asks you to unplug one arm at a time and watches which
`/dev/ttyACM*` node disappears. Nothing is opened or energised.

You only need it when an arm moves to a different socket, and it is worth knowing
what it is really for. USB identity already separates followers from leaders: the
followers sit behind a Damiao USB-to-CAN adapter (`2e88:4603`) and the leaders
behind a Dynamixel serial adapter (`1a86:55d3`), which `--inspect` shows you for
free. What identity *cannot* see is which arm of a pair is the left one — that is
a fact about the room. So the unplugging settles the sides, and the result is
cross-checked against the adapter families before anything is written; if you
unplug the wrong cable, it refuses rather than writing a port that cannot be what
it claims.

**`dk1 find cameras`** grabs one still per attached camera, opens it in an image
viewer, and asks which view it is — `top`, `left` or `right`. There is no way to
derive this: the three cameras are the same model and all report the same serial,
so someone has to look. The still is captured with the mounting rotation already
applied, so you judge the picture the way the policy will see it. Along the way it
checks each camera really advertises every `[capture.*]` profile. Nothing is
energised and the arms do not move; it opens video devices only.

The stills stay on disk (`--outdir` to choose where) so you can re-open one when a
label turns out to have been wrong two cameras later.

Each of these rewrites **only its own section** of `dk1.toml`. Running
`find arms` cannot disturb the camera settings, and vice versa; comments and
every other section survive untouched.

### What is in `dk1.toml`

**`[arms.follower]`, `[arms.leader]`** — one `/dev/ttyACM*` per arm. Two arms
sharing a port is rejected as a discovery mistake. Note that `/dev/serial/by-id`
is no help for the followers: both adapters report serial `00000000050C`, so it
collapses to a single entry for the pair, exactly as the cameras do.

**`[cameras.top|left|right]`** — the names are not free choices. The MolmoAct2
checkpoint's image keys are `observation.images.{top,left,right}` and a mismatch
fails at startup. Addressed by `/dev/v4l/by-path`, which encodes the physical USB
hub port. The alternatives do not work: `/dev/videoN` moves between reboots, and
`/dev/v4l/by-id` is unusable because all three cameras report serial `20010101`,
so only one wins the symlink. `rotation` is per camera; all three are currently
mounted upside down (180).

**`[limits.<activity>]`** — how fast the followers may move, per activity.
`dk1 teleop` reads `[limits.teleop]`. `max_joint_rate = false` means no limiting
at all (TOML has no null, so `false` is how "off" is spelled); anything else must
be a positive number, so a typo cannot quietly disable the cap. `max_lag` is
anti-windup — how far a command may lead the *measured* position, so a blocked arm
cannot wind up and lunge when it comes free. Delete the section and built-in
defaults apply.

This is the limit that works. Upstream's `joint_velocity_scaling` only reaches
`control_Pos_Vel`, so it is a silent no-op in impedance mode, which is the mode
the bimanual follower runs by default. Do not treat it as a safety knob.

**`[capture.policy|teleop]`** — resolution differs by use, device identity does
not. `fourcc` is `MJPG` everywhere and should stay that way: YUYV at 720p60 needs
~884 Mb/s and the uvc driver fails to allocate it, so reads die immediately.

`config check --formats` asks each camera which modes it actually advertises,
because OpenCV will not. Given a size a camera does not offer, OpenCV accepts it
and silently substitutes the nearest one it does — so a profile that looks fine in
the config would quietly hand the policy a different aspect ratio than it was
trained on. Both profiles are confirmed offered by all three cameras.

A bad config fails on load with a message naming the offending key, and
`config check` reports *all* missing devices in one pass rather than stopping at
the first.

---

## 4. Teleoperate

```bash
uv run dk1 teleop --dry-run     # what would run; connects to nothing
uv run dk1 teleop               # the real thing. MOVES THE ARMS.
```

`--dry-run` builds every config and prints the ports, camera names, resolutions
and speed limit without opening a single device. Do that first on a cell you have
not run before — it is free and it catches a wrong port or a missing camera before
anything is energised.

The real run warns on stderr and asks before it connects. **Two things move when
it connects**, before you touch a leader arm at all:

* the follower energises every motor and self-zeroes both grippers by driving
  them closed until they stall;
* each *leader* torques its gripper servo and drives it open — easy to forget,
  because the leaders are otherwise passive handles. Keep fingers out of the
  leader triggers.

Ctrl-C stops. Stopping disconnects and does nothing else: the arms are never
swept home. That is deliberate — sweeping the arms home is the last thing you
want when you stopped because something was wrong.

### Options worth knowing

```bash
uv run dk1 teleop --no-cameras          # arms only; cheaper loop
uv run dk1 teleop --display             # stream to Rerun
uv run dk1 teleop --fps 30              # slower loop
uv run dk1 teleop --max-joint-rate 0.6  # impose a cap for this run
uv run dk1 teleop --duration 20         # stop by itself after 20 s
```

**The speed cap is off.** Teleoperation runs with no slew limit, which is what
the DK1 does natively — upstream has no rate limit in impedance mode at all. The
limiter exists to bound a policy nobody trusts yet; in teleop the commands come
from your hand, so a runaway is already bounded by you, and a cap tight enough to
matter is tight enough to feel. It also costs a serial round-trip per tick, so
turning it off makes the loop faster as well as smoother.

Impose one for a single run with `--max-joint-rate 0.8`, or permanently by editing
`[limits.teleop]` in `dk1.toml`. Policy rollout is a different activity and keeps
its own, much tighter, limit.

**Camera names are not an option.** They are always `top` / `left` / `right`,
because that is what the MolmoAct2 checkpoint requires and therefore what
recording will need. The old repo called them `wrist_left` / `wrist_right`, which
cannot work.

Teleop is also the checkpoint that the ported LeRobot plugin and the DM4310 /
DM4340 motor stack are intact. It runs LeRobot's own `teleop_loop` — the same loop
recording and policy rollout use — rather than a bespoke one, so it exercises the
path every later phase depends on.

## 5. Record a dataset

Not built yet — Phase 4.

## 6. Evaluate MolmoAct2 zero-shot

Phase 3, and the first real goal: find out how the off-the-shelf
`lerobot/MolmoAct2-BimanualYAM-LeRobot` checkpoint behaves on this cell before
recording anything or fine-tuning anything. Four commands, in escalating order of
risk. Do them in that order — each one catches failures the next would otherwise
find with the arms live.

```fish
dk1 policy check                       # reads JSON. No GPU, no robot, no motion.
dk1 policy smoke                       # loads the model, runs inference. GPU only.
dk1 policy dryrun --task "..."         # arms attached; actions PRINTED, never sent.
dk1 policy run --task "..."            # the rollout. The policy drives the arms.
```

The checkpoint comes from `[policy]` in `dk1.toml`; `--checkpoint` overrides it
per run.

**`check`** compares the checkpoint against what this cell provides: 14-D state
and action, the `yam_dual_molmoact2` normalisation statistics, and the
`top` / `left` / `right` image order. It reads the *saved processor pipelines*,
not just `config.json`, because those are what actually run.

**`smoke`** loads the policy on the GPU and runs inference on a synthetic frame.
Nothing is connected and no `/dev` node is opened, so it is safe with the cell
powered down. Its useful output is the steady-state latency: anything above one
control period (33 ms at 30 Hz) means the rollout needs `--rtc`, which is the
default.

**`dryrun`** does everything a rollout does except the last step. It prints, per
tick, where every joint is and where the policy wants it. Two things to look for:

* a large delta on the first tick means the policy disagrees with your start
  pose, and a rollout would begin by driving there — reposition first;
* the two gripper channels, watched with the grippers open and then closed, are
  what confirm the gripper convention on real hardware.

It **energises the arms** (connecting always does) but never calls
`send_action`. `--build-only` prints everything and connects to nothing.

**`run`** is the rollout. It is capped by `[limits.policy]` — 0.3 rad/s, about
17 deg/s — and stopping disconnects without moving anything. Keep a hand on the
e-stop. `--dry-run` prints the whole configuration without connecting.

### The gripper is inverted, and LeRobot will not do it for you

The DK1 normalises its gripper as 0 = open, 1 = closed. The checkpoint uses the
opposite: 1 = open, 0 = closed. Left uncorrected, the policy opens the gripper
every time it means to close it.

MolmoAct2 has the knob for this — `joint_signs` / `joint_offsets` — but on the
`lerobot-rollout` command line it does nothing. When a policy is loaded from a
path, LeRobot rebuilds both processor pipelines from the checkpoint's saved
`policy_preprocessor.json` and `policy_postprocessor.json`, and never consults
the policy config; the BimanualYAM checkpoint ships both with `joint_signs:
null`. `--policy.joint_signs=...` parses, validates, and is then ignored.

`dk1 policy` patches the two pipeline steps directly after loading, always, and
refuses to run if it cannot find them. There is no flag to turn it off.

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

Added in Phase 1: all three cameras advertise 640×360 MJPG at 30 fps, so
`[capture.policy]` is real and no 4:3 fallback is needed. The `top` / `left` /
`right` labels in `dk1.toml` were confirmed by previewing each camera. The
follower and leader ports were confirmed by USB adapter identity, which also
settles a contradiction the earlier project left behind — an untracked
`robot-ports.txt` there claimed `follower.left = /dev/ttyACM2`, but `ttyACM2` is a
leader adapter, so the tracked `ports.toml` was the correct one.

**Not verified.** Everything about MolmoAct2 on this robot. No dataset has been
recorded, no fine-tune completed, and no policy has ever driven these arms. The
evidence that zero-shot is worth trying is a colleague's simulation work, which
is promising but carries no measured success rates.

The arm sides were confirmed directly, so nothing about the device config is
open. Teleoperation through this fork has now driven the arms, which is what the
uncapped default came out of — the first run, at 1.5 rad/s, felt sluggish.

Added in Phase 3, and worth being precise about: the converted bfloat16
checkpoint on this machine has been read and checked (`dk1 policy check` passes),
and the gripper-inversion hole in LeRobot's rollout path was found by reading
LeRobot 0.6.1's own source. Neither of those is a result on the robot. No
inference has been run in this repo and the policy has still never driven these
arms.
