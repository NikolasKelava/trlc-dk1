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
and self-zeroes both grippers: each gripper is driven at velocity until it
stalls against its **open** stop, and that position is taken as zero. That
calibration runs every time, before any of your code. Clear the workspace, keep
hands and cables away from the grippers, and expect the arms to stiffen and hold
position the moment a command connects.

A gripper standing open reads `0.0000`, which is where the DK1's `0 = open,
1 = closed` convention comes from — and it is the opposite of the checkpoint's.
See section 6.

Every command that connects says so in its `--help` and warns again before it
acts. Commands that only read `/dev` — everything in section 3 — connect to
nothing.

**Stopping never moves the arms — unless you asked for homing.** LeRobot's
rollout defaults to sweeping the arms back to their startup pose during
teardown, on every exit path including a crash. That is motion caused by
pressing stop, which is the opposite of what you want when you stopped because
something was wrong. It is forced off here, permanently.

Homing is opt-in and is ours: `dk1 policy run --home` sweeps the arms to the
`[home]` pose in `dk1.toml` when the run ends — on the duration limit and on
Ctrl-C alike, but never after an error. It ramps at the same cap the policy ran
under, stops when the arms have actually arrived rather than after a fixed time,
and a second Ctrl-C stops the sweep where they are. Without `--home`, stopping
still disconnects and nothing else. See section 6.

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

**`hfov` / `target_hfov`** — the field of view, in degrees, and what to crop it
down to. `hfov` is the lens's own: **105°** for our Innomaker U30CAM-4K-S1, from
its user manual. `target_hfov` is optional and is the only one that changes
anything — set it and every frame from that camera has its centre cropped out and
stretched back to the configured size, so the picture spans the narrower angle.

Both wrist cameras are set to **87°**, which is what the MolmoAct2 BimanualYAM
checkpoint was trained on (its simulated wrist cameras are RealSense D405s, and
their intrinsics are built straight from that angle). Every rounding takes the
*larger* box, so the policy is never shown less of the scene than it was trained
on. The top camera is deliberately **left alone**.

**`crop_inset` / `crop_shift_x` / `crop_shift_y`** — the hand-tuned adjustments
on top of that. `crop_inset` takes extra pixels off the left and right edges
(top and bottom follow in proportion, so the box keeps the frame's aspect ratio
— a box that did not would come back out stretched, which is the distortion this
is all trying to remove). The two shifts move the box: **negative `crop_shift_y`
moves it up**, showing more of what is above the lens's centre line and dropping
the same amount off the bottom.

All three are in **pixels at 640 wide**, scaled to whatever the camera actually
delivers. They are eyeballed on a picture, so pixels are the natural unit — but
a pixel is a different angle at every capture resolution and this cell runs two,
so quoting them at one reference is what keeps teleop and rollout geometrically
identical. Change `[capture.policy]` and these numbers keep their meaning.

Currently `inset = 7`, `crop_shift_y = -50`. At the 1280×720 the policy captures
that is the box **905×509 at (187, 5)** = 85.3° H / 54.8° V, sitting 100 px above
centre. `dk1 config show` prints exactly this, so you never have to work it out.

That `y` of 5 is five pixels off the top of the sensor, so **−52 is about as far
as this box can be lifted**. Past that the shift clamps rather than failing — a
rollout must not stop for a bad number — and `dk1 config show` prints `CLAMPED`
with what it actually achieved. To lift further, drop `crop_inset` (a smaller box
has more room to move) or widen `target_hfov`.

Two things to know about this crop. It is done **in the camera**, so it applies to
everything this cell produces — what teleop displays, what recording stores, what
the policy is fed. `dk1 teleop --display` is therefore how you check it looks
right, and `dk1 config show` prints the box and the angle it achieves. And it is a
**pinhole** correction: our lens has real barrel distortion (the manual specifies
TV distortion < −6.2%), so the crop matches the trained geometry best at the
centre of the frame and only approximately at its edges. Fixing that needs a
calibration this cell does not have.

A side benefit visible immediately: the uncropped 105° frame has black vignette
corners, because the lens's image circle does not quite cover the sensor. The
crop removes them.

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

**`[home]`** — where `--home` sends the arms when a run ends, seven numbers per
arm in the 14-D vector's own order (`joint_1 .. joint_6`, then the gripper).
Optional, and now captured: both arms at their zero pose, every value within
0.024 rad of zero and both grippers open. Re-capture with
`dk1 policy home --capture` rather than typing radians; `dk1 config show` prints
what is currently set. If the section were removed, `--home` would fall back to
the pose the arms were in when the run connected, and the banner would say so.

**`[capture.policy|teleop]`** — resolution differs by use, device identity does
not. `fourcc` is `MJPG` everywhere and should stay that way: YUYV at 720p60 needs
~884 Mb/s and the uvc driver fails to allocate it, so reads die immediately.

`config check --formats` asks each camera which modes it actually advertises,
because OpenCV will not. Given a size a camera does not offer, OpenCV accepts it
and silently substitutes the nearest one it does — so a profile that looks fine in
the config would quietly hand the policy a different aspect ratio than it was
trained on. Both profiles are confirmed offered by all three cameras.

**Why the policy profile is 1280×720 and not 640×360.** MolmoAct2 resizes every
view to **378×378** — not 224×224, which is what the checkpoint's normalizer
*declares* but never applies. At 640×360 the cropped wrist box was 455×256, so
256 real rows were being upsampled to fill a 378-row model input: inventing
detail the sensor never captured. At 1280×720 the crop is 909×511 and 511 → 378
is a genuine downscale. It is free on the control loop — all three cameras
sustain 30.3 fps at 720p, decoding happens on per-camera background threads, and
the crop costs 1.6 ms per frame there.

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
  them open against their stop;
* each *leader* torques its gripper servo and drives it open — easy to forget,
  because the leaders are otherwise passive handles. Keep fingers out of the
  leader triggers.

Ctrl-C stops. Stopping disconnects and does nothing else: the arms are never
swept home. That is deliberate — sweeping the arms home is the last thing you
want when you stopped because something was wrong. When you do want it, ask for
it separately with `dk1 policy home` (section 6).

### Options worth knowing

```bash
uv run dk1 teleop --no-cameras          # arms only; cheaper loop
uv run dk1 teleop --display             # stream the cameras to Rerun
uv run dk1 teleop --display-policy-input   # ...and what the POLICY would be handed
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

**Watching the cameras.** `--display` opens Rerun and streams all three views
live, at `[capture.teleop]` resolution and with the wrist crop already applied —
so what you see is what this cell produces for a recording or a rollout.

**`--display-policy-input`** adds a second row: the **378x378 tensors the policy
would actually be handed**, produced by running the checkpoint's real
preprocessor on each observation. That is a different picture from the top row —
between them sit a rename, a fixed key order, a channel-layout change and a
resize to a square — and this is the only way to see it without starting a
rollout. Move a wrist by hand and watch the bottom row track: if the orientation,
framing or ordering is wrong anywhere in the policy pipeline, it shows up here.

It implies `--display`. It loads **no model weights and uses no GPU** — only the
saved preprocessor and the HF image processor, about 0.6 s off disk. And it costs
the control loop nothing: the work runs on a background thread and one tick in 12
is sampled, so the worst tick measured is 5 ms against a 16.7 ms budget at 60 Hz.
If a frame arrives while the thread is busy it is dropped rather than queued —
a stale picture is worth less than a loop that keeps time.

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
powered down. Measured here: **171 ms per model call**, 11.1 GiB peak, ~950 ms
for the first call. That is about five control periods at 30 Hz, so a chunk can
never be computed inside one tick — which is what the async chunk FIFO below is
for.

It reports two latencies because there are two. A policy call usually answers
from the 30-step chunk it already computed (~12 ms); only one call in thirty runs
the model. Timing consecutive calls measures the cheap one.

**`dryrun`** does everything a rollout does except the last step. It prints, per
tick, where every joint is and where the policy wants it. Two things to look for:

* a large delta on the first tick means the policy disagrees with your start
  pose, and a rollout would begin by driving there — reposition first;
* the two gripper channels, watched with the grippers open and then closed, are
  what confirm the gripper convention on real hardware.

It **energises the arms** (connecting always does) but never calls
`send_action`. `--build-only` prints everything and connects to nothing.

**`run`** is the rollout. It is capped by `[limits.policy]` — 1.0 rad/s, about
57 deg/s — and stopping disconnects without moving anything. Keep a hand on the
e-stop. `--dry-run` prints the whole configuration without connecting.

### How the chunk reaches the arms

The policy does not produce one action per tick. It produces a **30-row plan**,
one second of motion at 30 Hz, and something has to decide when to compute the
next one and what to do with it. That decision is what makes the arms look
smooth or make them stop and go, and it has three settings:

```fish
dk1 policy run --task "..."                      # async chunk FIFO — the default
dk1 policy run --task "..." --blocking-fifo      # the old behaviour, for comparison
dk1 policy run --task "..." --no-fifo            # LeRobot's own per-tick engine
dk1 policy run --task "..." --rtc                # LeRobot's real-time chunking
```

**The default computes the next chunk on a worker thread while the current one
is still being served**, so the control loop never waits for the model. When the
chunk lands, the rows describing time that has already passed are dropped and the
rest replace the queue, cross-faded over four rows so the seam is a ramp rather
than a step. Measured on this machine against the real checkpoint, paced at
30 Hz with no robot attached:

| | loop rate | ticks that ran a model call |
| --- | --- | --- |
| `--blocking-fifo` | 25.8 Hz | 17 in 20 s, worst 201 ms |
| async (default) | **29.7 Hz** | **1** — the cold start |

`--blocking-fifo` is what ran on the arms before 2026-08-21: the loop stopped
for one model call per chunk, so the arms froze for about a third of a second
once a second. That is what `Record loop is running slower (3.4 Hz)` in the log
was reporting — one tick that contained an inference, once per chunk.

Two knobs, both reported by `--trace` so a run tells you where to put them:

* **`--replan-at`** (default 15) — the queue depth at which the next chunk is
  started. It must exceed the inference latency in ticks (about 10 here) or the
  queue can run dry. Higher is fresher *and* safer, at the cost of running the
  GPU harder.
* **`--blend`** (default 4) — rows over which a new chunk is faded into the old
  one. `0` splices hard. Keep it well under the replan interval; blending most of
  what gets executed drags each new plan onto the old one, and the policy stops
  reacting.

`--no-fifo` is LeRobot's stock engine, which rebuilds the whole input pipeline on
every tick and throws it away on 29 ticks in 30 — about 22 ms of a 33 ms budget.
It is there to measure the difference, not to use.

### Watching a rollout from the inside

Two flags exist because two things about a rollout are not visible from the
outside, and both were open questions after the first run.

```fish
dk1 policy run --task "..."                          # --trace is on by default
dk1 policy dryrun --task "..." --display-policy-input   # safe: nothing is sent
dk1 policy run --task "..." --display --display-policy-input
```

**`--trace`** (on by default) prints one line per action chunk — not per tick,
so it stays readable — and an end-of-run summary. Per chunk it gives the whole
cost breakdown, which is the thing LeRobot never logs:

```
chunk  12  plan  196 ms old =  6 rows (pre 39 · model 158 · post 0)  queue  9 -> 24, blended 4
  over  15 ticks at  33.4 ms = 29.9 Hz  (engine 0.0 · robot 4.2 · wait 29.2)
  policy  +0.012 -0.310 ... grip L +0.991 R +0.988
  robot   +0.012 -0.310 ... grip L +0.991 R +0.988
```

The first line is the **chunk**: how old the plan was when the arms started
executing it, what it cost to compute, how many rows were in hand when it landed
and how many after. The second is the **tick**: the measured period, and where it
went. `wait` is `precise_sleep` — the headroom — and separating it from `robot`
is what tells a loop with 29 ms to spare from one that is 29 ms over budget.

`policy` is the model's **own** output, before the postprocessor unnormalises it
and before any gripper inversion; `robot` is the same row after both. They are
printed separately because a question about the policy cannot be answered with a
vector LeRobot rewrote.

The summary turns the numbers into a reading and says so in plain words: whether
the queue ever ran dry, whether a model call landed on the control loop, whether
a chunk arrived past its own last row, whether the loop held its rate, and
whether the policy ever moved a gripper. `--no-trace` turns the per-chunk
printing off; the summary stays.

If you saw a line reading `per cached tick 34.1 ms ... pre 32.4 · post 46.2`
before 2026-08-21, those two numbers were per **chunk**, not per tick — the
trace remembered the last pipeline timing it had seen and stamped it onto every
tick. 79 ms of pipeline inside a 34 ms tick is not a thing that can happen.

**`--display`** streams the cameras and the robot state, as it always has, and
now also draws **three lines per joint on one axis**:

* `policy/<joint>` — the model's own plan for this tick, in radians, *before*
  the chunk cross-fade and *before* the speed limiter;
* `command/<joint>` — what the follower actually sent, after both;
* `observation.<joint>` — where the joint got to.

The layout is one panel per joint, seven across, so the top row is the left arm
and the bottom the right. That is what makes rough motion attributable without
another run: if `policy` is rough and `command` follows it, the plans are rough;
if `policy` is smooth and `command` steps or lags, it is ours — the blend or the
speed cap; if `command` is smooth and `observation` is not, it is the arm.

Note the two console rows are **not** in the same units. `policy` is the model's
raw output in normalised space, nominally [-1, +1]; `robot` is the same row after
the unnormaliser, so radians. `+0.977` and `+2.208` are one number written twice.
The Rerun panels put both in radians, which is the comparison you want.

**`--display-policy-input`** opens Rerun and logs, under `policy_input/`, the
images **as the model receives them**: the 378×378 tensors unpacked straight out
of `pixel_values`, which is what the VLM is handed. Not the same picture as
`--display`, and the difference is the point — teleop already showed the
robot-side view is right way up, and that was never what was in question. What
was is everything after it: the key order, the channel layout, and the resize to
a square. It also logs the policy's own action channels and the RTC queue depth
as scalars, so a gripper that moves shows up as a line.

If you saw this panel before 2026-08-20, it was lying to you. It logged the
`observation.images.*` entries of the preprocessor's output, and MolmoAct2's pack
step leaves those **untouched** — so the panel showed the robot-side view at the
camera's own size, captioned as the model's input. It could not have caught a
resize or aspect bug, which is most of what it is for.

Do it on `dryrun` first. That energises the arms and sends nothing, so you can
check the orientation with no risk at all.

### The gripper inversion is a flag, and it is off

`--invert-gripper` flips both gripper channels, `x -> 1 - x`, in both
directions. It is **off by default**. See the section below for why the
inversion is very probably right and why it is nevertheless not the default.

### Ending a run at a home pose

```fish
dk1 policy home --capture     # put the arms where home is, then record it
dk1 policy home               # drive there now, without loading the model
dk1 policy home --show        # print the configured pose. No motion.
dk1 policy run --task "..." --home
```

`--capture` energises the arms but commands nothing; it reads both arms and
rewrites only `[home]` in `dk1.toml`. Position them by hand first — note that
connecting self-zeroes the grippers open, so that is what a captured home
records for them.

This has been done: the pose currently in `dk1.toml` was captured with both arms
at zero, so "home" on this cell means the zero pose with the grippers open. The
*sweep* to it has not been watched yet — do `dk1 policy home` on its own, from a
pose the arms are not already in, before you put `--home` on a rollout.

The sweep ramps from the last command, watches the measured positions, and stops
when every arm joint is within 0.03 rad of home. Grippers are commanded but
excluded from the arrival test, because a gripper holding something is supposed
to stall. If the arms do not arrive — something blocking, a cap too low for the
distance — it says so and disconnects anyway, and **disconnecting disables every
motor**, so support anything holding itself up.

**It is slow, and eased at both ends.** The peak is 0.3 rad/s — *not*
`[limits.policy].max_joint_rate`, which is 1.0: that cap is an upper bound on
what a policy nobody trusts may do, and homing used to inherit it as a speed,
which is why it felt fast. A tighter cap still wins, since commanding faster
than the limiter allows only means the limiter clamps it. On top of that the
speed is scaled by a smoothstep — up over the first 0.75 s, and down again over
the last 0.25 rad — so the arms neither snap into motion nor stop dead on the
target. A sweep too short for both simply never reaches full speed. The profile
never quite reaches zero, or it would only approach home asymptotically and time
out short of it.

`dk1 policy home --max-joint-rate 0.6` overrides the peak for one sweep; naming
a speed explicitly is honoured as given.

This deliberately replaces LeRobot's `return_to_initial_position`, which fires
from teardown on every exit path including a crash, sweeps for a fixed 3 s
whether or not the arms arrive, and targets the connect-time pose. Behind the
0.3 rad/s cap, anything further than ~0.9 rad cannot finish in its 3 s.

### The gripper inversion, and why LeRobot will not do it for you

The DK1 normalises its gripper as 0 = open, 1 = closed. The checkpoint uses the
opposite: 1 = open, 0 = closed. If that is right, then left uncorrected the
policy opens the gripper every time it means to close it.

MolmoAct2 has the knob for this — `joint_signs` / `joint_offsets` — but on the
`lerobot-rollout` command line it does nothing. When a policy is loaded from a
path, LeRobot rebuilds both processor pipelines from the checkpoint's saved
`policy_preprocessor.json` and `policy_postprocessor.json`, and never consults
the policy config; the BimanualYAM checkpoint ships both with `joint_signs:
null`. `--policy.joint_signs=...` parses, validates, and is then ignored.

So `dk1 policy` patches the two pipeline steps directly after loading, and
refuses to run if it cannot find them.

It does that **only when `--invert-gripper` is given**, and the flag is off by
default. That is a deliberate step back from where this started. The argument
for inverting is good — two independent sources for the YAM convention, plus the
checkpoint's own gripper statistics sitting at mean 0.64 — but it is an
argument, not an observation: no run on this cell has yet been watched to open
or close a gripper on purpose. Until one has, the sign is a hypothesis, and a
hypothesis belongs behind a flag you can turn both ways in consecutive runs.

`--trace` records the policy's own gripper channel either way, so the run that
settles it will have the evidence in its own log.

## 6b. The same policy, in simulation

`sai-prasanna/molmoact2`'s `sim_eval` drives a ManiSkill scene over HTTP. It
never imports a model — it posts three frames, a 14-D state and an instruction to
an `/act` endpoint and executes what comes back. `dk1 policy serve` answers that
endpoint with **the checkpoint from `[policy]` in `dk1.toml`**, through the same
LeRobot pipelines a rollout uses, so the sim drives the same policy the arms do.

Two terminals:

```fish
# here
uv run dk1 policy serve                      # GPU only. No robot, no /dev, no motion.

# in the molmoact2 clone
uv run python -m sim_eval.run_eval \
    --policy-type remote-yam \
    --remote-url http://127.0.0.1:8202/act \
    -e BimanualYAMPutEverythingInBox-v1 -n 10
```

Videos, the policy's own camera frames and `results.json` land in
`sim_eval/outputs/<timestamp>/`.

`serve` deliberately runs **without RTC** (the client blocks on each response, so
there is no real-time deadline to compensate for) and **without the gripper
inversion** (`sim_eval` already speaks the checkpoint's 1=open convention). Both
mean behaviour transfers between sim and hardware but timing does not — and that
the sim tests the policy with this cell's gripper-sign question removed.

Setup notes, once: the clone needs `torch` repinned off its `cu121` wheels (the
5090 is sm_120 and they have no kernel image for it), its own
`sim_eval/scripts/download_assets.py`, and ManiSkill's YCB pack via
`python -m mani_skill.utils.download_asset ycb`.

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

**Not verified.** Everything about MolmoAct2 on this robot that would count as a
result. No dataset has been recorded, no fine-tune completed, and no rollout has
been **scored** — no success count over labelled attempts. The policy has driven
these arms, and it reaches for the object; that is an impression, not a rate.

The arm sides were confirmed directly, so nothing about the device config is
open. Teleoperation through this fork has now driven the arms, which is what the
uncapped default came out of — the first run, at 1.5 rad/s, felt sluggish.

Added in Phase 3: the converted bfloat16 checkpoint has been checked
(`dk1 policy check` passes) and inference has been run on this GPU — 171 ms per
model call, 11.1 GiB peak, a 14-D action in the right key order, with the gripper
inversion applied. The gripper-inversion hole in LeRobot's rollout path was found
by reading LeRobot 0.6.1's source.

On the hardware, `dk1 policy dryrun` has now run: the plumbing works end to end,
the policy agrees with the start pose to within 0.065 rad, its intended speed is
about 0.2 rad/s, and `0 = open` on the grippers is confirmed — they read 0.0000
standing open, and connecting does not close them.

`dk1 policy run` has driven the arms four times. The judder is gone (RTC's
prefix blend was collapsing to zero width), the queue no longer starves, and the
wrist field-of-view crop improved the alignment — the fourth run reaches for the
object and waits for a good position before closing the gripper.

The fifth thing found, and fixed on 2026-08-21, is the freeze: the loop waited
for one model call per chunk, so the arms held still for about a third of a
second once a second. That is the `Record loop is running slower (3.4 Hz)`
warning, and the async chunk FIFO removes it. **The sixth run confirmed it on
the arms**: 29.9 Hz over 335 chunks, zero starved ticks, the only over-budget
tick the cold start, and the home sweep completing in 8.5 s with the worst joint
0.028 rad off.

That closes this fork's side of the problem. The roughness that remains was
watched in `--display`'s per-joint panels — the policy's plan against the
command against the measurement — and it is in **the policy's own output**, not
in anything between the model and the motor.

Nothing on the hardware has been scored. In simulation it has:
**3/3** on `BimanualYAMPutEverythingInBox-v1` with a 120 s episode budget, on the
checkpoint's own embodiment, against about 50% for the reference HF server. The
0/10 recorded here earlier was scored with 27 s episodes, and the policy barely
acts for the first 30 s.

The home pose has been captured on the hardware — that path energises the arms
and reads them, nothing more. The home *sweep*, which drives both arms, has run
only against fakes in the test suite.
