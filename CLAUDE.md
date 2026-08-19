# CLAUDE.md — architecture and project state

Read this first. It is the whole picture; `GUIDE.md` is the operator-facing
version and does not repeat this.

## What this is

A fork of [robot-learning-co/trlc-dk1](https://github.com/robot-learning-co/trlc-dk1)
set up to operate a bimanual TRLC-DK1 cell (2 leader arms, 2 follower arms, 3 USB
cameras) with LeRobot, and to evaluate and fine-tune the **MolmoAct2** VLA policy
on it. Origin is `NikolasKelava/trlc-dk1`; upstream is the hardware repo and we
want to keep pulling its updates.

## Hard rules

- **Never open a PR, never push.** Commit locally on a branch; Nikolas publishes.
- **Do not modify upstream files.** `lerobot_robot_trlc_dk1/`, `trlc_dk1_control/`,
  `urdf/`, `hardware/`, `media/`, `README.md` belong to upstream. Extend by
  subclassing in `dk1lab/`. The only accepted upstream deltas are listed below.
- **Do not run anything that moves the arms without asking.** See Safety.
- **Do not jump phases.** Finish the current one; Nikolas gates each transition.

## Repository layout

```
dk1lab/                 everything this fork adds; the only Python we own
  layout.py             the 14-D vector contract + gripper inversion   (no lerobot import)
  config.py             dk1.toml load / validate / surgical write      (no lerobot import)
  limiter.py            slew-rate limiter                              (no lerobot import)
  discovery/arms.py     serial-port identification by unplugging       (no lerobot import)
  discovery/ports.py    serial-port identity from USB vid/pid          (no lerobot import)
  discovery/cameras.py  by-path enumeration + the labelling loop       (no lerobot import)
  discovery/formats.py  v4l2 capture-mode probing                      (no lerobot import)
  discovery/preview.py  grab a still and show it                       (cv2, no lerobot)
  checkpoint.py         read a MolmoAct2 checkpoint's metadata     (no lerobot import)
  home.py               the home sweep: ramp, arrival test, abort  (lerobot lazily)
  cameras.py            builds lerobot OpenCVCameraConfig from config
  robot.py              SafeBiDK1Follower — the rate-limited follower
  teleop.py             the one teleoperation implementation
  policy.py             MolmoAct2 deployment: smoke / dryrun / rollout
  trace.py              per-chunk latency, queue depth, and the policy's OWN action
  serve.py              the /act HTTP endpoint sim_eval drives — no robot
  cli/                  Typer app; `dk1` entry point
dk1.toml                THE device config. Tracked. Single source of truth.
tests/                  387 tests, none need hardware
GUIDE.md                operator docs
lerobot_robot_trlc_dk1/ UPSTREAM — LeRobot plugin classes
trlc_dk1_control/       UPSTREAM — DM4310/DM4340 chain, impedance, MuJoCo grav-comp
```

The dependency split is deliberate: the modules marked *no lerobot import* stay
importable without torch, so config handling and its tests are fast and work on a
machine with no robot stack. `discovery/preview.py` needs cv2 but not lerobot, and
imports it inside the functions so the module itself stays cheap.

Every discovery module separates the decision from the I/O — `parse_formats_ext`
from `probe`, `assign_labels` from the capture-and-prompt callbacks, `detect_removed`
from `find_arms`, `role_conflicts` from `list_ports`. That is why the whole suite
runs with no robot attached.

## Accepted upstream deltas (keep minimal, they are the rebase surface)

| File | Change |
| --- | --- |
| `pyproject.toml` | one hunk: `dk1lab` package, deps, `dk1` script |
| `.gitignore` | one appended hunk |
| `README.md` | 3-line pointer at the top |
| `.gitmodules` | **deleted** — pointed at `src/trlc-dk1/motors/DM_Control_Python`, a path that does not exist, for code committed as ordinary tracked files |
| two cherry-picks | `7212df3` bimanual follower feature descriptors, `c0760ef` control_mode exposure |

Those two cherry-picks are **load-bearing**: on plain upstream,
`BiDK1Follower.action_features` raises `AttributeError: 'DK1Follower' object has
no attribute 'motors'`. They are genuine upstream bugs worth upstreaming — say so,
do not do it.

## Invariants that must not be broken

**The 14-D vector layout.** `dk1lab/layout.py` derives it once:
```
0..5  left_joint_1.pos .. left_joint_6.pos     6  left_gripper.pos
7..12 right_joint_1.pos .. right_joint_6.pos  13  right_gripper.pos
```
Left block first, gripper last within each arm — matching MolmoAct2 BimanualYAM.
Never restate this as a literal anywhere else; derive from `layout.ACTION_KEYS`.
`tests/test_layout.py` asserts it still equals the live `BiDK1Follower`.

**Camera names.** `top`, `left`, `right`, in that order. The policy's image keys
are `observation.images.{top,left,right}`; LeRobot derives them from the robot's
camera keys and the rollout context fails on a mismatch. Note the order is *not*
alphabetical — sorted would be left/right/top.

**Config writes are surgical.** `write_arms` touches only the arms tables,
`write_cameras` only the cameras tables; comments and all other sections survive
byte-identical, and writes are atomic. Tested both directions. The previous
attempt at this project lost its whole camera section to a port scan that rewrote
the file wholesale. When editing `dk1lab/config.py`: mutate existing tomlkit
tables in place — replacing a table object eats the comment introducing the *next*
section, because tomlkit stores it as the previous table's trailing trivia.

**Nothing else hardcodes a port or a `/dev` path.** Only `dk1.toml`, plus the
discovery globs.

**`[home]` is optional and all-or-nothing.** Seven numbers per arm in
`layout.ARM_KEYS` order; a section missing a joint is rejected rather than
defaulted, because homing thirteen joints and leaving one where the policy put
it looks like homing and is not. Written by `write_home` (surgical, like the
others) from `dk1 policy home --capture`; absent, `--home` falls back to the
pose captured at connect and every banner says which of the two it is using.
**It is now filled in** — captured on the hardware 2026-08-19, both arms at
their zero pose (every value within 0.024 rad of zero, both grippers ~0 = open).
So home is "zero pose, grippers open", not an arbitrary resting position.

## Safety (non-negotiable)

- **Connecting is not passive.** `connect()` energises every motor and self-zeroes
  both grippers by driving each at velocity until it stalls against its **open**
  stop, then zeroing there (`follower.py`: `gripper_open_pos = 0.0` is *greater*
  than `gripper_closed_pos = -4.7`, so the positive velocity in the calibration
  opens). Every command that connects must say so in `--help` and warn again on
  stderr before acting. Helpers: `dk1lab/cli/safety.py`.
  The inherited wording said "driving them **closed** until they stall" and was
  wrong — a safety notice pointing at the wrong hazard. Corrected in Phase 3
  after Nikolas confirmed the grippers do not close on connect.
- **Stopping never moves the arms** unless homing was asked for.
  `return_to_initial_position` defaults to `true` in LeRobot's rollout — it is
  forced `false`, always, including under `--home`. Note what "never moves" does
  and does not mean: nothing is *commanded*, but a clean disconnect in impedance
  mode reaches `DK1MotorChain.stop()`, which **disables every motor** — so a
  raised arm sags. Support anything held up. That is also why a home sweep that
  does not arrive is reported loudly: the motors go off a second later.
- **Homing is ours, opt-in, and does not run after a fault.** `dk1lab/home.py`,
  reached by `dk1 policy run --home` and `dk1 policy home`. LeRobot's built-in
  return-to-home is wrong here on three counts and is not used: it targets the
  connect-time pose; it interpolates for a fixed 3 s and disconnects whether or
  not the arms arrived (behind the 0.3 rad/s cap, anything over ~0.9 rad cannot
  finish); and it fires from `teardown` on every exit path including a crash.
  Ours ramps from the previous command at the cap the run drove under, tests
  arrival against the *measurement*, derives its timeout from the distance, runs
  on a clean end only (duration limit or Ctrl-C — `policy.ended_cleanly`), and
  takes SIGINT for the length of the sweep so a second Ctrl-C stops it where the
  arms are instead of `sys.exit(1)` mid-command.
- **The speed limit lives in the follower**, and in `dk1.toml`. `SafeBiDK1Follower`
  (`--robot.type=bi_dk1_follower_safe`) limits in *both* control modes; the
  numbers come from `[limits.<activity>]`, where `false` spells "no cap".
  **Teleoperation runs uncapped by default** — see Phase 2 for why. Policy
  rollout does not, and must not.
  Upstream's `joint_velocity_scaling` only reaches `control_Pos_Vel`, so it is a
  **silent no-op in impedance mode** — which is the mode the bimanual follower
  runs by default. Do not present it as a safety knob.

Limiter design, each property tested in `tests/test_limiter.py`:
rates in **rad/s** (same meaning at 30 Hz and 200 Hz) · ramps from the **previous
command** so stiction cannot deadlock the setpoint · **`max_lag`** caps how far a
command leads the measurement so a blocked arm cannot wind up and lunge · the
**first tick holds position** so connecting cannot lurch · `dt` capped by
`max_dt` so a stalled loop cannot buy a large step.

`bi_dk1_follower_safe` is only registered once `dk1lab.robot` is imported —
LeRobot's plugin discovery scans *distribution* names matching `lerobot_robot_*`,
which `dk1lab` is not. Go through the `dk1` CLI, or import it explicitly.

## MolmoAct2 facts (established by reading, not by running)

Checkpoint: `lerobot/MolmoAct2-BimanualYAM-LeRobot` (LeRobot format);
`allenai/MolmoAct2-BimanualYAM` is the HF-format equivalent.
14-D state and action, absolute joint pose, chunk 30 @ 30 Hz,
`norm_tag=yam_dual_molmoact2`, `setup_type="bimanual yam robotic arms in molmoact2"`,
`control_mode="absolute joint pose"`.

- **The arm joint map is identity.** Corroborated two ways: the colleague's sim
  branch says "verified by FK", and the checkpoint's own norm statistics have
  every YAM joint range fitting inside the DK1's configured limits, including the
  asymmetric j2/j3 (0 → +π) and j4.
- **The gripper channel is probably inverted, and that is now a flag.** The YAM
  server contract is **1 = open, 0 = closed**; the DK1 is **0 = open, 1 = closed**
  (`DK1Robot.command_gripper`, `DK1Leader.get_action`). Sources:
  `sim_eval/inference/common.py` on the colleague's branch (states it in three
  places), plus the checkpoint's gripper stats sitting at mean 0.64 / median
  0.73, i.e. predominantly open. `layout.yam_joint_signs()` /
  `yam_joint_offsets()` implement it.
  It was on by default through Phase 3 and is now **off by default**, behind
  `--invert-gripper`, at Nikolas's request after the second rollout. That is a
  step back on purpose: the argument is good but it is still an argument, and
  nothing has yet watched the policy open or close a gripper here. A hypothesis
  that can be run both ways in consecutive rollouts belongs behind a flag.
  `dk1lab.trace` records the policy's own gripper channel either way, which is
  the measurement that settles it.
- **Image order is already pinned in the checkpoint.** Its `policy_preprocessor.json`
  carries `["...top","...left","...right"]` — read directly, in the converted
  bf16 copy. The inherited claim that the processor sorts alphabetically at
  deployment is **wrong**; the hazard is real only when *training* rebuilds the
  processor from a new dataset's features. Pin `--policy.image_keys` for training
  anyway.
- **`--policy.joint_signs` is a silent no-op at rollout.** Found in Phase 3 by
  reading LeRobot 0.6.1. `make_pre_post_processors` takes its `pretrained_path`
  branch whenever a policy is loaded from a path — which is every rollout — and
  rebuilds both pipelines from `policy_preprocessor.json` /
  `policy_postprocessor.json`. `config.joint_signs` is read only on the
  build-from-scratch branch. The BimanualYAM checkpoint ships both pipelines with
  `joint_signs: null`, so the CLI flag parses, validates, is stored, and does
  nothing. **The old repo's `eval.sh GRIPPER_FIX=1` would therefore have run with
  the gripper backwards** — and the failure is symmetric and silent.
  `dk1lab.policy.apply_gripper_inversion` patches the two loaded pipeline steps
  instead, and raises rather than warns if it cannot find them. Same class of
  upstream bug as the two cherry-picks: worth upstreaming, do not do it.
- **Converted bf16 checkpoint** exists at
  `~/Documents/RobotLearning/trlc-dk1/outputs/molmoact2_bimanual_yam_bf16` (11 GB).
  Reuse it. Its `config.json` bakes in `"device": "cpu"` and an absolute
  `pretrained_path` — override both at load or it silently loads on CPU.
- **Camera geometry from the colleague's sim**: 640×360 (16:9), top cam on the
  base at `p=[0.15, 0, 0.8]` pitched ~60° down (D435i, 69.4° HFOV), wrist cams on
  `link6-7` at `p=[0.076, 0, 0.094]` (D405, 87° HFOV), arms at y = ±0.24 m both
  facing +X (not mirrored). Our Innomaker U30CAM-4K have neither FOV — a
  divergence we cannot fix in software.

## Evidence status — keep this line sharp

Nothing about MolmoAct2 on the real DK1 has been **evaluated**. No dataset
recorded, no fine-tune completed, no rollout scored. The policy has now driven
these arms twice and moved toward the target both times, which is not nothing —
but it is an impression, not a result, and the second run was still stalling for
most of every second. That line stays sharp: a policy that reaches is not a
policy that works.

It now also scores **0/10 in the colleague's own simulator**, on the checkpoint's
own embodiment, with perfect tracking and none of this cell's timing or
gripper-sign problems — see *The sim run*, below. That is the strongest evidence
yet, and it points at the policy rather than at us.

The zero-shot case rests on the colleague's sim work
(`sai-prasanna/molmoact2`, branch **`sim-eval-dk1`**, single commit `797b179`).
Nikolas reports it "works quite well" and that nothing changed since that commit.
But the commit message itself says it was *"verified without a GPU"* (registration,
mesh refs, adapter round-trips, CLI parse), the repo has no success rates, and the
two scripts it cites as provenance for its `DK1_ARM_BIAS` constant are absent.
Treat it as a promising anecdote. Report Phase 3 results plainly, including "it
does nothing useful" — that is a legitimate, expected outcome and it is the input
to the fine-tuning decision.

Verified on hardware (port, do not rewrite): the LeRobot plugin classes; the
motor/impedance/grav-comp stack; bimanual teleoperation with and without cameras;
that all three cameras report serial `20010101` so `/dev/v4l/by-id` is unusable;
that MJPG is mandatory (YUYV at 720p60 exceeds the UVC bandwidth allocation); that
all three cameras are mounted upside down.

Added in Phase 1, on the hardware:

- **All three cameras advertise 640x360 MJPG at 30/50/60 fps.** `[capture.policy]`
  is real; no 4:3 fallback is needed and the 16:9 aspect the checkpoint was
  trained on is achievable. `[capture.teleop]` 1280x720 MJPG@60 likewise.
- **The `top` / `left` / `right` labels in `dk1.toml` are correct**, confirmed by
  previewing each camera. Hub `10.1` is the overhead view; between `4.3` and `4.4`
  a fixed object shifts left, so `4.4` is the further-right camera.
- **The follower/leader split is settled by USB identity.** Followers are
  `2e88:4603` (HDSC CDC, the Damiao USB-to-CAN adapter `follower.py` opens at
  921600 baud); leaders are `1a86:55d3` (the serial adapter behind
  `DynamixelMotorsBus`). Both followers report serial `00000000050C`, so
  `/dev/serial/by-id` collapses for the pair just as the cameras' does; both
  leaders have distinct serials. **This kills the old repo's contradiction**: the
  untracked `robot-ports.txt` claimed `follower.left = /dev/ttyACM2`, but
  `ttyACM2` is a leader adapter, so the tracked `ports.toml` was right.
- **A by-path string is not unique across the machine.** The name omits the root
  hub, and the cameras hang off `usb2` while the arms hang off `usb1`, so
  `...-usb-0:4.3:1.0` names both the left camera and a leader arm. Unique within
  each subsystem directory, which is all that is relied on — but do not compare
  the two namespaces.

Added in Phase 3, **on the hardware** (`dk1 policy dryrun`, 10 steps, nothing
sent):

- **`0 = open` is confirmed on the real grippers.** They were open before the run
  and stayed open through it, reporting exactly `0.0000`. Connecting does not
  close them — which also kills the inherited "self-zeroes by driving them closed"
  safety line, now corrected everywhere.
- **The policy agrees with the start pose.** Worst first-tick disagreement was
  `left_joint_4` at 0.065 rad (3.7°); everything else under 0.02 rad. Nothing
  would lurch.
- **The chunk's own speed is ~0.2 rad/s** — 0.0065 rad per step at 30 Hz, read
  off one 30-step chunk unrolling against a frozen robot. That sits just under
  the 0.3 rad/s cap, so the limiter is a bound rather than a brake.
- **The gripper command is consistent with the inversion.** The model output
  ≈0.99 ("open" in YAM) and the arms were told ≈0.008 ("open" on the DK1).
  Uninverted, the first tick would have commanded 0.99 = fully closed. Still not
  proof — the model has not been seen to *change* the gripper — but the right sign.

**The home pose is captured** (2026-08-19, `dk1 policy home --capture`): both
arms at zero, `[home]` in `dk1.toml` reads within 0.024 rad of zero on every
joint and ~0 on both grippers. That is a *read* — the command energises the arms
and commands nothing — so it confirms the capture path and the writer, and says
nothing yet about the sweep that drives back to it.

Also found here: `make_robot_from_config` builds a robot by *class-name lookup*
in the package holding its config's module, which registration alone does not
satisfy. `dk1lab/__init__.py` exposes `SafeBiDK1Follower` through a lazy
`__getattr__` for exactly this. Teleoperation never hit it.

Added in Phase 3, on this machine (GPU only — no robot was involved):

- **The converted bf16 checkpoint is intact and matches this cell.** 10.1 GiB of
  bfloat16, `norm_tag=yam_dual_molmoact2`, 14-D state and action, and the saved
  preprocessor pins `top`/`left`/`right`. `dk1 policy check` passes.
- **Inference is 171 ms**, measured here, which finally lands the inherited
  ~172 ms figure: ≈ 5.1 control periods at 30 Hz, so `--sync` would stall the
  loop every 30th tick and RTC is not optional. First call 953 ms (warmup +
  CUDA-graph capture), peak GPU memory 11.1 GiB on the 32 GB card.
- **Timing `select_action` naively measures nothing.** It serves from the cached
  30-step chunk, so 29 calls in 30 cost ~12 ms; only a reset forces a real
  forward pass. `dk1 policy smoke` measures the two separately — the first
  version of it reported 12 ms and was wrong.
- The gripper inversion applies cleanly to both loaded pipeline steps.

**The arm sides are confirmed** — Nikolas verified the four ports in `dk1.toml`
are correct as they stand, so `dk1 find arms` was not needed. Nothing about the
ports is open any more.

## Phases

| | | |
| --- | --- | --- |
| **0** | Foundation — package, config, CLI, limiter, tests | **done**, branch `phase0-foundation` |
| **1** | Device discovery on the hardware | **done** |
| **2** | Teleoperation | **done** — run on the arms, limits tuned |
| **3** | Zero-shot MolmoAct2 evaluation — the first real goal | **run on the arms twice**; judder fixed, the stall is diagnosed but not fixed |
| **3s** | The same policy in ManiSkill, via the colleague's `sim_eval` | **run: 0/10, and it is not the plumbing** |
| **4** | Record + LoRA fine-tune | gated on reviewing Phase 3 together |

**Phase 1** — done. Built `dk1 find cameras` (preview a still per candidate with
rotation applied, prompt for `top`/`left`/`right`, write via `write_cameras`),
`dk1 find arms --inspect` (read-only USB identity), `dk1 config check --formats`
(does each camera really advertise every `[capture.*]` profile — OpenCV will not
tell you: it accepts an unavailable size and silently substitutes the nearest one
it has, which would hand the policy a different aspect ratio than training used),
and a cross-check that refuses to write an arms section contradicting the adapter
families. Findings are in the verified list above.

**Phase 2** — built; run on the arms once, and tuned as a result.

`dk1 teleop` is the single entry point, `dk1lab/teleop.py` the single
implementation. The control loop is LeRobot's `teleop_loop`, imported rather than
reimplemented, because recording and rollout run that same loop — a bespoke loop
here could work while the one every later phase depends on does not.

**Teleoperation runs with no speed cap** (`[limits.teleop] max_joint_rate = false`).
The first run at 1.5 rad/s felt sluggish and Nikolas asked for the DK1's native
behaviour, which is what this is: upstream's plain `bi_dk1_follower` has no slew
limit in impedance mode at all, since `joint_velocity_scaling` only reaches
`control_Pos_Vel`. It is also the right default for the activity — the limiter
exists to bound a policy nobody trusts yet, and in teleop the commands come from a
human hand, so a runaway is already bounded by the person holding the leader arm.
A cap tight enough to matter is tight enough to feel. Disabling it also removes a
serial round-trip per tick, since `SafeBiDK1Follower` only reads
`measured_positions()` when the limiter is enabled.

**This does not extend to Phase 3.** A policy is exactly the case the limiter was
written for. Give rollout its own `[limits.policy]` profile with a real cap.

Also documented here: **connecting a leader is motion too.** `DK1Leader.configure`
torques the leader gripper servo and drives it to `gripper_open_pos`, so a finger
resting in a leader trigger gets pushed. `safety.LEADER_HELP` says so.

`dk1 teleop --dry-run` builds every config and prints it while connecting to
nothing.

**Phase 3** — four escalating commands, all built, **all four now run**, plus
a fifth (`home`) added afterwards that has not:

| | | risk |
| --- | --- | --- |
| `dk1 policy check` | reads the checkpoint's JSON | none — no GPU, no robot |
| `dk1 policy smoke` | loads it, runs inference on a synthetic frame | GPU only, nothing connected |
| `dk1 policy dryrun` | full deployment path, actions **printed, never sent** | arms energised, no pose commanded |
| `dk1 policy run` | the rollout | the policy drives the arms |
| `dk1 policy home` | the home sweep alone, no model | both arms move |

Two flags added after the second rollout, both read-only: `--trace` (on by
default) and `--display-policy-input`. `--invert-gripper` is now off by default.

`dk1lab/policy.py` is the single implementation; `dk1lab/checkpoint.py` is the
JSON-only reader behind `check`. Settings are decided in code, not left to a
command line: gripper inversion on, image keys pinned, `inference_action_mode
= continuous`, bf16 on cuda, RTC by default, `return_to_initial_position` forced
false. `--home` is the one opt-in: it runs `dk1lab.home` rather than LeRobot's
teardown sweep. The pose has been captured on the arms; **the sweep itself has
still not run on them** — its logic is tested only against fakes.

All four have been run, `run` twice. The first was juddering and stalling; the
judder was found and fixed. The second judders much less, still stalls, and
reacts less to the scene — diagnosed to a 900 ms in-situ chunk latency, not yet
fixed. See *The first rollout* and *The second rollout*, below.

Two things in the bf16 checkpoint's `config.json` are wrong for us and are
overridden at load: `"device": "cpu"` and a stale absolute `pretrained_path`.
`dk1 policy check` passes on it — the weights are 10.1 GiB of bfloat16, the norm
tag is `yam_dual_molmoact2`, the vectors are 14-D, and the saved preprocessor
pins top/left/right. That is a file check, not a result.

The speed cap lives in `[limits.policy]`: **0.3 rad/s**, about 17 deg/s. Timid on
purpose; it is the first number to raise once the policy has been watched, and it
must not be turned off for a rollout the way teleop's is.

The checkpoint path lives in `[policy]` in `dk1.toml` rather than in Python, for
the same reason the ports do. `~` is expanded.

### The first rollout, and what was wrong with it

The first `dk1 policy run` ("pick up the marker") moved, reached toward the
marker, and actuated the grippers — but juddered, hesitated, and stalled for
seconds at a time. The log carried both `Record loop is running slower` and
`Indexes diff is not equal to real delay`. Raising `--execution-horizon` to 30
made it smoother and *more* confused.

**The cause was RTC's prefix blending collapsing, and it was our number that
collapsed it.** RTC merges each new chunk with the unexecuted tail of the
previous one, weighting the first `delay` steps at 1.0 — the steps that went by
while the GPU was thinking — then ramping linearly to 0 at `execution_horizon`.
The ramp is the whole mechanism. `DEFAULT_EXECUTION_HORIZON` was **10**, chosen
from the smoke test's 172 ms, which is ~5 ticks at 30 Hz.

But 172 ms is the **`select_action` path**. A rollout runs `predict_action_chunk`
through RTC, which is a different, slower path — measured here at **324 ms**,
almost exactly **10 ticks**. So `delay >= execution_horizon`, the ramp had zero
width, and `get_prefix_weights` degenerated to a step function: ten steps pinned
rigidly to the old chunk, then twenty unconstrained, meeting at a discontinuity
about six times a second. That is the judder, and it is reproducible in one line
against LeRobot's own code (`tests/test_policy.py`).

`--execution-horizon 30` then pins *every* step at weight 1.0, so the new chunk
is dragged onto the old one for its whole length and the policy stops reacting to
what it sees. Smoother and more confused is exactly the predicted symptom.

Three things made it worse:

- **The latency tracker is poisoned by the warmup call.** `LatencyTracker.max()`
  is a running maximum that never decays, and the exclusion for warmup samples is
  gated on `use_torch_compile`, which is `False` here. So the first inference —
  measured at 511 ms with a cold model — set the delay for the entire run at 16
  ticks instead of 10. `dk1lab.policy.prewarm` now runs one inference before the
  RTC thread starts.
- **RTC ran with autograd on for nothing.** Its guidance step runs the action
  expert under `torch.enable_grad()`, so with the parameters still flagged
  trainable every flow step built and kept a full autograd graph. That graph is
  never used: `denoise_step` computes `v_t` *before* calling
  `x_t.requires_grad_(True)`, so the only differentiable path is
  `x1_t = x_t - time * v_t`, an identity in `x_t`, and the recovered correction
  equals the `grad_outputs` handed in. `freeze_for_inference` drops it: 324 ms →
  **272 ms**, action chunks **bit-identical** under a fixed seed.
- **Two serial round-trips per tick.** `SafeBiDK1Follower.send_action` called
  `measured_positions()` for the limiter's anti-windup clamp, a second full read
  of all 12 motors in a tick that had already read them. It now reuses the
  reading from `get_observation()` when it is younger than 15 ms, and falls back
  to a real read otherwise, so a caller that does not observe every tick is
  unaffected.

Measured on this machine, GPU only, no robot involved:

| | |
| --- | --- |
| backbone (VLM) forward alone | 55 ms |
| `predict_action_chunk`, non-RTC, 8 flow steps | 168 ms |
| `predict_action_chunk`, **RTC**, 8 flow steps | 324 ms |
| the same, after `freeze_for_inference` | **271 ms** = 9 ticks at 30 Hz |
| 30 Hz control loop, while RTC infers in a background thread | 29.9 Hz, p95 33.5 ms |

That last row matters: **the control loop is not GIL-starved by inference**, so
the async-inference server in LeRobot's `async` docs addresses a real problem
that this cell does not have. RTC is the in-process answer to the same problem
and it was already the default; it was simply mis-tuned.

`DEFAULT_EXECUTION_HORIZON` is now **20** — a 9-tick delay leaves an 11-step
blend, and 20 also matches the steady-state leftover length (`chunk - delay`), so
the previous chunk is used whole and never zero-padded. `dk1 policy smoke` now
measures the RTC path as well as the sync one and prints the delay it implies,
and both `smoke` and the `run` banner refuse to stay quiet when the blend is
thinner than `MIN_RTC_BLEND_STEPS`. Measuring only the path you are not going to
run is what caused this.

### The second rollout: the judder is gone, the stall is not

Re-run on the arms with the fixes above, plus the home sweep another session
added. Reported: **it judders much less, still stalls, and reacts noticeably
less to the object being moved.** Its log carried two warnings, roughly one pair
per twenty debug lines:

```
lerobot.policies.rtc.action_queue: Indexes diff is not equal to real delay.
                                   indexes_diff=10, real_delay=27
trlc_dk1_control.robot:            No command for 0.50 s — holding last commanded target.
```

Those two lines are the same event from both ends, and together they say the
whole thing. Reading them against `rollout/inference/rtc.py`:

- `real_delay` is **not** the latency tracker. It is `ceil(new_latency / period)`
  for *this* chunk, computed fresh each iteration. `real_delay=27` therefore
  means one chunk cost **27 ticks — 900 ms** of wall time, against the 271 ms
  measured on the bench. Something in situ costs 3.3× the bench number, and
  nothing measured so far explains it.
- `_replace_actions_queue` then discards `real_delay` actions from the front of
  the 30-step chunk, because they describe time that has already passed. 30 − 27
  leaves **three actions**: 100 ms of motion, then 900 ms of nothing. That is the
  stall, structurally — not a hiccup but the steady state.
- `indexes_diff=10` is how many actions the control loop actually consumed while
  the chunk computed. Ten consumed over twenty-seven ticks means **seventeen
  ticks with nothing to send**, which is exactly what the motor chain reports as
  `No command for 0.50 s`.
- And `delay=27` is also passed into `predict_action_chunk`, so RTC's prefix
  weights are 1.0 across the entire execution horizon of 20. The new chunk is
  pinned to the old one for its whole usable length. **That is why it reacts less
  to the object moving** — it is the `--execution-horizon 30` failure mode
  arriving through a different door. Less judder and less reactivity is the
  matching pair of symptoms, and both follow from the one number.

So there is one root cause with two faces, and it is not a tuning constant this
time: **a chunk that takes 900 ms cannot drive a 30-step chunk at 30 Hz.** RTC
needs inference well under `chunk / fps` = 1 s, with room to spare. Band-aids
(trim by consumption instead of wall clock, cap the delay) all amount to
commanding stale targets. The fix is to find the missing 620 ms.

`dk1lab/trace.py` exists to find it, and it is the deliverable of this round
rather than a guess at the answer. It times the three things the RTC thread
does — `predict_action_chunk`, the preprocessor, the postprocessor — and
subtracts them from RTC's own per-chunk figure. The remainder is time the thread
was not running at all. The two outcomes point opposite ways:

| what the trace shows | what it means |
| --- | --- |
| `model_ms` ≈ 900 | the GPU path really is that slow live; look at the model |
| `model_ms` ≈ 270, `other` ≈ 620 | the RTC thread is losing the CPU to the control loop's camera and serial work. No model tuning helps; per-tick work has to come down |

The second is the more likely, and it would also revise the "the control loop is
not GIL-starved by inference" measurement in the table above — that was taken
with no robot attached, so it measured the loop while nothing was decoding three
MJPG streams or talking to two CAN adapters.

**Also still unexplained, and needing the arms:** whether the gripper opens at
the right moment. The direction is still inferred rather than observed changing,
which is why it is now a flag rather than a default.

### Watching a rollout: `--trace` and `--display-policy-input`

Both added this round, both read-only.

`--trace` (default **on** for `dk1 policy run`) prints one line per chunk — not
per tick — with the cost breakdown, the consumed count, the queue depth after
the merge, and the starved ticks; then the policy's **own** action row next to
the same row after the postprocessor. Those two vectors are kept separate on
purpose: a question about the policy cannot be answered with a vector LeRobot
rewrote. At the end it prints a summary that states, in words, whether the queue
ran dry, whether inference is too slow for RTC to leave anything, where the
unaccounted milliseconds are, and whether the policy ever moved a gripper.
`RolloutTrace` works under `--sync` too, where there is no queue: records are cut
at the postprocessor instead and the queue readings are *absent* rather than
zero.

`--display-policy-input` opens Rerun and logs, under `policy_input/`, the images
**as the model receives them** — the preprocessor's output, not the robot-side
view `--display` shows. Teleop already confirmed the robot-side view is upright
and correctly named; that was never the open question. What was unverified is
everything the policy pipeline does after that. Available on `dryrun` as well,
which is where to check it: arms energised, nothing sent.

Both attach by wrapping, never by replacing: `_TimedPipeline` proxies the
pipeline objects (forwarding `reset()`, `.steps` and everything else), and the
queue's `merge` is wrapped on the instance. `attach` runs after `build_context`
so `prewarm`'s cold call is not counted as a chunk; `attach_queue` runs after
`strategy.setup`, because the RTC queue does not exist until `engine.start()`.

What remains for this phase: re-run the rollout with the fixes, hand on the
e-stop, and report what happens plainly — including "it does nothing useful".
The home pose is captured (see below); what is still unwatched is the *sweep*.
Run `dk1 policy home` on its own, from a pose the arms are not already in,
before putting `--home` on a rollout — that is both arms moving along a path no
hardware has seen yet.

**Phase 4** — record → LoRA from the same checkpoint → deploy → scored, labelled
eval attempts.

### The sim run: the policy behaves the same way with every hardware excuse removed

`sai-prasanna/molmoact2`'s `sim_eval` is a **pure HTTP client** — it posts three
camera frames, a 14-D state and an instruction to an `/act` endpoint and executes
the chunk that comes back. It never imports a model. So `dk1lab/serve.py` answers
that endpoint with **the exact checkpoint the arms run**, through the same
LeRobot pipelines, and `sim_eval` is used byte-identical to upstream.

Clone at `~/Documents/RobotLearning/molmoact2` (scratch, not tracked here). Two
forced deviations, both recorded there: `torch` repinned from `2.5.1+cu121` to
`>=2.7+cu128`, because the 5090 is sm_120 and the pinned wheels have no kernel
image for it; and ManiSkill's YCB asset pack downloaded separately
(`python -m mani_skill.utils.download_asset ycb`), which the sim_eval README does
not mention.

Two structural differences from a rollout, both deliberate and both *in our
favour* for this experiment:

- **No RTC.** `sim_eval` blocks on each response, so there is no deadline, no
  latency to compensate, no queue to starve and no seam to blend. Every timing
  fault found on the arms is absent by construction.
- **No gripper inversion.** The wire protocol *is* the YAM convention —
  `sim_eval`'s own `yam_state_adapter` sends `grip in [0,1] (1=open)` and
  `yam_action_adapter` maps `1.0` back to ManiSkill's `-1.0` = open. That is a
  **third** independent statement of the convention, and it means the sim tests
  the policy with the DK1's gripper sign out of the picture entirely.

`BimanualYAMPutEverythingInBox-v1`, "put everything into the box", 10 episodes,
800 steps: **success 0/10**. Also 0/3 at `--n-action-steps 10` and 0/3 at `5`,
so it is not the 1-second open-loop chunk the repo defaults to.

It is not the plumbing, and these are the numbers that say so (one 400-step
episode, recorded state and action):

| | |
| --- | --- |
| tracking, median \|commanded − measured\| per arm joint | **0.001 – 0.009 rad** |
| right-arm travel, total per joint | up to **3.4 rad** |
| left-arm travel, total per joint | 0.16 – 0.72 rad |
| gripper state seen by the policy | **0.000 → 1.000, both hands** |
| gripper commanded | left −1.00 → +0.83, right −1.00 → +0.45 |

The first row is the important one: the simulator executes what the policy asks
to within a milliradian. Nothing is being lost between the model and the robot.
The policy reaches with the right arm and **opens and closes both grippers** —
so "the policy never actuates a gripper" is now dead as a hypothesis in sim, as
it already is on hardware. It simply does not do the task.

The camera views were checked by eye and are well-formed: an overhead view with
the box, the lego and the tennis ball, and two wrist views.

**What this does and does not establish.** On its own embodiment, in its own
simulator, with its own gripper convention, perfect tracking, no latency and no
queue, zero-shot MolmoAct2 scores 0/10 on the task the repo ships. That is the
same *class* of behaviour seen on the arms — moves, reaches, does not accomplish
— and it makes "the DK1 cell is what is breaking it" much harder to sustain.

The one thing not yet ruled out is **this server versus the reference one**:
`examples/yam/host_server_yam.py` loads `allenai/MolmoAct2-BimanualYAM` in HF
format through `transformers`' `predict_action`, which is a different code path
from our LeRobot bf16 copy. Running both and comparing is the last step that
would close this, and it costs a ~20 GB download. Until then, "the policy is
weak zero-shot here" and "our LeRobot packaging is subtly wrong" are not fully
separated — though the perfect tracking and the sane action ranges argue hard
for the first.

The colleague's `sim-eval-dk1` branch turns out to carry a **full DK1 in sim** —
`bimanual_dk1.urdf`, the meshes, `robots/bimanual_dk1.py`, a DK1 task and DK1
adapters — so running our own embodiment in sim is a checkout away, not a build.
That is the agreed next step after YAM.

## Environment

RTX 5090 (32 GB), Python 3.12, uv, LeRobot **0.6.1**, fish shell.
Old project for reference (read-only): `~/Documents/RobotLearning/trlc-dk1`,
branch `wip/molmoact2` — real knowledge in `MOLMOACT2.md` / `MOLMOACT2_EVAL.md`,
but its `molmoact2/` scripts beyond `smoke_test.py` and `convert_bf16.py` were
never run. Its two docs contradict each other on whether zero-shot will work;
neither is a result.

`uv run pytest -q` · `uv run dk1 --help`
