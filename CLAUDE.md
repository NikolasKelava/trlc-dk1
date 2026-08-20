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

## How to explain things to Nikolas

When reporting what happened — a bug, a diagnosis, a change — write it so it
lands without the reader holding this whole document in their head. The shape
that works:

1. **What you saw**, in the plain terms of the symptom.
2. **What caused it**, in ordinary language, one mechanism at a time.
3. **What you did about it.**

Rules for it, all learned from a report that was right and too long:

- **Short.** Cut anything that does not change the reader's understanding or
  their decision.
- **Report conclusions, not the investigation.** A hypothesis you tested and
  discarded is worth one sentence, not a section. "It turned out that preparing
  a 720p picture rather than a 360p one barely costs anything while the policy
  is running" is the whole of what a reader needs; the paired-A/B methodology
  and the two benchmarking traps belong in this file, not in the explanation.
- **Name components by what they do**, not by their class name. "The step that
  resizes the camera pictures for the model", not `MolmoAct2PackInputsProcessorStep`.
- **Numbers only where they carry the argument.** One number that decides
  something beats six that describe it.
- Say plainly what was *not* established, and whether the arms were involved.

This is about the explanation, not the record. `CLAUDE.md` keeps the full
detail — methodology, discarded hypotheses, exact figures — because the next
session needs it. The explanation to Nikolas is a different document with a
different job.

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
  fov.py                field-of-view arithmetic: the crop box     (no lerobot import)
  crop.py               CroppedOpenCVCamera — the crop, in the camera
  checkpoint.py         read a MolmoAct2 checkpoint's metadata     (no lerobot import)
  home.py               the home sweep: ramp, arrival test, abort  (lerobot lazily)
  cameras.py            builds lerobot OpenCVCameraConfig from config
  robot.py              SafeBiDK1Follower — the rate-limited follower
  teleop.py             the one teleoperation implementation
  policy.py             MolmoAct2 deployment: smoke / dryrun / rollout
  trace.py              per-chunk latency, queue depth, and the policy's OWN action
  modelview.py          the model's-eye view, live, during teleoperation
  fifo.py               ChunkFIFOInferenceEngine — one model call per chunk
  serve.py              the /act HTTP endpoint sim_eval drives — no robot
  cli/                  Typer app; `dk1` entry point
dk1.toml                THE device config. Tracked. Single source of truth.
tests/                  508 tests, none need hardware
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

**The wrist cameras are cropped, the top one is not.** `[cameras.*].target_hfov`
in `dk1.toml`; the crop lives in the *camera* (`dk1lab/crop.py`), not in a policy
processor, so it is true of every image this cell produces — teleop display,
recording and rollout alike. A crop that applied to only some of them would be
worse than none: a recorded dataset and the rollout it was fine-tuned for would
disagree about what the lens does. Frame size is unchanged (the crop is stretched
back to the configured `width x height`), so nothing downstream had to move.

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
  not the arms arrived (behind the speed cap — anything over ~3 rad at the
  current 1.0 rad/s, and it was ~0.9 rad when the cap was 0.3 — cannot finish);
  and it fires from `teardown` on every exit path including a crash.
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
  facing +X (not mirrored). **Ours are 105° HFOV / 116° diagonal** — read out of
  the Innomaker U30CAM-4K-S1 user manual (`INNO-MAKER/U30CAM-4K-S1` on GitHub),
  a 2.25 mm f/2.0 M12 lens on a 1/2.8" IMX415. The model string is confirmed off
  the USB descriptor on this machine, so it is the right datasheet.
  The old line here said the mismatch was "a divergence we cannot fix in
  software" and that is **wrong in one direction**: too *wide* can be cropped
  down, only too narrow is unfixable. Ours is wide, so **the wrist crop is now
  built and configured** — see *The camera crop*.

## Evidence status — keep this line sharp

Nothing about MolmoAct2 on the real DK1 has been **scored**. No dataset
recorded, no fine-tune completed, no success rate measured on the arms. The
wrist crop **has** now run on the arms — the fourth rollout — and alignment is
better and the gripper waits for a good position before closing. The retune on
top of it (inset 6, view lifted 20, capture raised to 1280×720) has **not**:
that is checked against the live cameras, the real preprocessor and the test
suite, and that is all. The
policy has now driven these arms three times. The third run — after the limit
and engine changes below — moves **smoothly, without stalling**, visually
tracks the dice and reaches for it, and **misaligns the gripper with the dice**.
That is a real, specific, diagnosable failure rather than an impression, and it
is a long way from where the first two runs were. But it is still not a scored
result: a policy that reaches is not a policy that works.

**In simulation, on 2026-08-20, our checkpoint scored 3/3.** That reverses the
0/10 recorded here a day earlier, and the thing that changed was the episode
budget, not the policy: 3600 steps (120 s) instead of 800 (27 s). The policy
barely acts for the first ~30 s and successful episodes average 54 s, so every
earlier run was scored before it had a chance to finish. Sai's reference HF
server, same task and budget, scored about 50%. See *The sim run*, below.

So the policy works zero-shot on its own embodiment, and our LeRobot bf16
packaging is not merely correct but ahead of the reference path. What remains
unevaluated is this **cell** — no rollout on the arms has been scored, and the
two that ran were both cut short by timing faults.

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
  off one 30-step chunk unrolling against a frozen robot. Read at the time as
  "the limiter is a bound rather than a brake", and that reading was **wrong**:
  one chunk from a resting start is the calmest part of a run. Measured over a
  whole 120 s episode the demand is bursty — median 0.036 rad/s but p95 0.31 and
  peaks of 4.56. See *What the caps were doing*.
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
  30-step chunk, so 29 calls in 30 cost ~12 ms *in total*; only a reset forces a
  real forward pass. `dk1 policy smoke` measures the two separately — the first
  version of it reported 12 ms and was wrong. Note "cheap" is relative: measured
  live at 30 Hz a cached call is ~5.5 ms, of which 1.8 ms is `self.eval()`
  walking 1737 submodules. See *The 27.7 Hz loop*.
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
| **3** | Zero-shot MolmoAct2 evaluation — the first real goal | **run on the arms four times**; judder and stall fixed, wrist FOV crop improved alignment and the gripper now waits for position. Crop retune, 720p capture and the chunk FIFO **built, not yet run** |
| **3s** | The same policy in ManiSkill, via the colleague's `sim_eval` | **run: 3/3 with a 120 s budget; 0/10 was a too-short episode** |
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
`--fifo` (on by default, `run` and `dryrun`) is **not** read-only in the same
sense — it changes which engine drives the arms — but it is action-identical;
see *The 27.7 Hz loop*.

`dk1lab/policy.py` is the single implementation; `dk1lab/checkpoint.py` is the
JSON-only reader behind `check`. Settings are decided in code, not left to a
command line: gripper inversion on, image keys pinned, `inference_action_mode
= continuous`, bf16 on cuda, **sync by default** (RTC starved the queue on the
arms; see *The stall*), `return_to_initial_position` forced
false. `--home` is the one opt-in: it runs `dk1lab.home` rather than LeRobot's
teardown sweep. The pose has been captured on the arms; **the sweep itself has
still not run on them** — its logic is tested only against fakes.

All four have been run, `run` three times. The first juddered and stalled; the
judder was found and fixed. The second juddered much less, still stalled, and
reacted less to the scene — diagnosed to a 900 ms in-situ chunk latency. The
third, after the limit raise and the switch to `--sync`, is **smooth and does
not stall**: it tracks the dice and reaches for it, but misaligns the gripper.
See *The first rollout*, *The second rollout* and *The third rollout*, below.

Two things in the bf16 checkpoint's `config.json` are wrong for us and are
overridden at load: `"device": "cpu"` and a stale absolute `pretrained_path`.
`dk1 policy check` passes on it — the weights are 10.1 GiB of bfloat16, the norm
tag is `yam_dual_molmoact2`, the vectors are 14-D, and the saved preprocessor
pins top/left/right. That is a file check, not a result.

The speed cap lives in `[limits.policy]`: **1.0 rad/s** (~57 deg/s) and
**`max_lag` 0.4 rad**, raised from 0.3 / 0.1 on 2026-08-20 from measurement —
see *What the caps were doing*, below. It must still not be turned off for a
rollout the way teleop's is.

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
`RolloutTrace` works under `--sync` too, and on its own terms: a record is the
window between two real inference calls, its total is measured wall clock, its
costs are **per tick**, and the queue readings are *absent* rather than zero.
Cutting a record at the postprocessor — which is what it used to do — cut one
per tick and reported nonsense; see *The 27.7 Hz loop*.

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

What remains for this phase: **a scored run with the crop and the FIFO on**. The motion
faults are closed, the failure is spatial, and the wrist crop that addresses it
is built and configured but **has never driven a rollout** — see *The camera
crop*. Run it, then score it: labelled attempts with a success count, which is
the input to the Phase 4 decision.

Still unwatched, and unchanged by any of this: the home *sweep*. The pose is
captured, but run `dk1 policy home` on its own, from a pose the arms are not
already in, before putting `--home` on a rollout — that is both arms moving
along a path no hardware has seen yet. And the gripper inversion has never been
seen to open and close correctly on the arms.

**Phase 4** — record → LoRA from the same checkpoint → deploy → scored, labelled
eval attempts.

### The third rollout: smooth, tracking, and misaligned

Run on the arms 2026-08-20, after raising `[limits.policy]` to
`max_joint_rate = 1.0` / `max_lag = 0.4` and defaulting `dk1 policy run` to
`--sync`. Reported by Nikolas, task "pick up the dice":

- **Motion is very smooth.** The judder is gone and stayed gone.
- **It does not really stall any more.** The `--sync` change did what the trace
  predicted it would: with `n_action_steps = 30` the engine executes the whole
  chunk instead of letting RTC discard 27 of every 30 rows.
- **It tracks the dice and moves toward it.** Visual servoing works — the policy
  is grounded on the object and drives the arms at it.
- **It misaligns the gripper with the dice.** This is now the failure.

So the timing faults are closed and the remaining problem is **spatial**, which
is a different class of bug and points somewhere specific: the cameras. The
checkpoint was trained on a top view at 69.4° HFOV and wrist views at 87°
(D435i / D405 in the colleague's sim). Our Innomaker U30CAM-4K match neither.
A policy that has learned "the gripper is aligned when the object appears *here*,
at *this* size" will be systematically off if the lens maps the world onto the
sensor differently than in training — which is exactly a consistent misalignment
on top of correct tracking.

**The next step is therefore to crop the camera images to the trained field of
view**, not to touch the policy. That is now done for the wrist views and is
described in *The camera crop*, below: ours are 105°, comfortably wider than the
87° the wrists were trained at, so the crop is possible. It has not yet driven a
rollout, so it is a built change and not a result.

Two things still not established by this run, and worth stating: nothing was
**scored** (no success/failure count over labelled attempts), and the gripper
inversion was still never *watched* to open and close correctly on the arms.

### The fourth rollout: the crop helped

Run on the arms 2026-08-20 with the wrist crop (87.1°, no inset, no shift) and
`[capture.policy]` still at 640×360. Reported by Nikolas:

- **Alignment is better.** The spatial failure the third rollout ended on is
  reduced, which is the first evidence that the field-of-view mismatch was
  really part of it.
- **The gripper hesitates to close until it is in a good position.** That is a
  *behaviour*, not a fault — and it is the first time the gripper channel has
  been watched doing something sensible on the arms, which the inversion
  question has been waiting for. It is still not a controlled test of the
  inversion, and `--invert-gripper` is still off by default.

Not scored, again. But the crop is now a change with evidence behind it rather
than an argument, and the follow-up work in this round — the retune, the capture
raise, and the corrected `--display-policy-input` — all comes from it.

### The camera crop: our lens is 105 degrees, the checkpoint's wrists are 87

Built 2026-08-20 in response to the third rollout's spatial failure. Nothing has
been run on the arms with it yet.

**The two numbers.** Ours is **105° HFOV / 116° diagonal**, from the Innomaker
U30CAM-4K-S1 user manual — a 2.25 mm f/2.0 M12 lens on a 1/2.8" IMX415. The
model string is confirmed off the USB descriptor here (`0bda:5883
Innomaker-U30CAM-4K-S1`), so it is the right datasheet, but the angle is the
manufacturer's figure and not a measurement. Theirs is in
`sim_eval/robots/bimanual_yam.py` on the colleague's `sim-eval-dk1` branch, and
it is not a lens spec at all — the sim *builds its intrinsics from an angle*:
`fx = (w/2)/tan(hfov/2)`, `fy = fx`, top 69.4° (D435i), wrists 87.0° (D405). So
the trained geometry is exactly a pinhole at those angles, which is what makes
the correction well-posed.

**The correction.** On a pinhole, the fraction of the frame width that spans the
target is `tan(target/2) / tan(source/2)` = `tan(43.5°)/tan(52.5°)` = **0.7282**.
Every rounding takes the *larger* box, at Nikolas's stated preference: too much
field of view degrades more gracefully than too little.

**On top of that sit three hand-tuned adjustments**, added after the fourth
rollout: `crop_inset` (extra pixels off the left and right edges, with top and
bottom following so the box keeps the frame's aspect ratio — an anisotropic box
would be stretched on the way back out, which is the exact distortion this is
undoing), and `crop_shift_x` / `crop_shift_y`. Currently **inset 7, shift_y −50**,
i.e. the view is lifted. At 1280×720 that is the box **905×509 at (187, 5)** =
**85.3° H / 54.8° V**, sitting 100 px above centre — and note the `y` of 5, which
means about −52 is as far as this box can be raised before the shift clamps.

**All three are quoted in pixels at a 640-wide reference** (`fov.REFERENCE_WIDTH`)
and scaled to whatever frame the camera delivers. They are eyeballed on a
picture, so pixels are the natural unit — but a pixel is a different angle at
every capture resolution and this cell runs two. Scaling from one reference is
what keeps teleop and rollout geometrically identical, which is the only thing
that makes "it looked right in teleop" evidence about what the policy gets. It
also means changing the capture resolution does not silently retune the crop.
A shift that would run off the sensor is **clamped, not raised** — retuning a
number beats refusing to produce a picture mid-rollout — and `fov.describe`
prints `CLAMPED` with the shift it actually achieved, because reporting the
number that was *asked* for would be a lie in exactly the case an operator most
needs to know about.

**The model input is 378×378, not 224×224.** This was wrong here for a day and
it matters. The 224s in the checkpoint's `molmoact2_masked_normalizer` are
declared feature shapes under `VISUAL: IDENTITY` — they normalise nothing and
resize nothing. The real resize is in the HF image processor
(`processor_config.json`: `crop_mode: "resize"`, `size: 378×378`,
`patch_size: 14`, `resample: 2`), and the packed tensor proves it: `pixel_values`
comes out `[3, 729, 588]` = three images of 27×27 patches of 14×14×3 = **378×378**.
It is a *stretch*, not a letterbox — 16:9 becomes 1:1 — which is what training
did too, so it is consistent.

**So the crop only costs no detail if it clears 378 in both axes**, and at
`[capture.policy]` 640×360 it did not: the tuned wrist box was 455×256, and 256
rows were being upsampled 1.5× to fill a 378-row input. That is why the policy
capture is now **1280×720** — see *The capture resolution*, below.

**Where it lives, and why there.** `dk1lab/fov.py` is the arithmetic (no lerobot
import, like every other decision module here); `dk1lab/crop.py` is
`CroppedOpenCVCamera`, an `OpenCVCamera` subclass whose only addition is a crop
and resize at the end of `_postprocess_image`. In the *camera*, not in a LeRobot
processor and not in `dk1lab/policy.py`, because the crop has to be true of every
image this cell produces: a processor step covers rollout and misses teleop, and
a `policy.py` step covers rollout and misses **recording**, which is the one that
would quietly poison a Phase 4 fine-tune. Frame size is unchanged, so no feature
shape, no capture profile and no test downstream had to move.

Two mechanical notes worth keeping. The crop runs *after* the base class's
rotation, so the box is centred on the picture as finally seen — right at 0° and
180°, meaningless at 90°/270°, where the output's width is the sensor's vertical
axis; that combination is rejected in `config.load` and again in the config
dataclass rather than cropped wrongly. And registering
`CroppedOpenCVCameraConfig` under `type: opencv_cropped` is not enough on its
own: `make_cameras_from_configs` has a hardcoded branch per built-in type and
falls through to `make_device_from_device_class`, which looks the *class* up by
name in the package holding the config's module. Same trap as
`SafeBiDK1Follower`, same fix — `dk1lab/__init__.py`'s lazy `__getattr__`.

**Checked on the real cameras** (video devices only; no arms, nothing energised):
all three connect and deliver 1280×720, the wrists report the 909×511 box, and
stills through the cropped path are upright, correctly framed and visibly
tighter. One thing the stills showed that the datasheet does not: the uncropped
105° frame has **black vignette corners** — the lens's image circle does not
quite cover the sensor — and the crop removes them entirely.

Checked end to end as well, on the same read-only path: real frames from all
three cameras pushed through the **real** preprocessor come out as three upright,
correctly ordered 378×378 tensors. That is the check `--display-policy-input`
now does live — see *Watching what the model sees*.

**What this does not fix.** The lens has real barrel distortion (the manual
specifies TV distortion < −6.2%, and it is obvious in the stills: straight edges
bow). The crop is a pinhole correction, so it matches the trained geometry at the
centre of the frame and only approximately at the edges; undistorting properly
needs a calibration this cell does not have. Mounting pose is untouched too —
the sim's wrist cameras sit at a specific place on `link_6` and ours sit where
they sit. So this narrows one known divergence and leaves two.

### The capture resolution: 378 rows in, so more than 378 rows out

`[capture.policy]` was raised from **640×360 to 1280×720** on 2026-08-20, once
the 378×378 model input was established. The arithmetic is the whole argument:

| | wrist crop | rows the model wants | verdict |
| --- | --- | --- | --- |
| 640×360 | 455×256 | 378 | **1.5× upsample** — detail the sensor never caught |
| **1280×720** | **909×511** | 378 | genuine downscale |

The top view goes 720 → 378 likewise, where it was 360 → 378 before.

**It costs nothing on the control loop, and that was measured, not assumed.**
All three cameras sustain **30.3 fps** at 1280×720 MJPG; `OpenCVCamera` decodes
on a **per-camera background thread**, so `async_read` picks up the latest frame
rather than paying for the decode; and our crop-and-resize costs **1.6 ms per
frame** on that thread (0.22 ms at 640×360). Read latency measured from the loop
is frame-rate bound and identical at both sizes — median 32–33 ms against a
33.3 ms budget either way. `dk1 config check --formats` confirms all three
cameras advertise the mode.

One consequence worth noting: cropping the **top** view to its trained 69.4° is
now arithmetically possible — it would keep 683×384, still above 378 — where at
640×360 it kept 341×192 and was not. It is left uncropped by request, but it has
moved from "impossible" to "an experiment we could run".

### Watching what the model sees: `--display-policy-input`, corrected

The flag existed before this round and was showing the wrong thing. It logged
the `observation.images.*` entries of the **preprocessor's output** — but
`molmoact2_pack_inputs` leaves those **untouched**, at the camera's own size and
dtype, and puts what the model consumes in `pixel_values`. So the panel showed a
picture that looked like the model's input, was captioned as the model's input,
and was in fact the robot-side view at a different resolution and aspect ratio —
the one thing `--display` already showed. It could not have caught a resize or
aspect bug, which is most of what it was for.

`dk1lab.trace.model_input_images` now reconstructs the real thing: un-patchify
`pixel_values` (27×27 patches of 14×14×3 → 378×378), undo the `mean 0.5 /
std 0.5` normalisation, scale to bytes. Camera names come from
`layout.IMAGE_KEYS`, which is the order the checkpoint's preprocessor pins, so
row *i* really is that camera. It returns `{}` rather than raising on anything
unexpected — a display must never take a rollout down.

`dk1 policy dryrun --display-policy-input` is the way to check camera
orientation: arms energised, nothing sent, and Rerun carries both the robot-side
view under `--display` and the model's own 378×378 under `policy_input/`.
Verified here against the live cameras: upright, correctly ordered, correctly
stretched.

**`dk1 teleop --display-policy-input` does the same thing while you drive**,
which is the better place for it: you can move a wrist by hand and watch the
model's view track. `dk1lab/modelview.py`:

- It builds the checkpoint's **preprocessor only** — `make_pre_post_processors`
  reads `policy_preprocessor.json` and the HF image processor and never touches
  the 10 GiB of weights. 0.6 s on the CPU, no GPU.
- It **attaches by wrapping**, like `dk1lab/trace.py`: `ModelInputProbe` proxies
  the observation processor `teleop_loop` already calls, returns its result
  untouched, and logs as a side effect. The loop stays upstream's.
- The work runs on a **background thread**. One pass costs ~11 ms and the 60 Hz
  budget is 16.7, so inline sampling would drop ticks; measured with the thread,
  the worst tick is 5.0 ms and the median 1.6 ms. One tick in 12 is sampled and
  anything arriving while the worker is busy is **dropped, not queued**.
- It **pins the Rerun layout**, and that is not optional. `log_rerun_data` builds
  a blueprint from the first observation it sees and caches it on itself; that
  blueprint is an explicit grid, so `policy_input/*` — which never passes through
  it — would get no view and be invisible until the operator built one by hand.
  Filling the cache before the loop starts is what stops that, because
  `_ensure_blueprint` returns early when it is already set. A coupling to an
  upstream implementation detail, written down as one.

### The 27.7 Hz loop: the capture raise is innocent, the per-tick engine was not

Two problems were open here. Both are now closed: the trace is fixed, and the
loop's cost is diagnosed **and** fixed in `dk1lab/fifo.py`. Nothing in this round
has been on the arms.

#### B (fixed): `--trace` no longer lies under `--sync`

The old accounting cut a record at the **postprocessor**, and the sync engine
runs the postprocessor every tick — so a 180 s run reported `chunk 4390`, one
per tick, each handed RTC's `real_delay` of 0 as its total. Hence
`0 ms = 0 ticks` and `other -160`: three timers minus a total of nothing.

A chunk boundary under sync is **the tick that actually ran the model**, and
nothing else is. MolmoAct2's `select_action` calls `predict_action_chunk` only
when its own 30-deep queue is empty, so wrapping `predict_action_chunk` — which
`dk1lab/trace.py` already did — is enough to detect one. `RolloutTrace` now
folds sync ticks into a `SyncWindow` between two inference calls and measures
what actually matters at 27.7 Hz: **where each tick goes**.

| | |
| --- | --- |
| `wall_ms` | measured wall clock across the window, not inferred from a delay |
| `ticks` | ticks in it — 30 here: one inference, 29 cached |
| `model_ms` | the one real forward pass, at the head of the window |
| `pre_ms` / `post_ms` / `select_ms` | per **cached** tick, median |
| `outside_ms` | tick period minus the engine call: everything else in the loop |

The inference tick is reported separately as `infer_ms` rather than averaged in,
because it carries the whole forward pass and would hide the per-tick cost the
other 29 are paying. `unaccounted_ms` under sync is wall clock minus **every**
engine call, so it cannot go negative. `total_ticks`, `consumed`, `queue_after`
and `starved` are RTC readings and are simply absent. `trace.close()` cuts the
window still open when the run ends, so the last chunk is not dropped.

Sample, from the real engine and the real cameras with no robot attached:

```
chunk   2    1156 ms over  30 ticks  = 25.9 Hz  (pause 186 ms, model 118)
  per cached tick   33.4 ms = 29.9 Hz  (pre 17.5 · select 5.6 · post 0.2 · loop 10.2)
```

#### A: it is not the capture resolution — measured, paired, three rounds

The suspicion was that raising `[capture.policy]` to 1280x720 added ~5 ms to
every tick, because the sync engine re-runs the whole preprocessor per tick.
**It did not.** The paired A/B — one model load, one process, cameras swapped
between 720p and 360p and back, three rounds, 145 timed cached ticks each:

| | engine per cached tick | `pack_inputs` |
| --- | --- | --- |
| 1280x720 | 22.72 ms | 15.90 ms |
| 640x360 | 21.83 ms | 15.86 ms |
| **difference** | **+0.89 ms** | **+0.04 ms** |

Round-to-round variance is ~1.5 ms, i.e. **larger than the effect**. So
reverting the capture would buy under a millisecond of the ~3 ms needed, and
would cost the policy the thing the raise was for: at 640x360 the tuned wrist
box is 455x256 and 256 rows get upsampled 1.5x into a 378-row model input.
**Do not revert it.** That option is closed.

The 6.1 / 11.0 ms figures this section previously carried are not reproducible
and were measured some other way. Two confounds that produce exactly that kind
of error, both hit while chasing this:

- **A flat-out loop and a paced one disagree about the same function.** Run
  back-to-back, `pack_inputs` costs 7.5 ms at 360p and 14.0 at 720p — resolution
  matters and the effect is large. Paced at 30 Hz, which is what a rollout does,
  both sit at ~15.5 ms. Benchmark the duty cycle you are going to run.
- **Separate processes drift.** Run one resolution per process, the engine reads
  23.2 vs 20.0 ms and the story looks confirmed. Feed both to the *same* loaded
  engine and the gap collapses to 0.89 ms. The 3 ms was between-run variance.

Not the GIL: `thread_time` equals wall time inside `pack_inputs` to within
0.06 ms with all three camera decode threads running, so it is real CPU burn in
the calling thread, not waiting behind the cameras. Not core placement either —
pinning to a P-core changes nothing (the machine is a 14900K on the `powersave`
governor, so this was worth ruling out).

#### A: what it actually is — 22 ms of engine on a 33.3 ms tick, thrown away 29 times in 30

Measured with the real `SyncInferenceEngine`, the real bf16 checkpoint and the
real cameras, paced at 30 Hz, **no robot attached**:

| per cached tick | |
| --- | --- |
| `MolmoAct2PackInputsProcessorStep` | **15.9 ms** |
| the rest of the preprocessor (rename, batch, normalise, clamp, device) | 0.8 ms |
| `select_action` and the engine's own per-tick work | **5.5 ms** |
| postprocessor (clamp, unnormalise, frame transform, device) | 0.2 ms |
| **the inference engine, total** | **~22 ms of a 33.3 ms budget** |

and once per chunk, a **118 ms** model call inside a **~190 ms** engine call.

On 29 of those 30 ticks **every millisecond of it is discarded**:
`select_action` looks at its queue first, finds it non-empty, and returns
`popleft()` without reading the batch at all. So the three camera views are
resized to 378x378, patchified and normalised, and thrown away, thirty times per
chunk instead of once.

Two smaller findings inside that 5.5 ms: `select_action` calls `self.eval()` on
every call, and walking 1737 submodules of a 7B model costs **1.83 ms** each
time; the remainder is `prepare_observation_for_inference`, the `copy`, and
`action.squeeze(0).cpu()` synchronising the device for 14 floats. The
"29 calls in 30 cost ~12 ms" line recorded earlier in this file is a *total* for
29 calls; per call it is ~5.5 ms, and only ~0.4 ms of that is the queue.

The arithmetic against the in-situ 36.1 ms tick: the engine is ~22 ms of it, so
the robot reads, the limiter and the dataset write are ~14 ms. Remove the engine
from the 29 cached ticks and the tick is ~14.5 ms — 30 Hz with room to spare.
Revert the capture instead and it is ~35.2 ms, still 28.4 Hz. Only one of those
two is a fix.

#### The fix: `dk1lab/fifo.py`, built and measured, not yet on the arms

`ChunkFIFOInferenceEngine` is what LeRobot's own comment block above
`SyncInferenceEngine` sketches: run the model once, postprocess the whole chunk
at once, queue the rows, hand out one per tick. It is a drop-in for the sync
engine — same `InferenceEngine` lifecycle, same `get_action` contract — and
deliberately **not a subclass**, because the only method it would inherit is the
one it replaces.

**What it is worth**, measured on the real checkpoint and the real cameras,
paced at 30 Hz, no robot, paired in one process with the order alternated:

| | engine cost on a cached tick | the model tick |
| --- | --- | --- |
| `SyncInferenceEngine` | 23.2 ms | ~186 ms |
| `ChunkFIFOInferenceEngine` | **0.02 ms** | ~193 ms |

0.02 ms is a `deque.popleft()`. That takes **23.2 ms out of 29 ticks in every
30**; the in-situ tick of 36.1 ms becomes ~13 ms against a 33.3 ms budget, so
the loop stops being the binding constraint rather than merely clearing the bar.
The once-per-chunk pause is unchanged and is meant to be: it is inherent to
synchronous inference and is what `--rtc` exists to remove.

**The actions are bit-identical.** Thirty rows from each engine, same frozen
observation, same seed, the real bf16 checkpoint: max absolute difference
`0.000e+00` on all 14 channels. That is the equivalence claim tested directly
rather than argued.

Why the equivalence holds, and what it rests on:

- `select_action` slices the same `predict_action_chunk` output to
  `n_action_steps` and pops it in order; this does the same slice, same order.
- All four postprocessor steps — clamp, unnormalise, action-frame transform,
  device move — are stateless and elementwise, so postprocessing a chunk equals
  postprocessing its rows. RTC already does exactly this on the same pipeline.
- The observations dropped were dropped anyway: on a cached tick `select_action`
  never reads the batch.
- It requires **absolute** actions. A relative-action policy re-anchors to the
  current state every call, so a precomputed chunk would drift — upstream's own
  fourth caveat. `dk1lab/fifo.py` **checks** for a relative step in the pipeline
  and raises rather than assuming; ours is absolute joint pose.
- Upstream's other three blockers are SAC (raises from `predict_action_chunk`),
  ACT (ensembler inside `select_action`) and the Diffusion family (obs-history
  queues filled as a side effect). MolmoAct2 has none of them.

A bonus, since `select_action` is no longer on the per-tick path: `self.eval()`
now runs once in `start()` instead of walking 1737 submodules every tick.

Worth noting as corroboration: **`dk1lab/serve.py` has always worked this way** —
it calls `predict_action_chunk`, postprocesses the chunk whole and returns it,
because that is what the `/act` protocol is. That is the server that scored 3/3
in ManiSkill. So the arrangement the FIFO brings to the rollout is the one our
only successful evaluation of this checkpoint already ran under.

**Wiring.** `policy.use_chunk_fifo(ctx)` replaces `ctx.policy.inference` between
`build_context` and `strategy.setup` — `BaseStrategy._init_engine` keeps
whatever it finds there, so that attribute is the whole of the seam and nothing
upstream is touched. It runs **after** `build_context` so `prewarm` has already
built the CUDA graph, and **before** the trace attaches so the trace wraps the
engine that will actually be driven. The pipelines are carried across by
reference, so a gripper inversion already applied to them still applies.

`--fifo` is **on by default** for `dk1 policy run` and `dk1 policy dryrun`, and
is a silent no-op under `--rtc`, which already serves chunks whole. `--no-fifo`
exists to measure the difference on the same rollout, not for ordinary use.

**What is still unverified:** it has never driven the arms. The equivalence is
established on this machine against the real weights, and the tick saving is
measured on the real cameras, but the number that matters — whether the in-situ
loop actually holds 30 Hz — needs a rollout. That is the same rollout that would
close the last open question in this section: the in-situ `pre 27 ms` against
15.9 measured here, ~11 ms most likely lost to contention with the follower
serial reads. Fold both into the next run; the fixed `--trace` reports them.

The machine's CPU governor stays on `powersave` by request. It was ruled out as
a cause here.

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

- **No RTC.** `sim_eval` blocks on each response and executes the whole 30-step
  chunk, so there is no deadline, no latency to compensate, no queue to starve
  and no seam to blend. Read as a control for the arms' timing faults at first;
  it turned out to be the *arrangement that works*, and `dk1 policy run` now
  defaults to the same thing (`--sync`). See *The stall*.
- **No gripper inversion.** The wire protocol *is* the YAM convention —
  `sim_eval`'s own `yam_state_adapter` sends `grip in [0,1] (1=open)` and
  `yam_action_adapter` maps `1.0` back to ManiSkill's `-1.0` = open. That is a
  **third** independent statement of the convention, and it means the sim tests
  the policy with the DK1's gripper sign out of the picture entirely.

`BimanualYAMPutEverythingInBox-v1`, "put everything into the box".

**The first day's answer was 0/10, and it was wrong — the episodes were too
short.** 800 steps is 27 s at 30 Hz. The policy barely acts for the first ~30 s,
so every one of those episodes was scored before the policy had started. Also
0/3 at `--n-action-steps` 10 and 5, which ruled out the open-loop chunk length
and made the too-short-episode explanation look less likely than it was.

**With 3600 steps (120 s), 2026-08-20:**

| server | checkpoint | result |
| --- | --- | --- |
| `dk1 policy serve` | our LeRobot bf16 copy | **3/3** |
| `examples/yam/host_server_yam.py` | `allenai/MolmoAct2-BimanualYAM`, HF | ~50% |

Successful episodes averaged **1627 steps = 54 s**. So the policy does the task
zero-shot on its own embodiment, and our packaging is not subtly wrong — it beat
the reference path on the same task, budget and seed. That closes the one thing
the previous round left open. Episode outcome is stochastic (flow-matching
sampler, and the server holds its own RNG), so treat 3/3 vs ~50% as "ours is at
least as good", not as a measured margin: a later single 3600-step episode at
seed 42 recorded here did *not* succeed.

Supporting numbers from a recorded 120 s episode (state and action logged every
tick, correct 16-D `qpos` mapping — the sim interleaves left/right, `qpos[2k]`
is left joint k+1 and `qpos[2k+1]` is right):

| | |
| --- | --- |
| tracking, median \|commanded − measured\| | **0.0018 rad** (p95 0.025) |
| travel per arm joint, total | 5.2 – 17.9 rad |
| demanded joint rate | median **0.036** rad/s, p95 **0.31**, max **4.56** |
| gripper commanded | left −1.00 → +0.94, right −1.00 → +0.70 |

The simulator executes what the policy asks to within a couple of milliradians,
and the policy opens and closes both grippers. The camera views were checked by
eye and are well-formed.

**The gripper convention now has a fourth, behavioural confirmation.** The sim
succeeds while mapping the policy's `1.0` to *open*. So the checkpoint really
does speak YAM (1=open) and the DK1 really is 0=open — which means
`--invert-gripper` is very likely **required** on hardware, and the current
default of off is very likely wrong. Still not watched on the arms; still a flag.

The colleague's `sim-eval-dk1` branch turns out to carry a **full DK1 in sim** —
`bimanual_dk1.urdf`, the meshes, `robots/bimanual_dk1.py`, a DK1 task and DK1
adapters — so running our own embodiment in sim is a checkout away, not a build.
That is the agreed next step after YAM.

### What the caps were doing, and the numbers that raised them

Measured 2026-08-20 by recording a 120 s sim episode of this checkpoint and
replaying its commanded joint targets through the **real** `SlewLimiter`.

The policy's motion is **bursty, not fast**: median demand 0.036 rad/s, p95 0.31,
peak 4.56. So the old 0.3 rad/s cap did not slow everything evenly — it truncated
exactly the reach and transit moves and left the slow parts alone. 30% of ticks
had at least one joint demanding more than the cap.

Replayed through the limiter (best case: a perfectly tracking arm, so `max_lag`
never binds — real hardware is worse):

| cap | worst joint ends up behind the policy's intent | ticks >0.1 rad behind |
| --- | --- | --- |
| 0.3 rad/s (old) | **0.98 rad = 56°** | 25.6% |
| **1.0 rad/s (now)** | 0.40 rad = 23° | 3.3% |
| 2.0 rad/s | 0.10 rad | 0% |

**`max_lag` was the worse of the two, and it is not a position clamp — it is a
torque clamp.** Impedance torque is `arm_kp * (q_des − q)` with
`arm_kp = [100, 100, 100, 20, 20, 10]`, so `max_lag = 0.1` held the PD torque to
10 Nm on j1–j3 (motor limit 28), 2 Nm on j4/j5 (limit 10) and **1 Nm on j6**
(limit 10) — a tenth of the wrist's authority, on the joints the policy drives
fastest. And because `limiter.limit` writes the *clamped* value back into
`_prev_cmd`, the command can never build more lead: a joint that cannot break
stiction within 0.1 rad stalls there silently and permanently. That is precisely
the deadlock `dk1lab/limiter.py`'s own docstring argues against when it explains
why the ramp anchors to the previous command rather than to the measurement.
Now **0.4 rad**. `joint_torque_limits` downstream is the thing that should be
bounding torque, and it still is.

Note the earlier "±1.8 rad excursion ÷ 0.3 rad/s = six seconds" argument was
wrong in kind: ±1.8 rad was a *position range*, not a per-tick demand. Total
travel on the busiest joint (17.9 rad over 120 s) averages 0.15 rad/s, under even
the old cap. The cap's damage was to the peaks, not to the average.

### The stall: what LeRobot does with a chunk, and why `--sync` is now the default

Traced through LeRobot 0.6.1. `interpolation_multiplier` is 1, so there is no
interpolation — one chunk row per tick:

```
RTC thread:   predict_action_chunk -> 30x14 -> postprocessor
              -> ActionQueue.merge: blend with the previous tail,
                 DISCARD real_delay rows from the front
control loop: engine.get_action() -> one 14-D row -> ActionInterpolator
              -> robot_action_processor
              -> SafeBiDK1Follower.send_action   <- the limiter
              -> command_joint_pos -> 250 Hz impedance server
```

**LeRobot does not clip actions anywhere.** The processor pipeline has no clamp
on the action path; absolute joint angles pass through untouched. Everything that
changes the numbers is RTC's blend, our limiter, or upstream's position/torque
clamps. And when the queue is empty, `rollout/strategies/core.py:297` returns
`None` and **nothing is sent at all** — the motor chain holds its last target and
warns after `command_timeout_s = 0.5`. That is the stall, exactly.

The fix is not a tuning constant. `n_action_steps` is 30 on this checkpoint and
`select_action` serves from a 30-deep queue, so the **sync** engine executes the
whole chunk and then blocks for one model call — which is byte-for-byte the
arrangement `sim_eval` uses, and `sim_eval` scores 3/3. RTC, at the measured
900 ms in-situ latency, discarded 27 of every 30 actions and delivered 100 ms of
motion per second. Sync pays one visible pause per chunk and delivers all 30.

`dk1 policy run` therefore defaults to `--sync`. `--rtc` is still there and is
the right answer once in-situ inference is well under `chunk / fps` = 1 s; the
620 ms of unexplained in-situ latency is still unexplained, and `dk1lab/trace.py`
is still the instrument for it. Sync makes it non-blocking rather than solved.

`--duration` now defaults to **180 s**, because the policy is slow: ~30 s before
it does much, 54 s for a successful sim episode. A 30 s rollout cannot succeed.

## Environment

RTX 5090 (32 GB), Python 3.12, uv, LeRobot **0.6.1**, fish shell.
Old project for reference (read-only): `~/Documents/RobotLearning/trlc-dk1`,
branch `wip/molmoact2` — real knowledge in `MOLMOACT2.md` / `MOLMOACT2_EVAL.md`,
but its `molmoact2/` scripts beyond `smoke_test.py` and `convert_bf16.py` were
never run. Its two docs contradict each other on whether zero-shot will work;
neither is a result.

`uv run pytest -q` · `uv run dk1 --help`
