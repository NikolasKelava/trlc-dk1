# CLAUDE.md — architecture and project state

Read this first. It is what is true now, and why the code is shaped the way it
is. Three companions, none of which repeats this one: `GUIDE.md` is the
operator-facing version, **`STUDY.md` is the protocol** for the two-policy
comparison, and **`docs/DIAGNOSTICS.md` is the record** — every
measurement, every fault chased to its cause, and the hypotheses that turned out
wrong. Sections below point into it as `DIAGNOSTICS § name`. Read the section
before re-measuring or undoing anything it covers.

## What this is

A fork of [robot-learning-co/trlc-dk1](https://github.com/robot-learning-co/trlc-dk1)
set up to operate a bimanual TRLC-DK1 cell (2 leader arms, 2 follower arms, 3 USB
cameras) with LeRobot, and to evaluate and fine-tune the **MolmoAct2** VLA policy
on it. Origin is `NikolasKelava/trlc-dk1`; upstream is the hardware repo and we
want to keep pulling its updates.

> **The machine froze six times in three days. It is fixed and the file is
> closed. Do not open `docs/CRASH.md`.** It was the platform firmware, not this
> code: the BIOS went **F6 -> F8a** on 2026-08-27 and the machine has been stable
> since. The firmware had recorded three FATAL Intel SoC internal error records
> all along, which is why nothing ever reached the kernel log.
>
> Open it in exactly one case — **the machine freezes again** — and then read
> only § *How it was found*, which is four steps long. Everything else in that
> file is a superseded hypothesis. Reading it as background will cost you an hour
> and teach you things that are no longer true.

## Hard rules

- **Never open a PR, never push.** Commit locally on a branch; Nikolas publishes.
- **Ask before every commit, and commit late.** One commit at the end of a
  session, or a small self-contained fix on its own — never a running series of
  commits as the work goes. Nikolas confirms each one. The point is that the
  history stays reviewable and nothing lands before it has been verified on the
  cell; work that is built but unrun is a working tree, not a commit.
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
  session.py            one loaded policy, many rollouts — the task is a prompt
  record.py             an episode to a Rerun .rrd: images, plan, command, state
  trace.py              per-chunk latency, queue depth, and the policy's OWN action
  modelview.py          the model's-eye view, live, during teleoperation
  actionview.py         policy plan vs robot command vs measured, per joint, live
  fifo.py               ChunkFIFOInferenceEngine — one model call per chunk
  serve.py              the /act HTTP endpoint sim_eval drives — no robot
  runprofile.py         optimized vs common: what the policy sees, how fast (no lerobot)
  pi05.py               pi0.5: 32-D padding, renamed cameras, borrowed norm stats
  dataset.py            the LeRobot v3.0 recorder — alongside record.py, not instead
  demos.py              teleop demonstrations: Enter starts, Enter stops, `again` deletes
  study.py              the score sheet: scenes, attempts, the CSV  (no lerobot import)
  finetune.py           the LoRA recipe, the hold-out, the run directory (no lerobot import)
  recrop.py             the optimized crop, applied to a recorded dataset
  logs.py               a session log file, fsynced per record       (no lerobot import)
  telemetry.py          PSU, CPU and GPU once a second, fsynced      (no lerobot import)
  scene.py              the bimanual MuJoCo scene, generated from urdf/  (no lerobot)
  sim.py                SimRobot — the MuJoCo cell behind the real robot interface
  cli/                  Typer app; `dk1` entry point
dk1.toml                THE device config. Tracked. Single source of truth.
tests/                  the suite; none of it needs hardware
GUIDE.md                operator docs
docs/DIAGNOSTICS.md     the record: measurements, faults, discarded hypotheses
docs/HUB.md             pushing the datasets, the .rrd and the checkpoints; and getting them back
docs/CRASH.md           CLOSED. The machine freeze of 2026-08-25..27 — do not open it
study/results.md        THE RESULTS: every scored attempt, its provenance, how to continue
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
The numbers, the arithmetic and what the crop does *not* fix: DIAGNOSTICS §
*The camera crop*. Read the box off `dk1 config show`, never from prose.

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
  Never write that they self-zero *closed*: the inherited wording said so, and a
  safety notice pointing at the wrong hazard is worse than none.
- **Stopping moves the arms only through our home sweep**, which is now **on by
  default** for `run` and `session` (`--no-home` opts out) because leaving them
  wherever the policy stopped is what wears them. It runs on a clean end only —
  the duration limit or Ctrl-C — and never after a fault.
  `return_to_initial_position` defaults to `true` in LeRobot's rollout — it is
  forced `false`, always, including under `--home`. And note what "the arms are
  not commanded" does not mean: a clean disconnect in impedance mode reaches
  `DK1MotorChain.stop()`, which **disables every motor** — so a raised arm sags.
  Support anything held up. That is also why a home sweep that does not arrive
  is reported loudly: the motors go off a second later.
- **Homing is ours, and does not run after a fault.** `dk1lab/home.py`, reached
  by `dk1 policy run` (on by default since 2026-08-21), `dk1 policy session` and
  `dk1 policy home`. LeRobot's built-in return-to-home is wrong here on three
  counts and is not used: it targets the
  connect-time pose; it interpolates for a fixed 3 s and disconnects whether or
  not the arms arrived (behind the speed cap — anything over ~3 rad at the
  current 1.0 rad/s, and it was ~0.9 rad when the cap was 0.3 — cannot finish);
  and it fires from `teardown` on every exit path including a crash.
  Ours ramps from the previous command, tests arrival against the *measurement*,
  derives its timeout from the distance, runs on a clean end only (duration limit
  or Ctrl-C — `policy.ended_cleanly`), and takes SIGINT for the length of the
  sweep so a second Ctrl-C stops it where the arms are instead of `sys.exit(1)`
  mid-command.
  **Its speed is its own — 0.3 rad/s, eased at both ends — and not the policy's
  cap.** DIAGNOSTICS § *The home sweep speed*.
- **A session keeps the arms energised while nobody is driving them.**
  `dk1 policy session` connects once and stays connected between rollouts —
  that is the whole point of it — so live motors sit in the room while the
  operator types. Nothing is commanded between episodes, so each arm holds its
  last target and the chain warns once (`No command for 0.50 s`); that warning
  is expected there and nowhere else. Quitting disconnects, which disables every
  motor. Every one of those facts is in the command's banner and its `--help`.
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

## MolmoAct2 facts

Checkpoint: `lerobot/MolmoAct2-BimanualYAM-LeRobot` (LeRobot format);
`allenai/MolmoAct2-BimanualYAM` is the HF-format equivalent.
14-D state and action, absolute joint pose, chunk 30 @ 30 Hz,
`norm_tag=yam_dual_molmoact2`, `control_mode="absolute joint pose"`.

- **The converted bf16 checkpoint** is at
  `~/Documents/RobotLearning/trlc-dk1/outputs/molmoact2_bimanual_yam_bf16`
  (10.1 GiB). Reuse it. Its `config.json` bakes in `"device": "cpu"` and a stale
  absolute `pretrained_path`; both are overridden at load, or it silently loads
  on the CPU and runs there.
- **The arm joint map is identity** — the colleague's sim says "verified by FK",
  and every YAM joint range in the checkpoint's norm statistics fits inside the
  DK1's configured limits.
- **The model input is 378×378**, a *stretch* of the 16:9 frame, not 224×224 and
  not a letterbox. The 224s in the checkpoint normalise nothing. This decides
  what a crop may throw away — DIAGNOSTICS § *The camera crop*.
- **Image order is pinned in the checkpoint itself**: `policy_preprocessor.json`
  carries `["...top","...left","...right"]`. The hazard is real only when
  *training* rebuilds the processor from a new dataset's features, so pin
  `--policy.image_keys` for training anyway.
- **The gripper channel is inverted, and the inversion is ON by default.** The
  checkpoint speaks YAM (**1 = open**); the DK1 is **0 = open**
  (`DK1Robot.command_gripper`). Four independent sources agreed, the fourth
  behavioural — the ManiSkill run succeeds while mapping the policy's `1.0` to
  open — and **Nikolas confirmed it on the arms (2026-08-21)**. It was a flag,
  off by default, for exactly as long as that was an argument rather than an
  observation; it is now the default on `run`, `session`, `dryrun` and `smoke`.
  `--no-invert-gripper` re-tests it. **`dk1 policy serve` keeps it off** and
  must: `sim_eval` already sends 1 = open on the wire, so inverting there would
  introduce the sign error the sim exists to be free of.
  `layout.yam_joint_signs()` / `yam_joint_offsets()` implement it.
- **`--policy.joint_signs` does nothing at rollout.** `make_pre_post_processors`
  rebuilds both pipelines from the checkpoint's JSON whenever a policy is loaded
  from a path — which is every rollout — and the config field is read only on
  the build-from-scratch branch. `dk1lab.policy.apply_gripper_inversion` patches
  the two *loaded* pipeline steps instead, and raises rather than warns if it
  cannot find them. Same class of upstream bug as the two cherry-picks: worth
  upstreaming, do not do it.
- **The cameras do not match the training rig.** Theirs: top 69.4° HFOV
  (D435i), wrists 87° (D405), 640×360, arms at y = ±0.24 m both facing +X.
  Ours: Innomaker U30CAM-4K-S1, **105° HFOV / 116° diagonal**, confirmed model
  string off the USB descriptor. Too *wide* is fixable, which is why the wrist
  crop exists; the lens's barrel distortion and the mounting pose are not fixed.

## Evidence status — keep this line sharp

**`study/results.md` is the record — read it before quoting any number.**
It carries every scored attempt, where its frames are, and the provenance of
each figure. **It carries no interpretation yet, on purpose**, and neither does
this section: three of the study's six rows have numbers, one of those is a
third of a row, and a comparison written now would be quoted long after the
missing rows arrived. `docs/HUB.md` is where the data lives — the datasets, the
`.rrd` and the checkpoints are far past what git can hold.

**Two zero-shot rows are scored, and both are 0/9 with a ceiling of grasp. One
fine-tuned row is a third scored.**

| highest step reached, three attempts per scene | scene 1 | scene 2 | scene 3 | success |
| --- | --- | --- | --- | --- |
| **R0** MolmoAct2 zero-shot, `optimized` (2026-08-28) | 3, 3, 3 | 3, 3, 3 | 2, 2, 2 | **0/9** |
| **A0** MolmoAct2 zero-shot, `common` (2026-08-27) | 2, 2, 2 | 3, 3, 3 | 3, 1, 1 | **0/9** |
| **R1** MolmoAct2 + LoRA @4 000, `optimized` (2026-08-28) | **5, 5, 2** | — | — | **partial, 3 of 9** |

Read the rubric in `STUDY.md` before quoting any of that. **R1 is three
attempts at one layout and is not a row**; do not set it against R0 or A0 until
its other six exist. Its checkpoint is step 4 000 of an unfinished 8 000-step
run, which is itself an open protocol question — `STUDY.md`'s amendment of
2026-08-28.

**The right arm is not the weak one.** Across A0's nine it scored 3 in all four
attempts it was used for against the left's 2, 2, 2, 1, 1; R0 split evenly by
scene and shows nothing either way. That is enough to retire "the right arm does
not pick anything up" and not enough to be a study of arms.

What *is* settled:

| | |
| --- | --- |
| The policy works zero-shot **in simulation**, on its own embodiment | 3/3 on `BimanualYAMPutEverythingInBox-v1` with a 120 s budget, 2026-08-20. Sai's reference HF server scored ~50% on the same task. The earlier 0/10 was a 27 s episode budget, not a failure |
| Our bf16 packaging is not subtly wrong | same checkpoint, our `/act` server, ahead of the reference path |
| The control loop is not the problem any more | 29.9 Hz over 335 chunks on the arms, zero starved ticks, `robot read+send` 0.4 ms |
| The roughness that is left is **the policy's own output** | read off `--display`'s per-joint panels: the plan itself is not a smooth trajectory, and command and measurement follow it |
| The wrist crop helped | fourth rollout: alignment better, and the gripper waits for a good position before closing |
| The policy uses the gripper channel | commanded +0.033 .. +1.000 on the arms |
| The gripper inversion is right | confirmed on the arms by Nikolas, 2026-08-21. It is now the default |
| The session and the recorder work on the arms | eight episodes recorded 2026-08-21, four tasks, all four streams present in every file |
| The teleop dataset recorder works on the arms | one throwaway episode, 2026-08-27: 628 frames at 30 Hz under `common`, read back as a LeRobot v3.0 dataset, then deleted. A crash mid-session keeps every committed episode and the directory resumes — verified by killing the process |
| The home sweep works | run on the arms, including the eased profile. Now the default at the end of a run |

What is not:

- **what the fine-tune bought.** R1 has three attempts at one scene. Six more
  are needed before it is a row, and A1 — the same LoRA recipe under `common` —
  has not been trained at all;
- **anything comparing the rows.** R0 and A0 differ by lens *and* speed cap and
  both scored 0/9; R1 differs from R0 by the LoRA alone but is a third of a row.
  The interpretation section of `study/results.md` is deliberately unwritten;
- **π0.5 on these arms.** B0 and B1 are unrun and blocked on the gated
  `google/paligemma-3b-pt-224` licence;
- anything about the crop retune (inset 6, view lifted 40) or the 1280×720
  capture beyond "it ran";
- **why all three cameras timed out at once** in one session on 2026-08-28
  (`TimeoutError ... read failed (status=False)`), killing an episode. One
  occurrence, no diagnosis.

Verified on hardware earlier, and not to be re-derived: the LeRobot plugin
classes; the motor/impedance/grav-comp stack; bimanual teleoperation with and
without cameras; all three cameras report serial `20010101`, so
`/dev/v4l/by-id` is unusable; MJPG is mandatory (YUYV at 720p60 exceeds the UVC
bandwidth allocation); all three cameras are mounted upside down; all three
advertise both `[capture.*]` profiles; the `top`/`left`/`right` labels are
correct; the four arm ports in `dk1.toml` are correct, settled by USB identity
(followers `2e88:4603`, leaders `1a86:55d3`) and confirmed by Nikolas; `0 = open`
on the real grippers, and connecting does not close them; the home pose is
captured (2026-08-19: zero pose, grippers open) and the sweep has run — 8.5 s,
worst joint 0.028 rad off, and again since with the eased profile. Homing is
settled; do not describe it as untested.

Two traps that cost a day each and will cost another: `make_robot_from_config`
and `make_cameras_from_configs` both look a class up **by name** in the package
holding its config's module, which registration alone does not satisfy —
`dk1lab/__init__.py`'s lazy `__getattr__` is what makes `SafeBiDK1Follower` and
`CroppedOpenCVCamera` findable. And a by-path string is unique only *within* a
subsystem directory: `...-usb-0:4.3:1.0` names both a camera and a leader arm.

## Phases

| | | |
| --- | --- | --- |
| **0** | Foundation — package, config, CLI, limiter, tests | **done**, branch `phase0-foundation` |
| **1** | Device discovery on the hardware | **done** |
| **2** | Teleoperation | **done** — run on the arms, limits tuned |
| **3** | Zero-shot MolmoAct2 evaluation | **done, and scored** — six debugging rollouts, then A0's nine labelled attempts on 2026-08-27. Every timing and motion fault this fork could cause is closed; what is left is the policy's own output, and it is 0/9 with a ceiling of grasp |
| **3s** | The same policy in ManiSkill, via the colleague's `sim_eval` | **done: 3/3** |
| **4** | Record + LoRA fine-tune | **done, and deployed.** 26 demonstrations recorded 2026-08-28 (`study/demos`), the R1 LoRA trained to ~4 400 of 8 000 steps (`study/finetune/R1-20260828-132023`), and its 4 000-step checkpoint taken to the arms the same day |
| **5** | The two-policy comparison — MolmoAct2 vs π0.5, one task, N=9 per row (3 scene configurations x 3 attempts) | **in progress**, protocol in `STUDY.md` and results in `study/results.md`, both of which carry their own phase numbering. Its Phases 0–4 are done: **A0 0/9, R0 0/9, 26 demonstrations recorded, the R1 LoRA trained and partly deployed (3 of 9)**. Next is finishing R1's other six attempts |

**Phase 1** built `dk1 find cameras`, `dk1 find arms --inspect` (read-only USB
identity) and `dk1 config check --formats`. That last one matters: OpenCV
accepts an unavailable capture size and silently substitutes the nearest one it
has, which would hand the policy a different aspect ratio than training used.

**Phase 2.** `dk1 teleop` is the single entry point, `dk1lab/teleop.py` the
single implementation, and the loop is LeRobot's `teleop_loop` imported rather
than reimplemented — because recording and rollout run that same loop, and a
bespoke one here could work while the one every later phase depends on does not.
**`--record-dataset` is the one exception in this fork**, and `dk1lab/demos.py`
says why in its docstring: `teleop_loop` takes a duration and nothing else, so it
cannot be ended on a keypress, and it prints a line per tick over what the
operator is typing. `demos.tick` is the six calls it makes, in its order, kept as
its own function so the copy is short enough to read beside the original.
**Teleoperation runs with no speed cap** (`[limits.teleop] max_joint_rate =
false`): that is upstream's native behaviour, it is what Nikolas asked for after
the first run felt sluggish, and it is right for the activity, since the
commands come from a human hand. **It does not extend to Phase 3** — a policy is
exactly the case the limiter was written for. Note also that **connecting a
leader is motion**: `DK1Leader.configure` torques the gripper servo and drives
it open, so a finger resting in a trigger gets pushed.

**Phase 3** — the commands, in escalating order of risk:

| | | risk |
| --- | --- | --- |
| `dk1 policy check` | reads the checkpoint's JSON | none — no GPU, no robot |
| `dk1 policy smoke` | loads it, runs inference on a synthetic frame | GPU only, nothing connected |
| `dk1 policy serve` | the same policy over HTTP for the sim | GPU only, nothing connected |
| `dk1 policy dryrun` | full deployment path, actions **printed, never sent** | arms energised, no pose commanded |
| `dk1 policy run` | the rollout | the policy drives the arms |
| `dk1 policy session` | load once, then rollout after rollout, task by task | the policy drives the arms |
| `dk1 policy home` | the home sweep alone, no model | both arms move |
| `dk1 policy finetune` | the LoRA training run | GPU only, nothing connected |
| `dk1 policy curve` | which checkpoint to deploy, off the log | none — two text files |
| `dk1 policy pause` | ask a running fine-tune to stop at its next checkpoint | none — writes one file |
| `dk1 policy resume` | continue a fine-tune from its last checkpoint | GPU only, nothing connected |
| `dk1 dataset check` | is the recorded dataset what we meant? | none — metadata only |
| `dk1 dataset crop` | the optimized crop, into a copy of a dataset | the video encoder, nothing else |
| `dk1 dataset clamp` | repair a gripper command the robot never executed | none — parquet and JSON |
| `dk1 study photo` | one still of a scene layout | a video device, nothing else |
| `dk1 study scores` | reads a scored row's CSV back | none — a text file |
| `dk1 doctor watch` | samples PSU, CPU and GPU once a second | none — sysfs and nvidia-smi |
| `dk1 doctor report` | reads the last telemetry file back | none — a text file |

`dk1lab/policy.py` is the single implementation; `dk1lab/checkpoint.py` is the
JSON-only reader behind `check`; `dk1lab/session.py` holds the loaded policy
across rollouts. Settings are decided in code, not left to a command line —
image keys pinned, `inference_action_mode = continuous`, bf16 on cuda,
`return_to_initial_position` forced false always, including under `--home`.

**This phase has its score.** A0: nine labelled attempts, 0/9 successes, ceiling
step 3 — it grasps the dice and never delivers it. The `arm` column also closed
the oldest loose end here: **the right arm is the better one**, 3 in all four of
its attempts against the left's 2, 2, 2, 1, 1. "The right arm does not pick
anything up" was the policy on the day, and can be retired.

**Phase 4** — record → LoRA from the same checkpoint → deploy → scored,
labelled eval attempts.

### The session, and recording an episode

Both added 2026-08-21, for the scoring run Phase 3 now needs, and **both ran on
the arms the same day**: eight episodes recorded and kept over four tasks
(dice, red dice, dice-in-bowl, pen-in-cup, marker, ball-in-bowl). Each file
verifies clean and carries all 45 entities — 14 policy, 14 command, 14
observation, 3 cameras. No score was reported. The workflow they exist for, and
the one to keep working:

```
dk1 policy session --record          # load once, connect once
  task> pick up the dice             # the arms go
  ^C                                 # the attempt is over, either way
  keep this episode? [Y/n]           # Enter keeps recordings/0007_pick-up-the-dice.rrd
                                     # then both arms sweep home
  task>                              # Enter runs the same instruction again
  task> :quit
```

`--invert-gripper` and `--home` are on by default, so that command line is
complete as written.

**`dk1lab/session.py` splits loading from rolling out.** `PolicySession.open()`
does everything expensive once — weights, CUDA graph, cameras, both CAN adapters
— and `rollout(task)` runs one episode. What changes between episodes is one
string: the instruction lives in the inference engine's `_task`, which
`prepare_observation_for_inference` reads at every model call, so writing it is
enough. `set_task` **raises** if the engine has no such attribute rather than
running the previous task quietly, because a session that reported one
instruction and executed another would produce evidence about the wrong thing.

Three things it must keep doing:

- **The instruments attach once and are reset per episode.** They wrap methods
  on live objects; re-attaching per episode would stack a layer every time.
  `RolloutTrace.reset()` exists for this. A recorder is per-episode by nature
  and attaches *last*, so detaching it restores the chain exactly.
- **The engine is stopped after every episode and started again by the next
  `strategy.setup`.** Otherwise the async FIFO's worker keeps running forward
  passes on the GPU while the operator is thinking.
- **SIGINT is ours for the length of a rollout.** LeRobot's
  `ProcessSignalHandler` counts signals for the life of the process and
  `sys.exit(1)`s on the second — which in a session is the second rollout you
  stop, killing the process with the arms energised. `session.interrupt_stops`
  replaces it: first signal ends the episode, second raises normally.

**`dk1lab/record.py` writes one episode to a Rerun `.rrd`** — `--record` on both
`run` and `session`. Four streams and no fewer: the camera images (after the
rotation and the crop, since both live in the camera), the policy's own plan,
the command `send_action` returned, and the observation. The layout written is
`actionview.build_blueprint`, the same one the live panel pins, so a replay is
the thing that was watched.

Named `<index>_<task>.rrd`, where the index is read off the recordings
directory and counts up across sessions — a session-local counter would
overwrite yesterday's first episode with today's. The task is in the filename
*and* in the file. **`recordings/` is tracked, not ignored**, at Nikolas's
request: these files are for colleagues. Note what that costs — the first eight
episodes are **150–910 MB each, 3.9 GB in total**, and every one of them is over
GitHub's 100 MB per-file hard limit, so they cannot be pushed without Git LFS. That is why the operator is asked
`keep this episode?` when each one ends — the file has to be written while the
arms move, so declining is a delete afterwards. Keeping is the default and a
non-interactive run keeps everything: an attempt that cannot be repeated must
not be lost to a stray keypress.

It is deliberately **not** a LeRobot dataset: that format has no slot for the
policy's own plan, which is the stream a rollout has to be diagnosed against,
and this is not the Phase 4 recorder. Chosen with Nikolas. Two mechanical notes:
it writes to **its own** `RecordingStream`, because `log_rerun_data` logs images
with `static=True` and a static entity keeps only its latest value — a file
written through it would hold three final frames rather than three streams; and
the images are JPEG-encoded with **cv2** on a worker thread (1.8 ms per tick on
the control thread, against 5.1 ms if the encode runs inline, and cv2 is 2.7x
faster than rerun's own `Image.compress`). A frame that cannot be kept up with
is dropped, counted, and reported.

## The two-policy comparison — what is built for it (2026-08-25)

`STUDY.md` is the protocol. This is what exists in the tree for it. **Nothing has
been run on the arms and nothing has been scored**; every item here is code and
its tests.

**`--profile {optimized,common}`** on `dk1 policy run` and `session`.
`dk1lab/runprofile.py` owns it, and it is a *derived config*, not an edit:
`profile.apply(settings)` returns a `DK1Config` with the crop stripped out of
every camera, so `dk1.toml` is never written and every consumer downstream — the
camera builders, the banner, the recorder's notes — reads the same thing.
`optimized` stays the default and is the identity. `common` selects the new
`[limits.study]` table (0.6 rad/s; `max_lag` deliberately unchanged, it is a
torque clamp). `POLICY_LIMITS` moved to `runprofile.py` and is re-exported from
`policy.py`.

**π0.5** — `dk1lab/pi05.py`, and `dk1 policy check` / `smoke` detect the family
off the checkpoint's own `type` rather than taking a flag. Three gaps are closed
at load, all three printed before anything runs:
32-D padded state/action narrowed to 14 by `output_features["action"].shape`,
which is what trims the chunk (verified: `(1, 50, 14)`);
`top/left/right` renamed onto `base_0_rgb`/`left_wrist_0_rgb`/`right_wrist_0_rgb`
through the pipeline's own rename step, because the model embeds the views
**positionally**; and the missing normalisation borrowed from
`andreaskoepf/dk1-merge-2026-03`'s `meta/stats.json`, whose channel names are
checked against `layout.ACTION_KEYS` before use and **match exactly**.
The gripper inversion is **off for π0.5, always**.

> **π0.5 is blocked on a licence.** Its prompt tokenizer is
> `google/paligemma-3b-pt-224`, a **gated** HF repo. Nikolas must accept it and
> `hf auth login`. `dk1lab.pi05.tokenizer_available` checks before loading 14 GB
> of weights. Do not substitute a mirror: the tokenizer decides what the prompt
> means, and the study compares two policies on one prompt.

**`dk1lab/study.py` and `--study <row>`** (2026-08-25) — the scored session.
`dk1 policy session --study A0` walks the marked scene layouts in order, **three
scenes x three attempts = 9**, prints which layout to set up before the first
attempt at each, and asks for the 0–5 rubric the moment an attempt ends —
appending it to `study/scores/<row>.csv` as it happens, never afterwards. The
module is pure bookkeeping and imports no lerobot: the score grammar
(`<0-5> [arm] [seconds] [note]`, arm required above 0, and a time only on a 5 —
where it is **derived from the episode's own length** and typed only to override
it, since the episode ends when the operator stops it, at the success), the
CSV, and `ScenePlan`, which is **built from the rows already in the file** so an
interrupted row resumes where it stopped. Three things it must keep doing:

- **the scene never enters the task string** — that string is the prompt and is
  byte-identical at every rollout, so the scene is a CSV column and nothing else;
- **the `episode` column is the only join** from a score to its frames (the
  dataset episode index, or the `.rrd` stem when that is all there is);
- **a scored session keeps every recording without asking.** A 0 is as much
  evidence as a 5, and `keep this episode?` is one keypress from deleting an
  attempt that cannot be repeated. The score prompt replaces it.

`dk1 study photo --scene N` writes `study/scene/N.jpg` off the top camera —
video device only, no motor — and `dk1 study scores <row>` reads a row back with
its per-scene grid. Under `--study`, `.rrd` recordings default to
`study/rrd/<row>/`, so a scored row's recordings never mix into `recordings/`.
**R0 is in the study again** (dropped 2026-08-25, restored 2026-08-26): it is
the tuned rig — crop, 1.0 rad/s — and it is what says whether the fine-tune was
worth more than tuning the rig. It is scored but goes to `.rrd` only, never a
dataset: its lens differs from every other row's.

**`dk1lab/dataset.py`** — a LeRobot **v3.0** recorder, `--record-dataset` on both
`run` and `session`. Alongside `record.py`, which is untouched: the `.rrd` keeps
the policy's own plan, which no dataset format has a slot for. Same four-call
instrument shape, so `dk1lab.dataset.one()` drives both at once and the operator
is asked **once**. One dataset per directory, episodes appended, an existing
directory resumed.
**A crashed session must not take its recorded episodes with it** (2026-08-26).
LeRobot v3.0 keeps one parquet writer open across episodes and writes the footer
on `finalize`, and buffers episode metadata ten at a time — so the freeze during
A0 on 2026-08-25 left seven recorded episodes that nothing can open, videos
intact, per-frame state gone. `DatasetSession` now rotates a data file per
episode (`update_chunk_settings(data_files_size_in_mb=…)`), sets the metadata
buffer to one, and closes both writers after every commit. Closing without the
rotation would be worse than nothing: the next episode reopens the same path and
truncates it — `tests/test_dataset.py` covers exactly that.

**Video is encoded on the GPU, and never behind a fork** (the fork added
2026-08-27). `--vcodec auto` resolves to `h264_nvenc` here; LeRobot's default
SVT-AV1 costs minutes per episode on the CPU. Two things about NVENC, and both
present as `avcodec_open2` failing with a bare `UNKNOWN`, i.e. as *no video*:
it refuses LeRobot's GOP of 2, so `dk1lab.dataset` raises it to 4; and **it
cannot start in a forked child**, because a CUDA context does not survive a
fork and the policy holds one. LeRobot encodes the three cameras in a
`ProcessPoolExecutor`, so `_parallel_encoding()` returns false for a GPU codec
and the streams are encoded serially, in-process. It costs about a third more
wall clock and the parallelism was never worth much — the wait is the PNG
staging, not the encode. **Judge the *resolved* codec, never `self.vcodec`:**
the default is `auto`, which is not a hardware encoder by name and would fork
anyway. A CPU codec still forks. § *Recording: the encode that could not fork*.
**Keeping an episode is slow, and that is the accepted price.** It was measured
at **4 min 25 s** on A0's first 120 s attempt, with the arms energised and the
operator waiting. It is not the codec — it is writing every frame to PNG and
reading it back, about half the wait each; NVENC itself is a second of it.

`--stream-video` skips the cache and encodes as the arms move, taking the save
to seconds — and it is **OFF by default and stays off for anything scored**.
Tried on the arms 2026-08-27 and reverted the same day at Nikolas's call: the
bench said ~5 ms of a 33 ms tick, the cell said a **984 ms worst tick, six
starved ticks and 29.2 Hz**. The queue running dry means the arms held their
last target instead of executing a new one, which is a worse attempt, and a
worse attempt costs more than a wait. **The loop is the experiment.**
**The mode is recorded per episode in `dk1_notes.jsonl`** — it is the one
recording setting that changes the control loop. Batch encoding costs the loop
essentially nothing (0.12 ms a tick, 0.7 ms worst, measured paired), because the
PNG writing is on the image-writer threads. § *Recording: four minutes to keep
one episode*.

### Recording demonstrations by hand (2026-08-27)

`dk1 teleop --record-dataset` is `STUDY.md` Phase 3's recorder — 45 teleoperated
episodes into one LeRobot v3.0 dataset, and the one thing in this study that
cannot be bought back once the day is spent. `dk1lab/demos.py`. **It has run on
the arms** — one throwaway episode on 2026-08-27, 628 frames over 20.9 s,
`common` / `[capture.policy]` / 30 Hz / uncapped, video streamed through NVENC,
read back as a v3.0 dataset and then deleted. That is the path working end to
end, not a recording session: **`study/demos` is empty and Phase 3 has not
started.**

The operator's whole vocabulary is four things, read from stdin *while the arms
are live*: **Enter** starts an episode and Enter ends it, **`again`** throws the
last one away, **`scene <n>`** labels the ones that follow, **`done`** finishes.
Five things it must keep doing:

- **The teleoperation loop never stops** — not between episodes, not while the
  operator types. That is safety, not convenience: teleop is uncapped, so a loop
  that paused would let the passive leaders sag while the followers held their
  last target, and the first tick after the pause would command the sagged pose
  at full speed. The price is that stdin is polled per tick rather than read
  (`demos.TerminalConsole`), and fd 2 is silenced between episodes for the same
  reason the session silences it — the cameras' libjpeg chatter lands in the
  middle of the line being typed.
- **What is typed while the loop is not listening is discarded** (2026-08-28).
  The loop is the only thing reading the keyboard and it stops reading while it
  writes the held episode — a second with `--stream-video`, minutes without. The
  quiet terminal reads as a hung one, the operator presses Enter again, and those
  keystrokes were then read back *after* the next episode had started: the first
  stopped it after one frame and the second ended the session. That is what made
  recording unreliable on 2026-08-27, and `study/demos` still carries two 1-frame
  episodes from it. `TerminalConsole.drain` throws away every queued line where
  the loop resumes — at the top of `loop` and after `commit_held` — and says how
  many. An episode under `demos.MIN_EPISODE_S` (0.5 s) is **dropped, not
  written**: it is a keystroke, not a demonstration.
  § *Recording demonstrations: the Enter that stopped the episode it started*.
- **An episode is committed one episode late.** `stop` leaves it in the dataset's
  buffer and the *next* start writes it, which is what makes `again` a real
  deletion: `save_episode` cannot be undone and v3.0 has no way to take an
  episode back out. What it costs is bounded — everything before the held one is
  sealed and readable, so a crash costs at most the attempt just made. Chosen
  with Nikolas over deleting a written episode off disk, which is surgery on the
  dataset both fine-tunes are built from.
- **Three defaults move with `--record-dataset`**, and all three are printed
  before anything is energised: `--profile common` (the full lens — the crop is
  applied at *training* time, so one dataset serves both the cropped and the
  uncropped row), `--capture policy` (1280x720) and `--fps 30` (the policy's
  rate, which is what gives every action chunk its time scale). The profile's
  `[limits.*]` table is deliberately **not** read: teleop stays uncapped.
- **`--stream-video` is ON here and nowhere else.** No policy is holding the GPU,
  and the wait is what would cost a day of hands. It must stay off for anything
  scored — § *Recording: four minutes to keep one episode*.

The scene is a column, never the prompt: the task string is byte-identical on
every frame, and the scene goes to `dk1_notes.jsonl` with the profile, the
capture, the rate and the codec. `DatasetEpisodeRecorder.attach_robot` is new —
the same wrapping as `attach`, without a rollout context to reach the robot
through.

### The LoRA fine-tune (2026-08-27)

`STUDY.md` Phase 4, and **nothing has been trained yet** — every item here is code
and its tests. `dk1lab/finetune.py` is the recipe and the bookkeeping (no lerobot
import); `dk1lab/recrop.py` is the crop applied to a dataset; the training itself
is LeRobot's, run through its own entry point so the recorded command line is the
one that ran.

```
dk1 dataset check study/demos                        # is it what we meant?
dk1 dataset crop  study/demos study/demos-optimized  # R1's lens, materialised
dk1 policy finetune --row R1                         # the run
dk1 policy curve study/finetune/R1-<stamp>           # which checkpoint to deploy
```

Six things it must keep doing:

- **The lens gate.** `finetune` refuses a dataset whose lens does not match the
  row's profile — R1 on uncropped frames, A1 on cropped ones. That is `STUDY.md`'s
  one invariant made mechanical, and it matters because the failure is invisible:
  a checkpoint trained for a camera that does not exist looks exactly like the one
  that was asked for. The lens comes from `dk1_lens.json` (written by `crop`) or,
  failing that, from the episodes' own `profile` in `dk1_notes.jsonl`.
- **The crop is materialised, not applied at load.** LeRobot's `image_transforms`
  hook reaches the *training* dataset and not the evaluation one, and is called
  per camera key **without the key** — so it can neither keep the held-out loss
  honest nor leave the top view alone. `recrop` copies the dataset and rewrites
  only the two wrist streams, through `dk1lab/crop.py`'s own box, cropping and
  stretching back to the same size so every frame's count and timestamp survives
  and the copy inherits the source's `meta/` unaltered. Its price is one encode
  generation, stated in the module.
- **The hold-out spans every scene.** LeRobot's `eval_split` takes the *last*
  `ceil(n x split)` episodes, and the demonstrations are recorded grouped by
  layout — so the last ten of 45 are all scene 3. `split_episodes` takes an even
  spread from each scene and hands LeRobot the episode list in an order whose
  **tail is the hold-out**, which needs no patch: `LeRobotDataset` stores
  `episodes` verbatim and the split walks it in order.
- **One repair is patched in, and it is not optional.** `lerobot_train` builds
  `preprocessor_overrides` for `normalizer_processor` and
  `postprocessor_overrides` for `unnormalizer_processor` whenever a policy loads
  from a path, and the pipeline **raises** for an override naming no saved step.
  MolmoAct2 has neither — it normalises through `molmoact2_masked_normalizer` —
  so `lerobot-train` dies before step 1. Verified on the real checkpoint.
  `finetune.patched` narrows the overrides to the steps that exist. Same class as
  the two cherry-picks: worth upstreaming, do not do it.
- **The run directory is written before training**, into
  `study/finetune/<row>-<stamp>/`: `dk1_run.json` (row, recipe, budget, split,
  checkpoint SHA-256, git SHA), a **copy** of `dk1.toml`, `command.txt`,
  `dk1_command.txt`, and `train.log`. LeRobot owns `train/` beneath it, because
  `TrainPipelineConfig.validate` refuses an `output_dir` that already exists.
- **The gripper inversion goes OFF for every fine-tuned row.** The demonstrations
  are in DK1 convention, so the weights end up speaking DK1. `curve` prints the
  `dk1 policy session` line with `--no-invert-gripper` already in it.

**Two faults found by the first probe, both of which stop the run silently.**
`DIAGNOSTICS §` *The gripper command that was never executed*:

- **A recorded gripper command outside [0, 1] kills the run at its first
  evaluation.** MolmoAct2 passes the gripper channel through its normaliser
  **unnormalised** and raises on anything outside [-1, 1]; upstream's
  `command_gripper` clips to [0, 1] *inside* the robot, so a leader trigger
  squeezed past the stop was executed as 1.0 and recorded as 1.03. 7% of
  `study/demos` was. `SafeBiDK1Follower.send_action` now returns the clipped
  value — which is what its docstring always claimed — `dk1 dataset clamp`
  repairs a recording made before that, and `dk1 dataset check` reports it.
  **Do not reach for `normalize_gripper=True`**, which the error message
  suggests: it would normalise the gripper with the YAM statistics, so a
  fine-tuned checkpoint's gripper would mean something different from A0's.
- **The run's own `train.log` was empty, so `curve` had nothing to read.** Two
  causes: `init_logging` clears every handler off the root logger (`patched`
  now wraps it, `logs.restore` puts them back), and `lerobot_train` logs with
  bare `logging.info(...)`, which arrives named **`root`** — held at WARNING by
  `logs.Interesting` until `APPLICATION_LOGGERS` was added. There is no early
  stop, so the log **is** the selection mechanism.

**What it costs, measured** (R1's configuration, `--adapt vlm+expert`, 106 M
trainable, batch 2, gradient checkpointing on): **1.13 step/s**, one evaluation
over the whole 4-episode hold-out **5 min 27 s**, ~75 s to load, **1.2 GB** per
checkpoint. So 20 000 steps is 4.9 h of training, and evaluating every 1 000
steps adds 1.8 h on top. Halving the evaluation *count* is the better economy
than `--max-eval-samples`, which takes the **first** N frames of the hold-out —
one episode's opening rather than a spread over the three scenes.

Two facts to carry with any number this produces:

- **there is no early stop.** LeRobot 0.6.1 logs a held-out loss every
  `eval_steps` and acts on none of it, so the budget runs to the end and
  `dk1 policy curve` names the checkpoint with the lowest loss. `STUDY.md` said
  *early-stop*; the amendment of 2026-08-27 is where the two were reconciled;
- **what the adapter reaches differs between the two models.** π0.5's default
  targets its action expert; **MolmoAct2's targets the VLM and leaves its 578 M
  action expert frozen**, because LeRobot's generic PEFT path freezes every base
  parameter first. `--adapt vlm+expert` extends MolmoAct2's *own* regex over the
  action expert; the flag's default is unchanged, the banner says which is in
  force, and **R1 and A1 are run with `--adapt vlm+expert`** (Nikolas, 2026-08-28
  — `STUDY.md` amendment). `dk1lab.finetune.molmoact2_target_modules` is pinned
  against LeRobot's method by a test, so it cannot drift.

**It can be stopped and picked up again**, which is what makes the budget a
decision rather than a commitment. `dk1 policy pause <run>` writes a `STOP` file;
`patched` wraps `update_last_checkpoint`, which LeRobot calls immediately after
`save_checkpoint`, so the run stops with a **complete** checkpoint and `last`
pointing at it — nothing lost. Ctrl-C stops now and gives up at most `save_freq`
steps; both are ordinary outcomes, neither exits non-zero, and both are resumable.
`dk1 policy resume <run>` continues from `checkpoints/last`, and its command line
is deliberately two arguments — `--config_path` and `--resume=true` — because
everything about the run comes back from the checkpoint's own
`train_config.json`. That is what keeps a resumed run the *same* experiment.
`train.log` is appended, so `curve` sees the whole run however many sittings it
took, and `dk1_resume.jsonl` records each one. A stop that lands on the last
checkpoint reads as **completed**, not paused.

Defaults, all recorded and all movable by flag: **8 000 steps** (a budget, not
epochs — Nikolas, 2026-08-28, with a checkpoint at 2 000, 4 000, 6 000 and
8 000), batch 2, **lr 1e-4** — not the checkpoint's 1e-5, which is a full
fine-tune's rate and barely moves a rank-32 adapter at scale 0.5 — warmup 200,
decay over the budget rather than the preset's 100 000, an evaluation **and** a
checkpoint every 2 000 steps (equal on purpose: otherwise the best loss belongs to
a checkpoint nobody saved), **4 episodes held out** (`STUDY.md` amendment of
2026-08-28: its 10 was set against 45 demonstrations and 26 were recorded, where
it would hold out 38% of them), gradient checkpointing **on**.
The recipe itself is `STUDY.md`'s and is not a flag: r=32, α=16, dropout 0.05,
`modules_to_save=[]`.

**`dk1 teleop --profile` changed meaning** (2026-08-27). It is now
`optimized`/`common`, as on `dk1 policy run`; the `[capture.*]` table it used to
name is `--capture`. One word had two meanings on the command line that records
the dataset the fine-tunes are built from.

**Every run that touches the arms writes two files** (2026-08-26), written while
the machine was still freezing hard and a terminal could not survive the reset.
The freeze is fixed; keep the files, because they are what a future one would be
diagnosed from:
`logs/<time>-<what>.log` — `dk1lab` at DEBUG, `lerobot` at INFO, tracebacks,
**fsynced per record** — and `logs/<time>-<what>.jsonl`, one sample a second of
PSU power and the +12 V rail, CPU and GPU temperature and power, memory and IO
stall, also fsynced, so the **last line is the state the machine froze in**.
`--no-log` / `--no-telemetry` opt out. `dk1 doctor watch` runs the sampler alone;
`dk1 doctor report` reads it back and says whether the file ends with a `stop`
event or with the machine. `docs/CRASH.md`.

**Opening the log lowers the *root* logger, so it owns the console too.** The
file handler filters, but `lerobot` leaves a bare `StreamHandler` on root at
import time (`import_utils` calls module-level `logging.debug`, which calls
`basicConfig`), and dropping root to DEBUG made that handler print every
library's DEBUG — thousands of `PIL.PngImagePlugin` lines over the operator's
screen. `logs.start` now applies the same policy to handlers it did not attach,
at `CONSOLE_LEVEL` (INFO). Never lower the root level without taming them.

**Prompts are written to stdout, never handed to `input()`.** `input()` puts its
prompt on **fd 2**, and `_quiet_stderr` points fd 2 at `/dev/null` while the
operator types so the cameras' libjpeg chatter cannot land in the typed line —
so passing the prompt through deleted it. That silently removed the session's
`task>` line (which names the live scene and attempt), `score>`, and
`keep this episode?`. `_ask` writes the prompt itself; `_confirm` replaces
`typer.confirm` for the same reason. § *The session console*.

**A failed write is now loud.** On 2026-08-26 one episode's encode raised, the
failure was a single log line, and the session scored five more attempts into a
dataset that stayed empty. A failure now prints in red after that attempt, the
traceback goes to the log, and the episode buffer is cleared so one bad encode
does not poison every episode after it. That episode was the fork bug above,
found for certain when it took A0's first attempt on 2026-08-27; the loudness is
what caught it the second time.

**An episode is not written until it is kept.** `stop()` leaves it in the buffer;
`keep()` calls `save_episode`, `discard()` clears it. The first version saved in
`stop()` and had a `discard` that reported success and changed nothing —
`save_episode` cannot be undone. `DatasetSession.close()` keeps anything pending.

**The simulator** — `dk1lab/scene.py` generates the bimanual MJCF from this
repo's own `urdf/` (upstream's file, read never written), and `dk1lab/sim.py`'s
`SimRobot` is a LeRobot robot registered `dk1_sim`. `dk1 sim scene|view|sweep`,
and `--sim` on `run` / `session`. Measured: **916 Hz free-run with all three
cameras rendered**, exactly 30.0 Hz under `--realtime`.

**Both policies drove it, 2026-08-25** — MolmoAct2 at 29.8 Hz over 40 chunks and
π0.5 at 29.9 Hz over 17, no starved ticks either way, home sweep reaching within
0.03 rad. That closes `STUDY.md`'s Phase 1 and is the whole of what it claims:
the pipeline runs. Two things had to be fixed for π0.5 to get there, both of
which would otherwise have failed silently:

- `dk1 policy run` built a **MolmoAct2** config whatever the checkpoint was, so
  π0.5 would have run with its MolmoAct2-only fields ignored and **no
  normalisation at all**. `policy.family()` now reads the type off the
  checkpoint's own `config.json`, `build_context` patches the borrowed statistics
  into the loaded pipeline (same argument as `apply_gripper_inversion`: the
  pipelines are rebuilt from JSON on every load, so the config is not a hook),
  the rename map is set, and the gripper inversion is **refused** for π0.5.
- The chunk FIFO's relative-actions guard matched by class name and ignored
  `enabled`. π0.5 ships a `RelativeActionsProcessorStep` with `enabled: false` so
  a fine-tune can switch it on; disabled it passes actions through untouched.
  LeRobot's own guard tests the flag. Ours now does too.

> **The scene itself is poor, and is not fixed.** The arms sit on 0.30 m
> pedestals and **cannot reach the table** — the pedestal exists because at the
> zero pose the DK1's elbow folds behind and below its own base, so a flat-
> mounted arm starts with contact points through the table and its base yaw
> pinned; solving that created this. The bowl is a **round base with four square
> walls**. The arm spacing and camera poses are the *training rig's*, not this
> cell's. So a sim rollout says the pipeline runs and nothing else — do not quote
> it as a task result. `STUDY.md` § *The simulator* carries the same list.

## The defaults that were tuned, and why they are what they are

Every one of these was moved from something else, on evidence. Change them only
against the section of `docs/DIAGNOSTICS.md` named in the last column.

| setting | value | why |
| --- | --- | --- |
| `[limits.policy] max_joint_rate` | **1.0 rad/s** | 0.3 truncated exactly the policy's reach and transit moves — 56° of lag at worst. Must **not** be turned off for a rollout the way teleop's is. § *What the caps were doing* |
| `[limits.policy] max_lag` | **0.4 rad** | it is a *torque* clamp, not a position one: 0.1 rad held j6 to a tenth of its authority and could deadlock the setpoint. § *What the caps were doing* |
| `[capture.policy]` | **1280×720** | the model input is 378 rows; at 640×360 the wrist crop kept 256 and upsampled. Costs under 1 ms per tick, measured paired. § *The capture resolution* |
| wrist crop | inset 6, shift_y −40 | 105° lens against an 87° training FOV. Read the box off `dk1 config show`, never from prose — it has gone stale twice. § *The camera crop* |
| inference engine | `--sync` + `--fifo` + `--async-fifo` | one model call per chunk, on a worker thread. RTC discarded 27 of every 30 rows at the measured in-situ latency. § *The stall*, § *The freeze*, § *The fix* |
| `--replan-at` / `--blend` | 15 / 4 rows | left 8 rows (270 ms) in hand at every splice on the arms. § *The fix* |
| `--duration` | 180 s | the policy barely acts for ~30 s and a successful sim episode averaged 54 s. A 30 s rollout cannot succeed |
| `DEFAULT_EXECUTION_HORIZON` | 20 | RTC only, and RTC is not the default. A horizon at or below the delay collapses the blend to a step function — that was the judder. § *The first rollout* |
| `--invert-gripper` | **on** | the checkpoint is 1=open and this cell is 0=open; confirmed on the arms 2026-08-21. Off on `serve` only, where the wire protocol is already YAM |
| `--home` | **on** for `run` and `session` | the sweep is tested on the arms, and ending a run wherever the policy stopped is what wears them |
| home sweep rate | 0.3 rad/s, eased | the cap is an upper bound, not a speed to aim for; reading it as one tripled the sweep speed as a side effect of raising the cap. § *The home sweep speed* |

## Where the detail lives

`docs/DIAGNOSTICS.md` is the record: every measurement, every fault chased to its
cause, and the hypotheses that turned out wrong. Read the section before
re-measuring anything.

| if you are about to | read |
| --- | --- |
| touch the timing, the engine, or the control loop | § *The stall*, § *The 27.7 Hz loop*, § *The freeze*, § *The fix* |
| touch the crop, the capture resolution or the camera geometry | § *The camera crop*, § *The capture resolution* |
| touch the speed caps or the home sweep | § *What the caps were doing*, § *The home sweep speed* |
| touch the trace, `--display` or `--display-policy-input` | § *The instruments* |
| re-run the simulator, or quote the sim result | § *The sim run* |
| run, change or quote the two-policy comparison | `STUDY.md` for the protocol and `study/results.md` for what it scored — not `docs/DIAGNOSTICS.md` |
| push or pull the datasets, the `.rrd` or a fine-tune checkpoint | `docs/HUB.md` — none of it is in git, and it says what is deliberately not pushed |
| wonder what a past rollout on the arms actually showed | § *The rollouts on the arms* |
| chase a machine freeze, if one ever returns | `docs/CRASH.md` § *How it was found* — closed 2026-08-27, a BIOS update. Do not re-run the investigation from the top |
| touch the fine-tune, the crop applied at training time, or the hold-out | `STUDY.md` § *Fine-tuning* and its *Amendments*, then `dk1lab/finetune.py` and `dk1lab/recrop.py` — both carry their reasoning in their module docstrings |
| touch the dataset recorder, the codec, or anything about saving an episode | § *Recording: the crash that ate seven episodes*, § *Recording: the episode that took minutes to save*, § *Recording: the encode that could not fork*, § *Recording: four minutes to keep one episode*, § *Recording demonstrations: the Enter that stopped the episode it started* |
| touch the session log, the console output, or any prompt the operator types at | § *The session console: a silenced prompt and a shouted decoder* |
| benchmark anything | § *The 27.7 Hz loop* — a flat-out loop and a paced one disagree about the same function, and separate processes drift by more than the effect you are chasing |

## Environment

RTX 5090 (32 GB), Python 3.12, uv, LeRobot **0.6.1**, fish shell. The CPU
governor stays on `powersave` by request; it was ruled out as a cause.
Old project for reference (read-only): `~/Documents/RobotLearning/trlc-dk1`,
branch `wip/molmoact2` — real knowledge in `MOLMOACT2.md` / `MOLMOACT2_EVAL.md`,
but neither of its two contradictory verdicts on zero-shot is a result.
The sim clone is at `~/Documents/RobotLearning/molmoact2` (scratch, not tracked).

`uv run pytest -q` · `uv run dk1 --help`
