# CLAUDE.md — architecture and project state

Read this first. It is what is true now, and why the code is shaped the way it
is. Three companions, none of which repeats this one: `GUIDE.md` is the
operator-facing version, **`STUDY.md` is the protocol** for the two-policy
comparison, and **`DIAGNOSTICS.md` is the record** — every
measurement, every fault chased to its cause, and the hypotheses that turned out
wrong. Sections below point into it as `DIAGNOSTICS § name`. Read the section
before re-measuring or undoing anything it covers.

## What this is

A fork of [robot-learning-co/trlc-dk1](https://github.com/robot-learning-co/trlc-dk1)
set up to operate a bimanual TRLC-DK1 cell (2 leader arms, 2 follower arms, 3 USB
cameras) with LeRobot, and to evaluate and fine-tune the **MolmoAct2** VLA policy
on it. Origin is `NikolasKelava/trlc-dk1`; upstream is the hardware repo and we
want to keep pulling its updates.

> **OPEN FAULT, read before running anything on the cell: the machine freezes.**
> Two scored sessions (2026-08-25, 2026-08-26) and at least one teleoperation run
> have taken the whole machine down — hard reset, nothing in the journal.
> **Unresolved.** `CRASH.md` is the account and the brief. Do not record the
> ~100 teleoperation demonstrations until it is understood.

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
  study.py              the score sheet: scenes, attempts, the CSV  (no lerobot import)
  logs.py               a session log file, fsynced per record       (no lerobot import)
  telemetry.py          PSU, CPU and GPU once a second, fsynced      (no lerobot import)
  scene.py              the bimanual MuJoCo scene, generated from urdf/  (no lerobot)
  sim.py                SimRobot — the MuJoCo cell behind the real robot interface
  cli/                  Typer app; `dk1` entry point
dk1.toml                THE device config. Tracked. Single source of truth.
tests/                  the suite; none of it needs hardware
GUIDE.md                operator docs
DIAGNOSTICS.md          the record: measurements, faults, discarded hypotheses
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

**Nothing on the real arms has been scored.** No dataset recorded, no fine-tune,
no success rate. That is the one sentence to keep straight.

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
| The home sweep works | run on the arms, including the eased profile. Now the default at the end of a run |

What is not:

- a **score** — labelled attempts with a success count, which is the input to
  the Phase 4 decision;
- why **the right arm was reported not to pick anything up**;
- anything about the crop retune (inset 6, view lifted 40) or the 1280×720
  capture beyond "it ran".

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
| **3** | Zero-shot MolmoAct2 evaluation | **six debugging rollouts plus a first recorded session of eight episodes.** Every timing and motion fault this fork could cause is closed; what is left is the policy's own output. **Still not scored** |
| **3s** | The same policy in ManiSkill, via the colleague's `sim_eval` | **done: 3/3** |
| **4** | Record + LoRA fine-tune | gated on reviewing Phase 3 together |
| **5** | The two-policy comparison — MolmoAct2 vs π0.5, one task, N=15 per row (5 scene configurations x 3 attempts) | protocol in `STUDY.md`, which carries its own phase numbering. **Its Phases 0 and 1 are done**; Phase 2 is the arms — see below |

**Phase 1** built `dk1 find cameras`, `dk1 find arms --inspect` (read-only USB
identity) and `dk1 config check --formats`. That last one matters: OpenCV
accepts an unavailable capture size and silently substitutes the nearest one it
has, which would hand the policy a different aspect ratio than training used.

**Phase 2.** `dk1 teleop` is the single entry point, `dk1lab/teleop.py` the
single implementation, and the loop is LeRobot's `teleop_loop` imported rather
than reimplemented — because recording and rollout run that same loop, and a
bespoke one here could work while the one every later phase depends on does not.
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
| `dk1 study photo` | one still of a scene layout | a video device, nothing else |
| `dk1 study scores` | reads a scored row's CSV back | none — a text file |
| `dk1 doctor watch` | samples PSU, CPU and GPU once a second | none — sysfs and nvidia-smi |
| `dk1 doctor report` | reads the last telemetry file back | none — a text file |

`dk1lab/policy.py` is the single implementation; `dk1lab/checkpoint.py` is the
JSON-only reader behind `check`; `dk1lab/session.py` holds the loaded policy
across rollouts. Settings are decided in code, not left to a command line —
image keys pinned, `inference_action_mode = continuous`, bf16 on cuda,
`return_to_initial_position` forced false always, including under `--home`.

**What remains for this phase is a score.** The next run is not a debugging run:
it is labelled attempts with a success count. One thing still unexplained to
watch while scoring: **the right arm was reported not to pick anything up**.

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

**Video is encoded on the GPU.** `--vcodec auto` resolves to `h264_nvenc` here;
LeRobot's default SVT-AV1 costs minutes per episode on the CPU. NVENC refuses
LeRobot's GOP of 2 — `avcodec_open2` fails — so `dk1lab.dataset` raises it to 4
for hardware encoders. `--stream-video` encodes during the rollout instead of
from a PNG cache (keeping an episode: ~1 min -> seconds) at ~3 ms a tick plus a
one-off stall; **off by default**.

**Every run that touches the arms writes two files** (2026-08-26), because the
machine has frozen hard three times and a terminal does not survive a reset:
`logs/<time>-<what>.log` — `dk1lab` at DEBUG, `lerobot` at INFO, tracebacks,
**fsynced per record** — and `logs/<time>-<what>.jsonl`, one sample a second of
PSU power and the +12 V rail, CPU and GPU temperature and power, memory and IO
stall, also fsynced, so the **last line is the state the machine froze in**.
`--no-log` / `--no-telemetry` opt out. `dk1 doctor watch` runs the sampler alone;
`dk1 doctor report` reads it back and says whether the file ends with a `stop`
event or with the machine. `CRASH.md`.

**A failed write is now loud.** On 2026-08-26 one episode's encode raised, the
failure was a single log line, and the session scored five more attempts into a
dataset that stayed empty. A failure now prints in red after that attempt, the
traceback goes to the log, and the episode buffer is cleared so one bad encode
does not poison every episode after it.

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
against the section of `DIAGNOSTICS.md` named in the last column.

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

`DIAGNOSTICS.md` is the record: every measurement, every fault chased to its
cause, and the hypotheses that turned out wrong. Read the section before
re-measuring anything.

| if you are about to | read |
| --- | --- |
| touch the timing, the engine, or the control loop | § *The stall*, § *The 27.7 Hz loop*, § *The freeze*, § *The fix* |
| touch the crop, the capture resolution or the camera geometry | § *The camera crop*, § *The capture resolution* |
| touch the speed caps or the home sweep | § *What the caps were doing*, § *The home sweep speed* |
| touch the trace, `--display` or `--display-policy-input` | § *The instruments* |
| re-run the simulator, or quote the sim result | § *The sim run* |
| run, change or quote the two-policy comparison | `STUDY.md` — the protocol, not `DIAGNOSTICS.md` |
| wonder what a past rollout on the arms actually showed | § *The rollouts on the arms* |
| chase the machine freezing, or add to that investigation | `CRASH.md` — the open fault, and what has already been eliminated |
| touch the dataset recorder, the codec, or anything about saving an episode | § *Recording: the crash that ate seven episodes*, § *Recording: the episode that took minutes to save* |
| benchmark anything | § *The 27.7 Hz loop* — a flat-out loop and a paced one disagree about the same function, and separate processes drift by more than the effect you are chasing |

## Environment

RTX 5090 (32 GB), Python 3.12, uv, LeRobot **0.6.1**, fish shell. The CPU
governor stays on `powersave` by request; it was ruled out as a cause.
Old project for reference (read-only): `~/Documents/RobotLearning/trlc-dk1`,
branch `wip/molmoact2` — real knowledge in `MOLMOACT2.md` / `MOLMOACT2_EVAL.md`,
but neither of its two contradictory verdicts on zero-shot is a result.
The sim clone is at `~/Documents/RobotLearning/molmoact2` (scratch, not tracked).

`uv run pytest -q` · `uv run dk1 --help`
