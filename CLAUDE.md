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
  cameras.py            builds lerobot OpenCVCameraConfig from config
  robot.py              SafeBiDK1Follower — the rate-limited follower
  teleop.py             the one teleoperation implementation
  cli/                  Typer app; `dk1` entry point
dk1.toml                THE device config. Tracked. Single source of truth.
tests/                  212 tests, none need hardware
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

## Safety (non-negotiable)

- **Connecting is not passive.** `connect()` energises every motor and self-zeroes
  both grippers by driving them closed until they stall. Every command that
  connects must say so in `--help` and warn again on stderr before acting.
  Helpers: `dk1lab/cli/safety.py`.
- **Stopping never moves the arms.** `return_to_initial_position` defaults to
  `true` in LeRobot's rollout — always set it `false`. Return-to-home is opt-in
  only.
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
- **The gripper channel is inverted.** The YAM server contract is **1 = open,
  0 = closed**; the DK1 is **0 = open, 1 = closed** (`DK1Robot.command_gripper`,
  `DK1Leader.get_action`). Sources: `sim_eval/inference/common.py` on the
  colleague's branch (states it in three places), plus the checkpoint's gripper
  stats sitting at mean 0.64 / median 0.73, i.e. predominantly open. Use
  `layout.yam_joint_signs()` / `yam_joint_offsets()` — **on by default** for
  zero-shot, not an opt-in flag.
- **Image order is already pinned in the checkpoint.** Its `policy_preprocessor.json`
  carries `["...top","...left","...right"]`. The inherited claim that the
  processor sorts alphabetically at deployment is **wrong**; the hazard is real
  only when *training* rebuilds the processor from a new dataset's features. Pin
  `--policy.image_keys` for training anyway.
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

Nothing about MolmoAct2 on the real DK1 has been verified. No dataset recorded,
no fine-tune completed, no policy has ever driven these arms.

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

**The arm sides are confirmed** — Nikolas verified the four ports in `dk1.toml`
are correct as they stand, so `dk1 find arms` was not needed. Nothing about the
ports is open any more.

## Phases

| | | |
| --- | --- | --- |
| **0** | Foundation — package, config, CLI, limiter, tests | **done**, branch `phase0-foundation` |
| **1** | Device discovery on the hardware | **done** |
| **2** | Teleoperation | **done** — run on the arms, limits tuned |
| **3** | Zero-shot MolmoAct2 evaluation — the first real goal | |
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

**Phase 3** — escalating risk: (1) smoke test, GPU only, no robot; (2) reuse the
bf16 checkpoint; (3) dry run — full deployment path with actions **printed, never
sent**; (4) slow rate-limited rollout with a human on the e-stop. Gripper
inversion on, image keys pinned, `inference_action_mode=continuous`, bf16, RTC
(inference measured at ~172 ms ≈ 5 control periods at 30 Hz — unverified number
inherited from the old repo).

**Phase 4** — record → LoRA from the same checkpoint → deploy → scored, labelled
eval attempts.

## Environment

RTX 5090 (32 GB), Python 3.12, uv, LeRobot **0.6.1**, fish shell.
Old project for reference (read-only): `~/Documents/RobotLearning/trlc-dk1`,
branch `wip/molmoact2` — real knowledge in `MOLMOACT2.md` / `MOLMOACT2_EVAL.md`,
but its `molmoact2/` scripts beyond `smoke_test.py` and `convert_bf16.py` were
never run. Its two docs contradict each other on whether zero-shot will work;
neither is a result.

`uv run pytest -q` · `uv run dk1 --help`
