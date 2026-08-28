# STUDY.md — the comparison, and how it will be run

The protocol for comparing two policies on this cell, decided *before* the
measurements so the measurements can be trusted.

`CLAUDE.md` is what is true now about the code and the cell. `docs/DIAGNOSTICS.md` is
the record of faults chased to their cause. This file is the experiment.
Results land in `study/results.md`. Nothing here changes once Phase 2 starts,
except via the *Amendments* log at the bottom.

---

## The question

> On one task, on this cell: does fine-tuning on ~100 of our own demonstrations
> close the gap between two general-purpose VLA policies — and does the ranking
> between them survive it?

Not "which policy is best". Two models on one task cannot answer that.

What it can establish: a **zero-shot number on real hardware** (this cell has
never scored anything), **what ~100 task-specific demonstrations buy** per
policy under an identical recipe, and **what the level playing field costs**
against the tuned MolmoAct2 configuration this cell runs today.

---

## The two policies

| | checkpoint | size | what it is |
| --- | --- | --- | --- |
| **MolmoAct2** | `lerobot/MolmoAct2-BimanualYAM-LeRobot` (local bf16 copy) | 5.44 B | Reasoning VLA. Molmo VLM emits a visual plan; a 578 M action expert emits 30-step chunks at 30 Hz, 14-D absolute joint pose. Already runs here |
| **π0.5** | `lerobot/pi05_base` | ~3.3 B | Flow-matching VLA on PaliGemma. Discrete language reasoning, then continuous action decoding. Broadest open cross-embodiment pretraining |

π0.7 was the original name and **has no open weights** — `openpi` ships π0,
π0-FAST and π0.5 only. Cosmos3-Nano is out on four counts: 16 B, no LeRobot
policy class, its own serving stack, and a Cartesian action space this cell's
14-D joint pipeline cannot drive. Both are decisions, not oversights.

MolmoAct2's split, read off its own safetensors header — these numbers decide
what "more than LoRA" can mean:

| module | params | share |
| --- | ---: | ---: |
| `model.model.transformer` (the Molmo LM) | 4.03 B | 74.0% |
| `model.model.action_expert` | 578 M | 10.6% |
| `model.model.vision_backbone` | 439 M | 8.1% |
| `model.lm_head` | 396 M | 7.3% |

---

## The one task

```
put the dice in the bowl
```

**That string is the prompt.** Byte-identical at every rollout, and the `task`
recorded in every demonstration.

Chosen because it resets in two seconds and its success is not a judgement call.

### Three scene configurations, three attempts each

One task string, but not one layout. **The dice and the bowl take three marked
positions, numbered 1 to 3, and each one is attempted three times** — nine
scored attempts per row. A policy that can only reach one corner of the table
scores the same as one that can do the task if all nine attempts share a layout;
three layouts is what tells them apart, and three attempts each is what keeps a
single lucky grasp from carrying a layout.

Two words that are easy to confuse, so they are fixed here:

- a **configuration** (or **row**) is R0 / A0 / A1 / B0 / B1 — a policy under a
  profile, one line of the results table;
- a **scene configuration** (**scene 1, 2, 3**) is where the dice and the bowl
  sit. The same three scenes are used by every row.

**Scene reset.** The positions are **marked on the desk in pen**, so a scene is
reproduced by putting the dice and the bowl back on their marks. Each scene is
photographed to `study/scene/1.jpg` … `3.jpg` with `dk1 study photo --scene N`,
which grabs a still from the `top` camera — cameras only, no motor is touched.
The rows are run on different days, so the marks and the photographs are what
hold them comparable.

**The scenes are run in order, and grouped.** A session does scene 1 three
times, then scene 2 three times, then scene 3. The session prompt shows which
scene is live and which attempt of the three is next, and asks the operator to
set the scene up before the first attempt of each one — the scoring is grouped by
scene in the CSV, and interleaving the layouts is how that grouping stops being
true.

---

## Two profiles: one frozen, one level

### `optimized` — the existing MolmoAct2 configuration, frozen

The wrist crop, its offset, the 1280×720 capture, `[limits.policy]`. **Default,
and nothing about it changes.** `dk1.toml` is not edited, `dk1lab/crop.py` is not
touched, and this configuration is **never fine-tuned**. It is one reference row,
so the price of the level playing field is measured rather than assumed.

> Read the crop box off `dk1 config show`, never from prose. It has gone stale
> three times, which is why the numbers are not restated here.

### `common` — the level playing field

Both policies get **identical observations**: no wrist crop, no offset, the full
105° frame from all three cameras, rotation 180 (a fact about the mounts).

A **`--profile common` flag** on `dk1 policy run` / `session`: builds cameras
without `target_hfov`, selects a new `[limits.study]` table. Existing sections in
`dk1.toml` are untouched and the default stays `optimized`, so nothing that
works today moves.

| | `[limits.policy]` | `[limits.study]` |
| --- | --- | --- |
| `max_joint_rate` | 1.0 rad/s | **0.6 rad/s** |
| `max_lag` | 0.4 rad | **0.4 rad — unchanged** |
| `max_gripper_rate` / `max_dt` | 1.0 / 0.1 s | unchanged |

Only the rate drops. `max_lag` is a *torque* clamp, not a position one —
tightening it once held j6 to a tenth of its authority and could deadlock the
setpoint. Lowering it would stall the arms, not make them safer.
`DIAGNOSTICS §` *What the caps were doing*.

0.6 sits between two measured points: at 0.3 the worst joint ended 0.98 rad
behind the policy's intent (26% of ticks lagging); at 1.0, 0.40 rad and 3.3%.
**Expect the cap to cost both policies something, and record what it costs.**
Both wear the same handicap, so the comparison survives; the absolute numbers
carry the caveat.

### The gripper convention — the trap that would silently ruin a fine-tune

MolmoAct2's checkpoint speaks YAM (**1 = open**); this cell is **0 = open**,
which is why `--invert-gripper` is on by default. Our demonstrations will be in
DK1 convention, because that is what the robot reports. So a fine-tuned
MolmoAct2 **outputs DK1 convention**, and leaving the inversion on flips every
grasp.

| condition | `--invert-gripper` |
| --- | --- |
| MolmoAct2 zero-shot | **on** — the weights genuinely speak YAM |
| MolmoAct2 fine-tuned | **off** — the weights now speak DK1 |
| π0.5, always | **off** |

Both models train on the same dataset bytes, which is what keeps the tuning
comparison clean. Confirm it behaviourally on the *first* fine-tuned episode
before spending nine attempts.

---

## The five configurations

**N = 9 scored attempts each** — three scene configurations, three attempts
apiece — one task string, 45 attempts across the five rows.

**R0 is back in** (2026-08-26). It is the tuned configuration this cell already
runs — the wrist crop, the 1.0 rad/s cap — and it answers a question the other
four cannot: *how much of what fine-tuning buys could have been had by tuning
the rig instead?* A1 beating A0 says the LoRA worked. A1 beating **R0** says the
LoRA was worth more than a crop and a speed cap, and that is the comparison
worth having, because the crop is free and the fine-tune is not.

| # | configuration | profile | tuning | phase |
| --- | --- | --- | --- | --- |
| R0 | MolmoAct2 optimized | `optimized` | none — the tuned rig, frozen | 9 |
| A0 | MolmoAct2 zero-shot | `common` | none | 2 |
| A1 | MolmoAct2 + LoRA | `common` | LoRA | 5 |
| B0 | π0.5 zero-shot | `common` | none | 6 |
| B1 | π0.5 + LoRA | `common` | LoRA | 8 |

**MolmoAct2 goes all the way first, then π0.5.** B0 was originally Phase 2's
third row; it now runs at Phase 6, after MolmoAct2 has been fine-tuned and
scored. The
cell is only worth setting up for one policy at a time, and the ~100
demonstrations are what both fine-tunes need, so nothing is learned by pushing
π0.5 through the arms before that dataset exists. Nothing about the *comparison*
changes: every configuration still runs the same task from the same three scenes
under the same profile, and B0 is still zero-shot weights.

`--duration` drops from the 180 s default to **120 s**, and that number is fixed
for every scored row: 180 was chosen for a multi-object sim task, and 60 — the
figure this protocol carried until A0 was about to start — was an estimate made
before anyone had watched this policy attempt this task on these arms. Nikolas
set it to 120 at the cell. What matters more than the value is that it is the
same for all four rows; an attempt that runs out of clock is a 3 that might have
been a 5.

The action-expert-only fine-tune is an **optional extra** (see *Fine-tuning*),
not one of the five.

---

## Scoring

Highest step reached, 0–5. Success rate is the fraction reaching 5; the partial
credit is what stops a table of zeros from being uninformative.

| step | reached when |
| --- | --- |
| 0 | no purposeful motion toward the dice |
| 1 | **approach** — an end-effector within ~5 cm of the dice |
| 2 | **contact** — the gripper touches it |
| 3 | **grasp** — lifted clear of the table, held ≥ 1 s |
| 4 | **transport** — carried over the bowl |
| 5 | **success** — released into the bowl, stays there |

Also per attempt: the scene configuration, the attempt number within it, time to
success, **which arm was used**, and a one-line failure note. That arm column is
not bookkeeping — `CLAUDE.md` carries an unexplained
observation that *the right arm was reported not to pick anything up*. **A0
killed it**: the right arm scored 3 in all four attempts it was used for and the
left scored 2, 2, 2, 1, 1 in five, so if either arm is the weak one it is the
left. Nine attempts is not a study of arms; it is enough to retire that claim. If both policies favour the left arm, that is a finding about the cell.

Written to `study/scores/<config>.csv` **during** the session, not reconstructed
afterwards. One row per attempt, nine rows per file, in the order they were
run:

```csv
scene,attempt,episode,score,seconds,arm,note
1,1,0,3,,left,grasped the dice, dropped it short of the bowl
1,2,1,5,21.4,left,
1,3,2,0,,none,no purposeful motion
2,1,3,2,,right,nudged the dice out of reach
```

| column | what it holds |
| --- | --- |
| `scene` | the scene configuration, 1–3 |
| `attempt` | which of the three attempts at that scene, 1–3 |
| `episode` | the dataset episode index this attempt was written as, or the `.rrd` stem when that is all there is, or empty when the attempt recorded nothing. **This is the only join between a score and its frames** — the task string is byte-identical everywhere and the scene is not in it |
| `score` | the rubric, 0–5 |
| `seconds` | time to success, only when `score` is 5. **Derived from the episode** — it ends when the operator stops it, which is at the success |
| `arm` | `left`, `right`, `both`, or `none` |
| `note` | one line, why it stopped where it did. Empty on a 5 |

The scene never goes into the **task string**: that string is the prompt and is
byte-identical at every rollout. It is a column here and nowhere else.

Report the success rate per row **and** the 3×3 grid per row. Three scenes of
three is a coarse instrument for per-scene skill, but "it only ever works at
scene 2" is a conclusion that nine attempts at one layout cannot reach at all.

---

## What gets recorded, and where

**LeRobot dataset v3.0** for the demonstrations and every scored rollout;
viewable with `lerobot-dataset-viz` and the Hub's dataset viewer.

**R0 is scored but not recorded to a dataset.** It runs under `optimized`, so
its frames carry the wrist crop and the offset — a different lens from every
other row's, and mixing them into one dataset would give a fine-tune two
geometries and no way to tell them apart. It records `.rrd` instead, into
`study/rrd/R0/`, and every one is kept: that is the visual record of the tuned
rig, and it carries the policy's own plan, which no dataset format has a slot
for. Every other row goes to its own LeRobot dataset under `--profile common`.

**Video is encoded with the GPU** (`--vcodec auto`, NVENC here) rather than
LeRobot's SVT-AV1 on the CPU, which spent minutes per episode with the operator
waiting and the arms energised. Most of what is left is LeRobot writing every
frame to PNG and reading it back to encode; `--stream-video` encodes as the arms
move instead, which takes keeping an episode from about a minute to a few
seconds — at about 3 ms a tick on the control loop, and one longer stall when
the encoder starts. **Off by default: the loop is the experiment.**

`dk1lab/record.py` and its `.rrd` output are unchanged and stay available behind
`--record`, for an attempt worth diagnosing against the policy's own plan — the
one stream no dataset format has a slot for. Under `--study <row>` they are
written to **`study/rrd/<row>/`**, a directory of their own, so a scored row's
recordings never mix into `recordings/`, which holds six unscored tasks from
before the study.

```
study/
  scene/1.jpg .. 3.jpg   the three scene configurations         [tracked]
  demos/                 ~100 teleop episodes, LeRobot v3.0     [not in git]
  rollouts/<config>/     one LeRobot dataset per scored run     [not in git]
  rrd/<config>/          .rrd kept when a row is diagnosed      [not in git]
  scores/<config>.csv    the rubric, 9 rows: scene x attempt    [tracked]
  results.md             the tables and the interpretation      [tracked]
recordings/              the eight legacy .rrd — UNCHANGED, and prior evidence
                         for the optimized configuration, not a scored row
                         (they span six different tasks and were never scored)
```

The datasets do not go in git — the eight `.rrd` already exceed GitHub's 100 MB
hard limit. They are LeRobot datasets, so they belong on the **Hugging Face
Hub**. Only `scores/`, `scene/` and `results.md` are tracked.

**A dataset holds all nine attempts of a row, in scene order** — one dataset per
configuration, not one per scene. Episodes are appended, so episode index 0–2 are
scene 1, 3–5 are scene 2, 6–8 are scene 3; the `episode` column of the CSV is
what records that rather than the arithmetic, because a failed attempt that is
re-run shifts it.

**One invariant, and it is the one that will be violated if anything is:**

> The demonstrations must be recorded through **exactly the observation path the
> fine-tuned policy is rolled out under** — `--profile common`, no crop, no
> offset, same capture size, same camera keys.

Record demos through the optimized crop and deploy without it, and the model is
tuned for a lens it never sees again. Same argument that put the crop in the
camera rather than in a policy processor.

Camera keys stay `top` / `left` / `right`, matching MolmoAct2's pinned
preprocessor.

---

## Fine-tuning

Same dataset bytes, same step budget, same schedule, both models.

**LoRA** via LeRobot's PEFT path (`--peft.method_type lora`). Both models define
default targets and they are the same recipe in shape: the action expert's
`q`/`v` projections plus the state and action IO projections. Fixed for both:
**r = 32, α = 16, dropout = 0.05**, `modules_to_save = []`.

**Optional extra — action expert only, fully trained.** VLM frozen, action head
fully trained; a native flag in both (`train_action_expert_only` on MolmoAct2,
578 M trainable, requires `action_mode="continuous"` which is what we run;
`train_expert_only` on π0.5, ~300 M). Run only if time allows, as two extra rows.

**There is no full fine-tune.** MolmoAct2 at 5.44 B needs ~87 GB with AdamW, or
~33 GB before activations even with an 8-bit optimizer. The 5090 has 32 GB.
π0.5 at 3.3 B is borderline, and a condition that runs natively for one model and
via CPU offload for the other is two experiments, not a comparison.

**Fixed for every run:** a **step budget, not epochs** — fixed once in Phase 4
and reused verbatim in Phase 7. Hold out 10 of the ~100 episodes for validation and
early-stop on it. Log the checkpoint hash, the `dk1.toml` in force, the command
line and the git SHA into each run directory.

### Running a training run

`dk1 policy finetune` is the one command. It drives LeRobot's own trainer through
its PEFT path and adds only what this study needs around it: the row, the lens
gate, the stratified hold-out, and the run directory.

```
# R1 — MolmoAct2 on the tuned rig. Needs the CROPPED copy of the demonstrations.
dk1 dataset check study/demos
dk1 dataset crop  study/demos study/demos-optimized
dk1 policy finetune --row R1

# A1 — the level playing field, on the demonstrations exactly as recorded.
dk1 policy finetune --row A1 --dataset-dir study/demos

# afterwards: which checkpoint to deploy
dk1 policy curve study/finetune/R1-<stamp>
```

`--dry-run` writes the run directory and prints the `lerobot-train` line without
training. `--steps`, `--batch-size`, `--lr`, `--eval-every` and `--holdout` move
the budget; whatever they are set to is recorded, and the same numbers are reused
in Phase 7.

**A run can be stopped and picked up again**, which is what makes an 8 000-step
budget a decision rather than a commitment:

```
dk1 policy pause  study/finetune/R1-<stamp>   # stops at the next checkpoint
dk1 policy resume study/finetune/R1-<stamp>   # continues from checkpoints/last
```

`pause` writes a STOP file that the loop reads immediately after each checkpoint
is written, so nothing is lost; Ctrl-C stops now and gives up at most `save_freq`
steps. Either way the run is resumable, and a resumed run is the **same**
experiment — the dataset, the split, the recipe, the budget and the schedule come
back from the checkpoint's `train_config.json`, not from what is typed the second
time. `--steps` on a resume raises the budget, and says so when that takes the run
past the horizon its learning rate was scheduled to decay over.

**The command refuses to start if the dataset's lens does not match the row's
profile** — R1 on uncropped frames, or A1 on cropped ones. That is the invariant
in *What gets recorded, and where* made mechanical: a checkpoint trained for a
camera that does not exist looks exactly like the one that was asked for.

Three things about the recipe are worth knowing before reading a result:

- **the normalisation is the checkpoint's, not the dataset's.** MolmoAct2
  normalises through its own masked normalizer keyed by `norm_tag`, so the
  fine-tune trains and deploys under the same statistics A0 ran under. Our
  demonstrations are in DK1 convention and the YAM statistics are the units they
  are expressed in; the adapter learns the rest. Changing that would make A1 and
  A0 differ by more than the LoRA;
- **what the adapter reaches is not the same for the two models** — see the
  amendment of 2026-08-27 below, and `--adapt`;
- **there is no early stop** — the budget runs to the end and `dk1 policy curve`
  names the checkpoint with the lowest held-out loss. Also an amendment below.

---

## The simulator

**MuJoCo with this repo's own DK1 URDF.** `mujoco` is already a dependency and
`trlc_dk1_control/gravity_comp.py` already loads `urdf/follower/` into it — with
meshes stripped, dynamics only, so a usable scene is new work.

Chosen because the sim must exercise **the same pipeline the arms use** and must
not favour either model. ManiSkill's BimanualYAM is MolmoAct2's own training
embodiment and runs over the separate HTTP `/act` path; `gym-aloha` has ALOHA
kinematics.

The design is a **`SimRobot` implementing the same robot interface
`dk1lab/policy.py` already calls** — only the robot object swaps; rollout, FIFO
engine, limiter and home sweep are untouched code. UI is `mujoco.viewer`. Sim
`dt` fixed at 1/30 s so the policy sees its expected cadence, wall clock
decoupled by a `--realtime` / `--free-run` flag.

**The sim produces no episodes and no scores.** It exists to confirm each policy
drives the pipeline before it drives the arms.

### The scene is wrong, and knowingly so

Phase 1 passed on the only thing it was gating: both policies drove the sim arms
through the unmodified pipeline at 30 Hz with nothing starved. The *scene* they
drove is not good, and none of it is fixed yet:

- **The arms are on poles and cannot reach the table.** They are mounted on
  0.30 m pedestals — added because at the zero pose the DK1's elbow folds behind
  and below its own base, so an arm bolted flat starts with contact points
  through the table and its base yaw pinned. The pedestal solved that and created
  this: the workspace is now out of reach.
- **The bowl has a round base and four square walls.** It was built as "enough to
  tell in from out"; it is not a bowl.
- The arm spacing and camera poses are the *training rig's* (y = ±0.24 m, both
  facing +X), not this cell's, and were never a claim about it.

So a sim rollout right now says the pipeline runs. It says nothing about whether
a policy can do the task, and it must not be quoted as if it did. Fixing the
mounting geometry and the bowl is worth doing before the sim is used for
anything beyond that — it is not a blocker for Phase 2, which is on the arms.

---

## Running it

The profile, the checkpoint and the recorder are flags on one command. Nothing
below edits `dk1.toml`.

**The simulator** — no `/dev` node, no motor, nothing in the room moves:

```
dk1 policy run --sim --profile common --duration 60 \
  --task "put the dice in the bowl" --yes

dk1 policy run --sim --profile common --duration 60 \
  --checkpoint ~/Documents/RobotLearning/policies/pi05/base \
  --task "put the dice in the bowl" --yes
```

A MuJoCo window opens; `--no-view` runs it headless, `--display` adds the
per-joint Rerun panels. Read the result as *the pipeline runs* and nothing more.

**The scene photographs** — a video device, nothing else. No motor is energised
and no arm moves:

```
dk1 study photo --scene 1        # -> study/scene/1.jpg, off the top camera
```

**The arms**, one scored session per configuration. `--study` is what turns a
session into a row: it walks the scenes, asks for the rubric as each attempt
ends, and appends to `study/scores/<row>.csv`.

```
# A0 / A1 / B0 / B1 — the level playing field, recorded as a LeRobot dataset.
dk1 policy session --study A0 --profile common --duration 120 \
  --record-dataset --dataset-dir study/rollouts/A0

# R0 — the tuned rig. Scored, and recorded as .rrd only (a different lens).
dk1 policy session --study R0 --profile optimized --duration 120 --record
```

Add `--record` to also write the `.rrd` with the policy's own plan, into
`study/rrd/A0/`. It is the stream to diagnose a rollout against, and it is
hundreds of MB an episode; the dataset is what the study scores and fine-tunes
from.

`--scenes` and `--attempts` default to 3 and 3; `--scores-dir` and `--scene-dir`
default to `study/scores` and `study/scene`. A session started against a CSV
that already has attempts **resumes where it left off** — the file is the state.

**The prompt walks the three scenes in order**, three attempts each, and says
which one it is on. Setting the scene up is the operator's job; the session asks
for it and waits at the prompt:

```
=== scene 1 of 3 — put the dice and the bowl on their marks for scene 1 (study/scene/1.jpg) ===
[A0 scene 1/3, attempt 1/3 | episode 0 | 60s | dataset] task> put the dice in the bowl
  ... the arms go; Ctrl-C ends the attempt, then the arms sweep home

  score this attempt (scene 1/3, attempt 1/3)
    0 no purposeful motion  1 approach  2 contact  3 grasp  4 transport  5 success
  score> 3 left dropped it short of the bowl
  scene 1 attempt 1: score 3, left — dropped it short of the bowl -> study/scores/A0.csv
  1/9 done — next: scene 1/3, attempt 2/3

[A0 scene 1/3, attempt 2/3 | episode 1 | 60s | dataset] task>     <- Enter repeats it
```

The score line is `<0-5> [arm] [seconds] [note...]`, and it is the only prompt.
The arm is **required for anything above 0** — that column exists to settle an
open question about this cell. **Time-to-success is derived from the episode**,
which ends when you stop it, at the success; type a number only to override that,
and only on a 5. A line it cannot understand is asked again rather than dropped:
the attempt has already happened.

Enter at the prompt repeats the task, which is the ordinary case — the string
never changes. After the third attempt at a scene the session advances and asks
for the next layout; after scene 3 it says the row is complete. `:scene <n>` goes
back to a layout for an attempt that was **void** — the dice knocked off the desk
before the policy moved, a camera that had dropped out — and the extra attempt is
numbered onward (4) with the reason in its note. `:quit` leaves.

**Every recording in a scored session is kept, without asking.** In a scored row
a failure is evidence, and `keep this episode?` is one keypress away from
deleting it. The score prompt is the only question after an attempt.

`dk1 study scores A0` reads a row back: every attempt, then the per-scene grid.

The checkpoint defaults to `[policy]` in `dk1.toml` (MolmoAct2). For π0.5 pass
`--checkpoint ~/Documents/RobotLearning/policies/pi05/base`; the gripper
inversion turns itself off for it, and the borrowed normalisation is applied and
announced. **The demonstrations in Phase 3 must be recorded under `--profile
common`** — that is the invariant above.

---

## Phases

| | | gate |
| --- | --- | --- |
| **0** | Setup | **done 2026-08-25.** Both policies load and infer; `--profile common` exists; the LeRobot recorder writes a readable dataset; the MuJoCo scene runs |
| **1** | Sim check — nothing recorded | **done 2026-08-25.** Both policies drove the sim arms through the unmodified pipeline at ~29.9 Hz, no starved ticks. The scene itself is poor — see *The simulator* |
| **2** | Zero-shot on the arms — A0 then R0, N=9 each | **done.** A0 scored 2026-08-27 (**0/9**, ceiling 3) and R0 scored 2026-08-28 (**0/9**, ceiling 3). 18 scored attempts, two rows. The two earlier attempts on 2026-08-25 and 2026-08-26 both ended in a machine freeze and are not evidence; `docs/CRASH.md` closed the cause on 2026-08-27. Numbers in `study/results.md` |
| **3** | Collect demonstrations | **done 2026-08-28: 26 episodes, 18 484 frames**, `study/demos`, recorded under `--profile common`. See the amendment of 2026-08-28 — the target was cut from ~100 to 45 and 26 is what one session of hands produced |
| **4** | MolmoAct2 + LoRA — the training run | **run, not finished.** `study/finetune/R1-20260828-132023`, stopped at ~4 400 of 8 000 steps; checkpoints at 2 000 and 4 000, held-out loss 0.0201 → 0.0194 and still falling. Resumable. The run directory is in git; the weights are on the Hub |
| **5** | R1 and A1 on the arms, N=9 each | **R1 partial: 3 of 9** — scene 1 only, from the 4 000-step checkpoint, 2026-08-28. Scenes 2 and 3 unrun. **A1 not started** — no A1 checkpoint has been trained |
| **6** | π0.5 zero-shot — B0, N=9 | 9 scored attempts, one row |
| **7** | π0.5 + LoRA — the training run | a checkpoint, same recipe, same dataset bytes |
| **8** | B1 on the arms, N=9 | 9 scored attempts, one row |
| **9** | *If time:* action-expert-only, both models | |
| **10** | Interpretation → `study/results.md` | `study/results.md` **exists and holds the results**; the interpretation section is deliberately still unwritten — three rows have numbers, one of them is a third of a row, and there is nothing yet worth comparing |

**Phase 0.** `lerobot[pi,peft,dataset]` into the existing environment
(`molmoact2` and `training` are already there). Lay out
`~/Documents/RobotLearning/policies/{molmoact2-yam,pi05}/` with `base/` and
`lora/`; the existing `molmoact2/` folder there is the colleague's `sim_eval`
clone and stays untouched. All code stays in `trlc-dk1-niko`. Fetch
`lerobot/pi05_base`, verify with `dk1 policy check` / `smoke`. Build
`--profile common`, the LeRobot recorder, and the MuJoCo scene.

**Phase 2.** A0 then R0, one session each of nine attempts: scenes 1 to 3 in
order, three attempts apiece, scored into the CSV as they happen. R0 is the same
policy on the tuned rig, so it needs no new escalation once A0 has run.
Escalate as `CLAUDE.md` prescribes — `check`, `smoke`, `dryrun`, then `run`.
Photograph the three layouts with `dk1 study photo --scene N` when the marks are
set. π0.5 has never commanded these
arms and does not until Phase 6.

**Phase 3.** ~100 episodes by teleoperation under `--profile common`, dice start
varied within the marked region and the three scored scenes among them, bowl
fixed. Teleop stays uncapped: the cap
exists to bound a policy, and demonstrations come from a human hand.

**Phases 4 and 7 are the training runs, and 5, 6 and 8 are the arms.** Each pair
is split because a fine-tune that trains is not a fine-tune that works, and
running them as one phase is how a bad checkpoint reaches the cell before anyone
has looked at its loss curve. A training phase ends with a checkpoint, its
curve, and the run directory recorded — the checkpoint hash, the `dk1.toml` in
force, the command line, the git SHA. A scored phase ends with nine attempts in
a CSV. Confirm the gripper convention behaviourally on the **first** episode of
Phase 5 (the fine-tune speaks DK1, so the inversion goes **off**) before spending
nine attempts on it.

**Phase 10.** Beyond the tables: does the ranking change after fine-tuning, and
does the pretraining mix explain it? Does MolmoAct2's visual plan degrade under
the uncropped 105° input — it is inspectable, and π0.5 has no equivalent, which
is itself worth writing down. Did the 0.6 rad/s cap bound either policy, read off
the trace rather than off impressions? Did either arm systematically
underperform? And does any scene configuration separate the rows — a layout one
policy handles and another does not is the closest thing nine attempts can offer
to a claim about generalisation.

---

## Known risks, stated before they bite

**The machine froze six times in three days, and it is fixed.** It was the
platform firmware — the BIOS went **F6 -> F8a** on 2026-08-27 and the machine has
been stable since; `docs/CRASH.md` is the closed record. What it cost this study is
still real: **A0 has been attempted twice and scored twice, and has no usable
dataset behind either attempt.** Both rows must be re-run from scratch, and the
earlier scores are not evidence.
**Phase 3 — the ~100 demonstrations — must not start until it is understood**:
those episodes cost a day of somebody's hands and a freeze in the middle of them
loses that day.

**A crashed session used to lose every episode it had recorded, and did.**
On 2026-08-25 the machine froze during A0's eighth attempt. Seven episodes were
on disk and none of them can be opened: LeRobot v3.0 keeps one parquet writer
open for the whole session and writes the footer only on `finalize`, and it
buffers per-episode metadata ten at a time. The videos survived; the per-frame
state and action, and the whole of `meta/episodes/`, did not. **Fixed
2026-08-26** — one data file per episode, a metadata buffer of one, and both
writers closed after every committed episode, so what is on disk is readable
before the next attempt starts. Do not undo that for tidiness: it is the reason
an interrupted row can be resumed at all.

**The sim scene is not usable for anything but a pipeline check.** The arms are
on pedestals and cannot reach the table, and the bowl is a round base with square
walls. Known, recorded, not fixed. See *The simulator*.

**π0.5 has no DK1 normalization statistics.** Its base checkpoint knows nothing
about this 14-D action space, so a literal zero-shot load may not build a
normalizer at all. Mitigation: `dk1-merge-2026-03`'s `meta/stats.json`, which
exists, covers exactly the `bi_dk1_follower` 14-D layout, and is available now.
**B0 is therefore "zero-shot weights, borrowed normalization" and must be
labelled that way every time it is quoted.**

**The 0.6 rad/s cap may dominate the result.** Accepted deliberately; report it
as a caveat on the absolute numbers, not on the comparison.

**N = 9 on one task is still a small number.** Three scenes of three is enough
to say a policy works nowhere, or works everywhere; 4/9 against 6/9 is not a
significant difference and will not be reported as one. Nor is one scene against
another at n = 3.

**Dice-in-bowl barely exercises bimanual coordination**, so the study says
nothing about it. Stated, not discovered.

---

## Out of scope

More tasks, more models, more seeds, Cartesian action spaces, `dk1-merge`
two-stage pretraining, and any claim of the form "model X is better than model Y".

---

## Amendments

Changes to this protocol after Phase 2 begins go here, dated, with the reason.
A protocol that moves silently is not a protocol.

| date | change | why |
| --- | --- | --- |
| 2026-08-25 | B0 moves from Phase 2 to Phase 5 — MolmoAct2 all the way through first | The cell is worth setting up for one policy at a time, and both fine-tunes need the same ~100 demonstrations. The comparison is unaffected: same task, same scene, same profile, and B0 is still zero-shot weights |
| 2026-08-25 | **N goes from 5 to 9 per row** — three scene configurations, three attempts each, run and scored grouped by scene | Five attempts at one layout measure one layout. The three marked dice/bowl positions are what separate a policy that can do the task from one that can reach one corner, and three attempts per scene keep a single lucky grasp from carrying it. The task string, the profiles, the rubric and the rows are unchanged; the study grows from 25 attempts to 45 |
| 2026-08-25 | A scored row's `.rrd` go to **`study/rrd/<row>/`** | So a scored row's recordings never mix into `recordings/`, which holds six unscored tasks from before the study |
| 2026-08-25 | **R0 is dropped.** The optimized configuration will not be captured again | Nikolas's call: it sits outside this comparison rather than as its first row. Every row that runs is now under one profile, `common`, which is what the level playing field was for. What is given up is a measurement of what that playing field costs — R0's only job — and it is not worth a session on the arms |
| 2026-08-25 | In a scored session every recording is kept **without asking** | `keep this episode?` is one keypress away from deleting an attempt that cannot be repeated, and in a scored row a 0 is exactly as much evidence as a 5. The score prompt replaces it |
| 2026-08-25 | The phases split: fine-tuning and scoring are separate phases (4/5 for MolmoAct2, 7/8 for π0.5), and π0.5 zero-shot gets its own phase 6 | A fine-tune that trains is not a fine-tune that works. Keeping them in one phase is how a bad checkpoint reaches the arms before anyone has read its loss curve, and it hides which half a slipped schedule is stuck in. Nothing about the rows, the task, the scenes or the rubric moves |
| 2026-08-25 | Time-to-success is **derived from the episode**, not typed | The episode ends when the operator stops it, which is at the success; asking for a number they would read off the same clock added a prompt and a chance to mistype. A time can still be typed inline to override it |
| 2026-08-25 | `--duration` for a scored row goes from 60 s to **120 s** | Nikolas's call, made at the cell before A0's first attempt. The 60 s was an estimate written before anyone had watched this policy attempt this task on these arms. It applies to every scored row, A0 included, so the rows stay comparable |
| 2026-08-26 | **R0 is back in**, after being dropped the day before | It answers what the other four cannot: how much of what fine-tuning buys could have been had by tuning the rig instead. A1 against A0 says the LoRA worked; A1 against R0 says it was worth more than a crop and a speed cap |
| 2026-08-27 | **A new row: R1 = MolmoAct2 optimized + LoRA**, and it is the fine-tune that runs *first*. A1 (`common` + LoRA) follows it; R0 stays where it is, late | The fine-tune now targets the rig this cell actually runs — the wrist crop and `[limits.policy]` — because that is the configuration a working policy would be deployed under, and tuning for the handicapped one first spends the training budget on a row that exists to make a comparison fair rather than to make the arms work. The five rows become six. Nothing about the task, the scenes, the rubric or the profiles moves: R1 is `optimized` + LoRA, and every existing row keeps its definition |
| 2026-08-27 | **The demonstration set is 45 episodes, not ~100** — 15 at each of the three scene configurations | Nikolas's call, before the first hour of it is spent. 45 is what one session of hands can produce, and it is enough to say whether a LoRA on this cell moves the number at all; if it does, more demonstrations are the cheapest next thing to buy. The ~100 in *The question* and in *Fine-tuning* should be read as "our own demonstrations", and the count reported with the result |
| 2026-08-27 | **Demonstrations are captured under `--profile common` — the full lens — and the `optimized` crop is applied at training time** for R1 | One dataset then serves both R1 and A1, which halves the hands and keeps the two rows trained on the same bytes. It does not weaken the invariant in *What gets recorded, and where*: the crop is a deterministic transform of a 1280x720 frame that `dk1lab/crop.py` already owns, so applying it to the recorded frames produces the same image the `optimized` camera would have delivered. Recording it the other way round — crop-first — cannot be undone and would leave A1 with no dataset at all |
| 2026-08-27 | `dk1 teleop --record-dataset` is how Phase 3 is recorded, and its `--profile` means what it means everywhere else | Teleoperation had no recorder; the demonstrations are the one thing this study cannot buy back. The capture profile it already had is now `--capture`, so `--profile optimized\|common` has one meaning across the CLI and cannot be confused with `[capture.teleop]` on the command line that records the dataset the fine-tunes are built from |
| 2026-08-27 | **The training-time crop is materialised, not applied at load.** `dk1 dataset crop` writes `study/demos-optimized`: a copy of the demonstrations whose two wrist video streams have been rewritten through `dk1lab/crop.py`'s own box. `dk1 policy finetune --row R1` refuses a dataset whose lens does not match the row | LeRobot's per-sample `image_transforms` hook cannot do it: `make_train_eval_datasets` passes it to the **training** dataset and `None` to the evaluation one, so the held-out loss would measure a different camera; and it is called per camera key without the key, so it cannot crop the wrists and leave the top view alone. Materialising also means the frames a row trained on can be opened and looked at. The price is one encode generation — the recorded frame is decoded, cropped and re-encoded, where the camera crops before its only encode — and a disk copy. Nothing else changes: the box is cropped and stretched back to 1280x720, so every frame's size, count and timestamp survives and the copy inherits the source's `meta/` unaltered |
| 2026-08-27 | **Early stopping becomes checkpoint selection.** The budget runs to the end, a checkpoint is written at every evaluation, and `dk1 policy curve` names the one with the lowest held-out loss | LeRobot 0.6.1 has no early stop: it computes an eval loss every `eval_steps` and logs it, and nothing acts on it. Selection is also the better instrument for a run that gets one overnight attempt — a stop cannot be un-run, and the curve is worth having whole. What is given up is GPU hours, not evidence. The hold-out, its size and its purpose are unchanged |
| 2026-08-27 | **The LoRA target modules are NOT the same in shape for the two models**, and `--adapt` is the lever. The default stays each policy's own | *Fine-tuning* above says both defaults are "the action expert's q/v projections plus the state and action IO projections". That is exactly pi0.5's default and **not** MolmoAct2's, whose default targets the vision-language model's linear leaves; and LeRobot's generic PEFT path freezes every base parameter first, so under the defaults R1 and A1 would adapt MolmoAct2's VLM with its 578 M action expert frozen solid while B1 adapts pi0.5's action expert. `--adapt vlm+expert` extends MolmoAct2's **own** regex over its action expert. Which to use is a decision, not a bug fix, so the default is unchanged and the banner says out loud what is being adapted |
| 2026-08-27 | **The hold-out is spread evenly across the three scene configurations**, not taken as the last ten episodes | LeRobot's `eval_split` holds out the last `ceil(n x split)` episodes per task, and the demonstrations are recorded grouped by scene — so the last ten of 45 are all scene 3. A validation set that is one layout measures one layout. Within a scene the picks are evenly spaced rather than the last few, because a session of teleoperation drifts and the steadiest demonstrations are at the end of each block. No patch was needed: the episode list is handed to LeRobot in an order whose tail is exactly the hold-out |
| 2026-08-28 | **26 demonstrations, not 45, and the hold-out drops from 10 to 4** | What one session of hands produced on 2026-08-28: 26 episodes, 18 484 frames, 10.3 minutes, recorded 10:47–11:08 under `common`. Ten held out of 26 is 38% of the set and leaves 16 episodes to train a rank-32 adapter on; four is 15% and leaves 22. The number is now `dk1lab.finetune.DEFAULT_BUDGET.holdout` and **Phase 7 must reuse it** — the two fine-tunes are only comparable if their splits are the same shape. The count is reported with every result: this study says what 26 of our own demonstrations buy, not ~100 |
| 2026-08-28 | **R1 and A1 run `--adapt vlm+expert`.** B1 keeps pi0.5's default, which already is its action expert | Nikolas's call, made against the amendment above. Under LeRobot's defaults MolmoAct2's adapter reaches its vision-language model and its 578 M action expert trains not at all, while pi0.5's reaches its action expert and nothing else — so the two rows would differ by which half of each model was adapted as well as by the model. `vlm+expert` extends MolmoAct2's **own** regex over its action expert, which is as close as the two get to one recipe. The choice is recorded per run in `dk1_run.json` (`adapt`, `adapting`) and printed in the banner, and Phase 7 must be read against it |
| 2026-08-28 | **The step budget is 8 000, with a checkpoint and an evaluation every 2 000** — so 2 000, 4 000, 6 000 and 8 000 | Nikolas's call, against the measured cost: 1.13 step/s and 5 min 27 s per full evaluation make 8 000 steps about 2.4 hours rather than the 6.7 that 20 000 would have been. It is roughly one pass over the 15 605 training frames, which is thin — and that is what the stops are for. The curve says whether it was enough, and `dk1 policy resume --steps` buys more without starting again. **Phase 7 reuses this budget**, so if it is raised for R1 it is raised for B1 too |
| 2026-08-28 | **A training run can be paused at a checkpoint and resumed.** `dk1 policy pause <run>` asks it to stop after the next checkpoint, losing nothing; `dk1 policy resume <run>` continues from `checkpoints/last` | A run is hours long on the machine that also drives the cell, and the alternative to a clean stop is Ctrl-C, which gives up everything since the last save. The resumed run is the *same* experiment and not a similar one: the dataset, the split, the recipe, the budget and the learning-rate schedule all come back from the checkpoint's own `train_config.json` rather than from what is typed the second time. Every resumed session is appended to `dk1_resume.jsonl`, so a checkpoint can say it came from more than one sitting |
| 2026-08-28 | **R1 was deployed from step 4 000 of an unfinished 8 000-step run**, and the row stands at 3 of 9. The budget was not raised and the run was not resumed | What happened, recorded rather than decided: the training run was stopped at ~4 400 steps with the held-out loss still falling, and the 4 000-step checkpoint was taken to the arms the same afternoon. Scene 1 was scored — 5, 5, 2 — and scenes 2 and 3 were not attempted. **Two questions are therefore open and both must be settled before Phase 7**, because that phase has to reuse whatever R1 and A1 used: *which checkpoint a row deploys* (finish the 8 000-step budget and select on held-out loss as the amendment of 2026-08-27 says, or amend the budget to 4 000 — either way the same rule has to apply to A1 and B1, or the rows differ by how far each training run got), and *whether R1's three scored attempts survive a change to that rule* (if the budget is finished, they belong to a different checkpoint from the six that would follow and the row is re-run whole). Nothing else about the protocol moves |
| 2026-08-28 | **R0 was run and scored: 0/9, ceiling step 3.** `.rrd` only, nine files in `study/rrd/R0/`, one session, no restarts | It is the control the amendment of 2026-08-26 put back in, and it is now measured rather than assumed. Recording it as `.rrd` and not as a LeRobot dataset is unchanged protocol — its lens differs from every other row's — which means R0 is the one scored row whose frames carry no per-frame state, and the `episode` column of `study/scores/R0.csv` holds the `.rrd` stem accordingly |
| 2026-08-28 | **`study/results.md` is the record and carries no interpretation yet**, and `docs/HUB.md` is where the data lives | Three of six rows have numbers and one of those is a third of a row; a comparison written now would be quoted long after the missing rows arrived. The tables, the provenance of every number, the loose ends and the continuation instructions are written; the reading of them is a later session's work, and Phase 10 is where it belongs. `docs/HUB.md` is new: the datasets, the `.rrd` and the checkpoints exceed what git can hold, so the push and pull commands are written down rather than improvised, and the file states what is deliberately not pushed and why |
