# STUDY.md — the comparison, and how it will be run

The protocol for comparing two policies on this cell, decided *before* the
measurements so the measurements can be trusted.

`CLAUDE.md` is what is true now about the code and the cell. `DIAGNOSTICS.md` is
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

**Scene reset.** Dice at a marked position, bowl at another. Photograph the
layout once to `study/scene.jpg` and reproduce it before every attempt — the
configurations are evaluated on different days, so the photograph is what holds
them comparable.

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
before spending five attempts.

---

## The five configurations

**N = 5 scored attempts each**, one task, 25 attempts total.

| # | configuration | profile | tuning | phase |
| --- | --- | --- | --- | --- |
| R0 | MolmoAct2 optimized | `optimized` | none — frozen reference | 2 |
| A0 | MolmoAct2 zero-shot | `common` | none | 2 |
| B0 | π0.5 zero-shot | `common` | none | 2 |
| A1 | MolmoAct2 + LoRA | `common` | LoRA | 4 |
| B1 | π0.5 + LoRA | `common` | LoRA | 5 |

`--duration` drops from the 180 s default to **60 s**: 180 was chosen for a
multi-object sim task, and dice-in-bowl is a single pick and place.

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

Also per attempt: time to success, **which arm was used**, and a one-line failure
note. That third column is not bookkeeping — `CLAUDE.md` carries an unexplained
observation that *the right arm was reported not to pick anything up*, and this
is the first thing that will produce enough labelled attempts to confirm or kill
it. If both policies favour the left arm, that is a finding about the cell.

Written to `study/scores/<config>.csv` **during** the session, not reconstructed
afterwards.

---

## What gets recorded, and where

**LeRobot dataset v3.0** for the demonstrations and every scored rollout;
viewable with `lerobot-dataset-viz` and the Hub's dataset viewer.

`dk1lab/record.py` and its `.rrd` output stay exactly as they are, behind
`--record-rrd`, used only when a rollout needs diagnosing against the policy's
own plan — the one stream the LeRobot format has no slot for.

```
study/
  scene.jpg              the reset layout, photographed once
  demos/                 ~100 teleop episodes, LeRobot v3.0     [not in git]
  rollouts/<config>/     one LeRobot dataset per scored run     [not in git]
  scores/<config>.csv    the rubric, one row per attempt        [tracked]
  results.md             the tables and the interpretation      [tracked]
recordings/              the eight legacy .rrd — UNCHANGED, and prior evidence
                         for the optimized configuration, not a scored row
                         (they span six different tasks and were never scored)
```

The datasets do not go in git — the eight `.rrd` already exceed GitHub's 100 MB
hard limit. They are LeRobot datasets, so they belong on the **Hugging Face
Hub**. Only `scores/` and `results.md` are tracked.

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
and reused verbatim. Hold out 10 of the ~100 episodes for validation and
early-stop on it. Log the checkpoint hash, the `dk1.toml` in force, the command
line and the git SHA into each run directory.

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
drives the pipeline before it drives the arms. Gripper contact tuning is the
schedule risk; if it slips, the sim degrades to a kinematic check — the arms
move, nothing is grasped — which is an acceptable fallback and not a reason to
delay Phase 2.

---

## Phases

| | | gate |
| --- | --- | --- |
| **0** | Setup | both policies load and infer; `--profile common` exists; the LeRobot recorder writes a readable dataset; the MuJoCo scene runs |
| **1** | Sim check — nothing recorded | both policies drive the sim arms through the unmodified pipeline |
| **2** | Zero-shot on the arms — R0, A0, then B0 **separately**, N=5 each | 15 scored attempts, three rows |
| **3** | Collect ~100 demonstrations | a LeRobot v3.0 dataset recorded under `--profile common` |
| **4** | MolmoAct2 + LoRA → A1, N=5 | |
| **5** | π0.5 + LoRA → B1, N=5 | |
| **6** | *If time:* action-expert-only, both models | |
| **7** | Interpretation → `study/results.md` | |

**Phase 0.** `lerobot[pi,peft,dataset]` into the existing environment
(`molmoact2` and `training` are already there). Lay out
`~/Documents/RobotLearning/policies/{molmoact2-yam,pi05}/` with `base/` and
`lora/`; the existing `molmoact2/` folder there is the colleague's `sim_eval`
clone and stays untouched. All code stays in `trlc-dk1-niko`. Fetch
`lerobot/pi05_base`, verify with `dk1 policy check` / `smoke`. Build
`--profile common`, the LeRobot recorder, and the MuJoCo scene.

**Phase 2.** MolmoAct2 fully, then π0.5, in separate sessions. Escalate as
`CLAUDE.md` prescribes — `dryrun` before `run`, for each new policy. π0.5 has
never commanded these arms.

**Phase 3.** ~100 episodes by teleoperation under `--profile common`, dice start
varied within the marked region, bowl fixed. Teleop stays uncapped: the cap
exists to bound a policy, and demonstrations come from a human hand.

**Phase 7.** Beyond the tables: does the ranking change after fine-tuning, and
does the pretraining mix explain it? Does MolmoAct2's visual plan degrade under
the uncropped 105° input — it is inspectable, and π0.5 has no equivalent, which
is itself worth writing down. Did the 0.6 rad/s cap bound either policy, read off
the trace rather than off impressions? Did either arm systematically
underperform?

---

## Known risks, stated before they bite

**π0.5 has no DK1 normalization statistics.** Its base checkpoint knows nothing
about this 14-D action space, so a literal zero-shot load may not build a
normalizer at all. Mitigation: `dk1-merge-2026-03`'s `meta/stats.json`, which
exists, covers exactly the `bi_dk1_follower` 14-D layout, and is available now.
**B0 is therefore "zero-shot weights, borrowed normalization" and must be
labelled that way every time it is quoted.**

**The 0.6 rad/s cap may dominate the result.** Accepted deliberately; report it
as a caveat on the absolute numbers, not on the comparison.

**N = 5 on one task is a small number.** 2/5 against 3/5 is not a significant
difference and will not be reported as one.

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
| — | — | — |
