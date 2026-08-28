# study/results.md — the record of what was run and what it scored

`STUDY.md` is the protocol. This file is the **result**: what actually ran, on
which day, under which configuration, what it scored, and where the frames are.

> **Interpretation is deliberately not written yet.** As of 2026-08-28 three of
> the six rows have numbers, one of them is a third of a row, and the control
> that would separate the fine-tune from the rig has only just been measured.
> The tables below are facts. The reading of them belongs in a later session,
> once the missing rows exist. Do not quote a comparison out of this file that
> the file does not make.

---

## Status of the six rows

| # | configuration | profile | tuning | N scored | status |
| --- | --- | --- | --- | --- | --- |
| R0 | MolmoAct2 optimized | `optimized` | none | **9/9** | **done** 2026-08-28 |
| A0 | MolmoAct2 zero-shot | `common` | none | **9/9** | **done** 2026-08-27 |
| R1 | MolmoAct2 optimized + LoRA | `optimized` | LoRA @ 4 000 steps | **3/9** | **partial** — scene 1 only |
| A1 | MolmoAct2 + LoRA | `common` | LoRA | 0/9 | not started; no checkpoint trained |
| B0 | π0.5 zero-shot | `common` | none | 0/9 | not started; blocked on the PaliGemma licence |
| B1 | π0.5 + LoRA | `common` | LoRA | 0/9 | not started |

One task string throughout, byte-identical: `put the dice in the bowl`.
Three marked scene configurations, three attempts each, `--duration` 120 s.
Rubric 0–5 as in `STUDY.md` § *Scoring*; success is a 5.

---

## The scored rows

### R0 — MolmoAct2 zero-shot on the tuned rig

`--profile optimized` (wrist crop, `[limits.policy]` 1.0 rad/s), stock bf16
checkpoint, `--invert-gripper` **on**. One session, 2026-08-28 16:51–17:13,
nine attempts, no restarts. `.rrd` only, by protocol — its lens differs from
every other row's.

| highest step reached | attempt 1 | attempt 2 | attempt 3 |
| --- | --- | --- | --- |
| scene 1 | 3 (left) | 3 (left) | 3 (left) |
| scene 2 | 3 (right) | 3 (right) | 3 (right) |
| scene 3 | 2 (left) | 2 (left) | 2 (left) |

**Success rate 0/9.** Ceiling step 3 — grasp. Notes: scene 1 attempt 1
"dropped"; all three of scene 3 "does not reach the dice".

### A0 — MolmoAct2 zero-shot on the level playing field

`--profile common` (no crop, 0.6 rad/s), stock bf16 checkpoint,
`--invert-gripper` **on**. 2026-08-27 16:40–17:45.

| highest step reached | attempt 1 | attempt 2 | attempt 3 |
| --- | --- | --- | --- |
| scene 1 | 2 (left) | 2 (left) | 2 (left) |
| scene 2 | 3 (right) | 3 (right) | 3 (right) |
| scene 3 | 3 (right) | 1 (left) | 1 (left) |

**Success rate 0/9.** Ceiling step 3 — grasp.

### R1 — MolmoAct2 + LoRA on the tuned rig — PARTIAL, 3 of 9

`--profile optimized`, checkpoint
`study/finetune/R1-20260828-132023/train/checkpoints/004000`,
**`--no-invert-gripper`** (the fine-tune speaks DK1 — confirmed behaviourally on
its first episode). 2026-08-28 17:15–17:28.

| highest step reached | attempt 1 | attempt 2 | attempt 3 |
| --- | --- | --- | --- |
| scene 1 | **5** (both, 31.3 s) | **5** (both, 36.7 s) | 2 (right) |
| scene 2 | — | — | — |
| scene 3 | — | — | — |

**Scene 1 only: 2/3 successes.** Scenes 2 and 3 have not been attempted under
the recorded row. **This row is not comparable to R0 or A0 until all nine
attempts exist.**

A fourth episode was rolled out at 17:26–17:28 (81.8 s) and is **not** part of
the row: it was not scored, and the session ended before the dataset committed
it, so `study/rollouts/R1` holds three episodes and its staged frames remain
orphaned under `images/episode-000003/` (4.4 GB of PNG). See *Loose ends*.

---

## An earlier, superseded R1 pass — recorded here so it is not lost

Before the row above, R1 was rolled out on 2026-08-28 14:41–15:11 from the same
4 000-step checkpoint and scored **by hand into `study/scores/R1.csv` at 15:14**,
after the fact. That file has since been overwritten by the recorded row. Its
contents were:

```csv
scene,attempt,episode,score,seconds,arm,note
1,1,,5,25.8,both,
1,2,,5,26.1,both,
1,3,,5,26.8,both
2,1,,5,108.7,both,
2,2,,3,,left,
2,3,,5,98.1,both,
3,1,,5,30.2,left
3,2,,4,50.2,left
3,3,,5,55.3,left
```

**It is not a scored row and must not be quoted as one.** Four reasons, all
established from the session logs:

- the `episode` column is empty in all nine rows and **the frames no longer
  exist** — the dataset those sessions wrote to `study/rollouts/R1` was deleted
  before the row was re-run, so there is no join from any score to any frame;
- it was typed after the fact rather than through `--study R1` during the
  session, and only two of the nine `seconds` values (25.8, 108.7) match a
  logged episode duration;
- the durations that do match are the `episode N ran X s` figures, which
  **include the 4–9 s home sweep**, so those times are not times-to-success;
- the nine attempts span **six session restarts**, one of which was killed
  mid-episode by all three cameras timing out at once.

It is kept in this file because it is the only surviving trace of those
attempts, and because a later reader will otherwise find the deleted dataset in
the logs and wonder what it held.

---

## The fine-tune behind R1

`study/finetune/R1-20260828-132023/` — the run directory is tracked; its
checkpoints are not.

| | |
| --- | --- |
| row | R1 — `optimized` lens, so trained on `study/demos-optimized` |
| base checkpoint | `outputs/molmoact2_bimanual_yam_bf16`, sha256 `d9be53a3…1951b` |
| dataset | 26 episodes, 18 484 frames, recorded 2026-08-28 under `--profile common` and cropped at training time by `dk1 dataset crop` |
| split | 22 train / 4 hold-out (`4, 11, 15, 25`), spread across the three scenes |
| recipe | LoRA r=32, α=16, dropout 0.05, `modules_to_save=[]`, `--adapt vlm+expert`, 106 M trainable of 5.55 B |
| budget | 8 000 steps, batch 2, lr 1e-4, warmup 200, eval + checkpoint every 2 000 |
| git | `9e6d5b1`, branch `phase0-foundation`, clean |

**The run was stopped at ~4 400 of 8 000 steps** (13:20–14:37) and never
resumed. Checkpoints exist at 2 000 and 4 000; `checkpoints/last` points at
4 000. **R1 was deployed from the 4 000-step checkpoint**, i.e. from an
unfinished run.

| step | held-out loss | training loss (last logged) |
| --- | --- | --- |
| 2 000 | 0.0201 | 0.014 |
| 4 000 | **0.0194** | 0.010 |

At the stop the run had covered **0.56 epochs** over the 22 training episodes,
and the held-out loss was still falling. Cost as measured: 1.13 step/s, 5 min
27 s per full evaluation, 1.2 GB per checkpoint.

---

## Where the data is

Nothing below is in git. Sizes are on-disk as of 2026-08-28.

| what | path | size | state |
| --- | --- | --- | --- |
| the 26 demonstrations, full lens | `study/demos` | 487 MB | v3.0, readable |
| the same, cropped for R1 | `study/demos-optimized` | 289 MB | v3.0, readable, derived — regenerable with `dk1 dataset crop` |
| A0's nine attempts | `study/rollouts/A0` | 728 MB | v3.0, 9 episodes / 28 832 frames |
| R1's three attempts | `study/rollouts/R1` | 4.5 GB | v3.0, 3 episodes / 3 432 frames — **4.4 GB of that is orphaned PNG staging** |
| R0's nine attempts | `study/rrd/R0` | 4.1 GB | nine `.rrd`, one per attempt, named `0001…0009` |
| the R1 fine-tune | `study/finetune/R1-20260828-132023/train/checkpoints/` | 2.4 GB | 2 000 and 4 000, each with `pretrained_model` + `training_state` |
| A0's first, crashed attempt | `study/rollouts/A0-crashed` | 7.8 GB | **unreadable** — `total_episodes: 0`, no `meta/episodes/`. The 2026-08-25 freeze. Videos only; no per-frame state. Scored separately in `study/scores/A0-crashed.csv` (6 attempts) and **not evidence** |
| eight pre-study `.rrd` | `recordings/` | 3.9 GB | six different tasks, never scored, from 2026-08-21. Untracked and left that way: every file exceeds GitHub's 100 MB limit |
| per-session logs and telemetry | `logs/` | — | one `.log` + one `.jsonl` per session, gitignored, machine-local |

The scored rows join to their frames through the `episode` column of
`study/scores/<row>.csv`: a dataset episode index for A0 and R1, the `.rrd`
stem for R0.

---

## Loose ends, stated so they are not rediscovered

- **R1 is 3/9.** Scenes 2 and 3 are unrun. The row needs six more attempts from
  the same 4 000-step checkpoint, `--profile optimized`,
  `--no-invert-gripper`, into the *same* `study/rollouts/R1` directory — the
  session resumes an existing dataset and `--study R1` resumes the CSV, so
  re-running the command continues at scene 2 attempt 1.
- **`study/rollouts/R1/images/episode-000003/`** holds 6 516 staged PNG frames
  of a fourth, unscored attempt whose parquet state was never committed. They
  cannot be turned back into an episode and they are 4.4 GB. Delete them before
  pushing the dataset, or the upload carries them.
- **The R1 fine-tune is unfinished at 4 400 of 8 000 steps.** It is resumable —
  `dk1 policy resume study/finetune/R1-20260828-132023` — provided
  `checkpoints/last/training_state` travels with it.
- **A1, B0 and B1 have no numbers.** B0 and B1 are additionally blocked on the
  gated `google/paligemma-3b-pt-224` licence.
- **`study/scores/R1.csv` was overwritten once.** The lost content is preserved
  above.
- **The camera dropout is live.** One session on 2026-08-28 died mid-episode
  with all three cameras raising `TimeoutError ... read failed (status=False)`
  at once. It has happened once, it cost one attempt, and it has no diagnosis.

---

## How to continue this

Everything a second machine needs, in order.

**1. Get the code.** Clone the fork, branch `phase0-foundation`, and read
`CLAUDE.md` first, then `STUDY.md`. `docs/DIAGNOSTICS.md` is the record of every
fault already chased — read the relevant section before re-measuring anything.

**2. Get the data.** The datasets, the `.rrd` and the checkpoints are on the
Hugging Face Hub, not in git; the push and pull commands are in
*Publishing to the Hub* below. `study/demos-optimized` need not be pulled — it is
`dk1 dataset crop study/demos study/demos-optimized`.

**3. Finish R1.** Six attempts, scenes 2 and 3:

```
dk1 policy session --study R1 --profile optimized --duration 120 \
  --checkpoint study/finetune/R1-20260828-132023/train/checkpoints/004000/pretrained_model \
  --no-invert-gripper \
  --record-dataset --dataset-dir study/rollouts/R1
```

The CSV is the state: it already holds scene 1, so the session opens at scene 2
attempt 1. Set the scene from `study/scene/2.jpg` before it starts.

**4. Then, in `STUDY.md`'s order:** A1's training run
(`dk1 policy finetune --row A1 --dataset-dir study/demos --adapt vlm+expert`),
A1 on the arms, then π0.5's B0 and B1 once the licence is accepted.

**5. Two protocol decisions are open and must be settled before B1**, because
Phase 7 has to reuse whatever R1 and A1 used:

- **which checkpoint gets deployed.** R1 ran from step 4 000 of an unfinished
  8 000-step budget. Either finish the budget and select on held-out loss as the
  amendment of 2026-08-27 says, or amend the protocol to a 4 000-step budget —
  but the same rule must apply to A1 and B1 or the rows differ by how far each
  run got.
- **whether the R1 row stands on the 4 000-step checkpoint.** If the budget is
  finished, the three scene-1 attempts already scored belong to a different
  checkpoint from the six that would follow, and the row has to be re-run whole.

**6. What must not drift.** The task string, the three scene marks and their
photographs, the rubric, `--duration 120`, the 26-episode dataset and its 4-
episode hold-out, and `--adapt vlm+expert` for every MolmoAct2 row. The gripper
inversion is **on** for a zero-shot row and **off** for a fine-tuned one.

---

## Publishing to the Hub

The exact commands used, kept here so a repeat is identical rather than
improvised: see the repository's `docs/HUB.md`.
