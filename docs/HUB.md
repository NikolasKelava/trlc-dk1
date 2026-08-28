# docs/HUB.md — publishing the study's data, and getting it back

The datasets, the `.rrd` recordings and the fine-tune checkpoints are **not in
git** and cannot be: the `.rrd` are hundreds of MB each and every one exceeds
GitHub's 100 MB hard limit, and the checkpoints are 1.2 GB apiece. They go to
the Hugging Face Hub. `study/results.md` § *Where the data is* is the inventory;
this file is the mechanics.

**What is in git**: the code, `dk1.toml`, the scores (`study/scores/*.csv`), the
scene photographs, `study/results.md`, and each fine-tune run's provenance —
`dk1_run.json`, its copy of `dk1.toml`, the two command lines and `train.log`.
Everything else is here.

**Nothing is pushed automatically.** Every command below is one you type.

---

## Once

```fish
set -x HF_USER <your-hugging-face-username>     # bash: export HF_USER=...
.venv/bin/hf auth login                          # a token with write scope
```

Fish is this machine's shell; in bash use `export`. The commands below assume
the repository root as the working directory.

`--private` is on every `hf upload` below. These are unpublished results — make
the repositories public deliberately, not by forgetting a flag.

---

## Before pushing R1: delete the orphaned staging frames

`study/rollouts/R1/images/episode-000003/` holds 6 516 staged PNG frames of a
fourth attempt that was never committed to the dataset — 4.4 GB of the
directory's 4.5 GB, and not recoverable as an episode (`study/results.md`
§ *Loose ends*). The `--exclude 'images/**'` in the R1 command below keeps them
out of the upload; if you would rather they were gone for good:

```fish
rm -rf study/rollouts/R1/images
```

Check what you are about to delete first — it is the only copy.

---

## The five pushes

Each is a single command and each is idempotent: re-running it uploads only what
changed.

### 1. The demonstrations — the one thing that cannot be re-recorded

26 teleoperated episodes, 18 484 frames, recorded 2026-08-28 under
`--profile common`. Both fine-tunes are built from these bytes; without them A1
and B1 cannot be trained at all.

```fish
.venv/bin/hf upload $HF_USER/dk1-demos study/demos . \
  --repo-type dataset --private \
  --exclude 'images/**' \
  --commit-message "DK1 dice-in-bowl teleop demonstrations, 26 episodes, profile=common"
```

`study/demos-optimized` is **not** pushed: it is a deterministic derivative,
rebuilt on any machine with
`dk1 dataset crop study/demos study/demos-optimized`.

### 2. A0 — MolmoAct2 zero-shot, `common`, nine scored attempts

```fish
.venv/bin/hf upload $HF_USER/dk1-rollouts-a0 study/rollouts/A0 . \
  --repo-type dataset --private \
  --exclude 'images/**' \
  --commit-message "A0: MolmoAct2 zero-shot, profile=common, 9 scored attempts, 0/9"
```

### 3. R1 — MolmoAct2 + LoRA, `optimized`, three scored attempts (partial row)

```fish
.venv/bin/hf upload $HF_USER/dk1-rollouts-r1 study/rollouts/R1 . \
  --repo-type dataset --private \
  --exclude 'images/**' \
  --commit-message "R1 (PARTIAL, scene 1 only): MolmoAct2 LoRA @4000 steps, profile=optimized, 3 of 9 attempts"
```

The row is unfinished — say so in the commit message and in the repository card,
because a dataset of three episodes named after a nine-attempt row is exactly
the thing a later reader will misread.

### 4. R0 — MolmoAct2 zero-shot, `optimized`, nine `.rrd`

R0 is scored but records no LeRobot dataset: its lens differs from every other
row's, so `STUDY.md` puts it in `.rrd` alone. These are not a dataset in the
LeRobot sense — they are nine files, 4.1 GB, each carrying the camera images,
the policy's own plan, the command and the observation.

```fish
.venv/bin/hf upload $HF_USER/dk1-rollouts-r0-rrd study/rrd/R0 . \
  --repo-type dataset --private \
  --commit-message "R0: MolmoAct2 zero-shot, profile=optimized, 9 scored attempts as Rerun .rrd, 0/9"
```

### 5. The R1 fine-tune — the 4 000-step checkpoint, resumably

This is the one push where **what is left out decides whether the work can be
continued**. `pretrained_model/` alone is enough to *deploy*; `training_state/`
— the 839 MB optimizer state, the RNG state and the scheduler — is what lets
`dk1 policy resume` carry the run to 8 000 steps as the *same* experiment. Both
go up. The 2 000-step checkpoint goes with them: it is the only other point on
the loss curve and the obvious alternative to deploy.

```fish
.venv/bin/hf upload $HF_USER/dk1-molmoact2-r1-lora \
  study/finetune/R1-20260828-132023 . \
  --repo-type model --private \
  --exclude 'train/checkpoints/last/**' \
  --commit-message "R1 LoRA on MolmoAct2-BimanualYAM: steps 2000 and 4000 of an unfinished 8000-step run, with optimizer state"
```

That uploads the whole run directory — the two checkpoints **and** the
provenance beside them: `dk1_run.json` (base checkpoint SHA-256, dataset,
split, recipe, budget, git SHA), the `dk1.toml` in force, `command.txt`,
`dk1_command.txt` and `train.log`. Those five text files are also in git; having
them next to the weights is what makes the repository self-describing.

`train/checkpoints/last` is a relative symlink to `004000` and is excluded —
recreate it after downloading (see below), because `dk1 policy resume` reads it.

---

## Getting it back on another machine

```fish
set -x HF_USER <the-same-username>
.venv/bin/hf auth login

.venv/bin/hf download $HF_USER/dk1-demos           --repo-type dataset --local-dir study/demos
.venv/bin/hf download $HF_USER/dk1-rollouts-a0     --repo-type dataset --local-dir study/rollouts/A0
.venv/bin/hf download $HF_USER/dk1-rollouts-r1     --repo-type dataset --local-dir study/rollouts/R1
.venv/bin/hf download $HF_USER/dk1-rollouts-r0-rrd --repo-type dataset --local-dir study/rrd/R0
.venv/bin/hf download $HF_USER/dk1-molmoact2-r1-lora --local-dir study/finetune/R1-20260828-132023

# the symlink that was excluded from the upload; dk1 policy resume reads it
ln -s 004000 study/finetune/R1-20260828-132023/train/checkpoints/last

# the cropped copy is derived, not downloaded
uv run dk1 dataset crop study/demos study/demos-optimized
```

Then, in order:

```fish
uv run dk1 dataset check study/demos                          # 26 episodes, lens=common
uv run dk1 study scores R0                                    # the scored rows read back
uv run dk1 study scores A0
uv run dk1 study scores R1
uv run dk1 policy curve study/finetune/R1-20260828-132023     # the loss curve and what to deploy
```

Two things do **not** come from the Hub and must be present locally:

- **the base checkpoint** `outputs/molmoact2_bimanual_yam_bf16` (10.1 GiB), or
  a fresh conversion of `lerobot/MolmoAct2-BimanualYAM-LeRobot`. `dk1_run.json`
  carries its SHA-256 — check it before trusting a re-conversion, because a
  LoRA adapter is meaningless against different base weights;
- **the cell itself.** `dk1.toml` is in git and is the single source of truth for
  ports and cameras, but the serial ports and `by-path` camera nodes are
  machine-specific. On new hardware run `dk1 find arms` and `dk1 find cameras`
  before anything is energised.

---

## What is deliberately not pushed

| | why |
| --- | --- |
| `study/rollouts/A0-crashed` (7.8 GB) | unreadable — `total_episodes: 0`, no `meta/episodes/`. The 2026-08-25 machine freeze took the per-frame state. Videos without state are not evidence and not worth 7.8 GB of anyone's bandwidth |
| `study/rollouts/R1/images/**` (4.4 GB) | staged PNG of an attempt that was never committed as an episode. Cannot be turned back into one |
| `study/demos-optimized` (289 MB) | a deterministic crop of `study/demos`; one command rebuilds it |
| `recordings/` (3.9 GB) | eight `.rrd` from 2026-08-21 across six different tasks, none of them scored and none part of the study. Push them if a colleague asks for them; they are not a result |
| `logs/` | per-session logs and machine telemetry, gitignored and machine-local. Keep them on this machine — `docs/CRASH.md` says what they are for |
