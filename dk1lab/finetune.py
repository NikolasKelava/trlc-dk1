"""The LoRA fine-tune: the recipe, the split, the run directory, and two repairs.

``STUDY.md`` Phase 4 is a training run, and what it has to produce is not only a
checkpoint: it is *a checkpoint whose provenance is recorded* — the base
checkpoint's hash, the ``dk1.toml`` in force, the command line, and the git SHA.
This module is that bookkeeping plus the arithmetic around it, and it imports
neither torch nor LeRobot at module level for the same reason
:mod:`dk1lab.config` does not: deciding what to train is a decision about
configuration, and it should be testable on a machine with no robot stack.

The training itself is **LeRobot's**, unchanged. :func:`train_argv` builds a
``lerobot-train`` command line and :func:`run` executes it through LeRobot's own
entry point, so what the run directory records really is what ran.

Two things had to be repaired to get there, and both are stated here because
both would otherwise be discovered at 2 a.m.

**The processor overrides do not fit MolmoAct2, and the run dies before step 1.**
``lerobot_train`` builds ``preprocessor_overrides`` for ``normalizer_processor``
and ``postprocessor_overrides`` for ``unnormalizer_processor`` whenever the
policy is loaded from a path — and then
``PolicyProcessorPipeline._validate_overrides_used`` **raises** for any override
key that names no saved step. MolmoAct2 normalises through
``molmoact2_masked_normalizer``, keyed by ``norm_tag``, and has no step by
either name. Verified on the real checkpoint::

    KeyError: Override keys ['normalizer_processor'] do not match any step in
    the saved configuration. Available step keys: ['rename_observations_processor',
    'to_batch_processor', 'molmoact2_state_frame_transform',
    'molmoact2_masked_normalizer', ...]

:func:`prune_overrides` drops the keys the saved pipeline does not have, and
:func:`patched` installs it. It is the same class of upstream bug as the two
cherry-picks in ``CLAUDE.md`` and as ``--policy.joint_signs`` doing nothing at
rollout: a config path that parses, validates, and then does not fit. Worth
upstreaming; do not do it.

Note what the pruning means for normalisation. MolmoAct2 keeps its statistics
inside the masked normalizer under its ``norm_tag``, so the fine-tune trains and
deploys under the **same** statistics the zero-shot rows ran under. Our
demonstrations are in DK1 convention and the YAM statistics are the units they
get expressed in; the adapter learns the rest. That is a property to state, not
one to fix — changing it would make A1 and A0 disagree about more than the LoRA.

**The held-out episodes must not all be one scene.** LeRobot's ``eval_split``
holds out *the last* ``ceil(n * split)`` episodes per task, and the
demonstrations are recorded grouped by scene — 15 of scene 1, then 15 of scene 2,
then 15 of scene 3. Taking the last ten would validate entirely on scene 3 and
say nothing about the other two. :func:`split_episodes` takes an even spread from
each scene instead, and hands LeRobot the episode list **in an order whose tail
is exactly that hold-out**, which needs no patch at all: ``LeRobotDataset``
stores ``episodes`` verbatim and the split walks it in order.

**There is no early stop, and this does not add one.** LeRobot 0.6.1 computes an
eval loss every ``eval_steps`` and logs it; nothing stops or selects on it. So
the budget is run to the end, a checkpoint is written at every eval, and
:func:`best_checkpoint` reads the log back and names the one with the lowest
held-out loss. That is strictly more informative than stopping — a stop cannot
be un-run, and an overnight run gets one attempt — and it is a departure from
``STUDY.md``'s wording, which says *early-stop on it*. Amend the protocol or
change this; do not leave the two disagreeing.
"""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Generator, Iterable, Mapping, Sequence

from .layout import IMAGE_KEYS
from .runprofile import COMMON, OPTIMIZED

__all__ = [
    "DEFAULT_BUDGET",
    "DEFAULT_RUNS_DIR",
    "LoraRecipe",
    "RECIPE",
    "ROWS",
    "Row",
    "Split",
    "TRAIN_SUBDIR",
    "TrainingBudget",
    "best_checkpoint",
    "episode_scenes",
    "eval_losses",
    "file_sha256",
    "git_sha",
    "patched",
    "molmoact2_target_modules",
    "prune_overrides",
    "read_notes",
    "row",
    "run",
    "split_episodes",
    "target_modules",
    "train_argv",
    "train_dir",
    "write_run_dir",
]


class FinetuneError(Exception):
    """Raised for a fine-tune that cannot be set up. Nothing has run yet."""


#: Where run directories go. One per training run, named by row and time.
DEFAULT_RUNS_DIR = Path("study/finetune")

#: LeRobot's output directory, **inside** the run directory rather than equal to
#: it. ``TrainPipelineConfig.validate`` raises ``FileExistsError`` for an
#: ``output_dir`` that already exists, and the provenance has to be written
#: *before* training starts — a run directory that only appears on success is
#: missing from exactly the runs worth explaining. So the run directory holds
#: ``dk1_run.json``, ``dk1.toml``, ``command.txt`` and ``train.log``, and LeRobot
#: owns the ``train/`` beneath it.
TRAIN_SUBDIR = "train"


def train_dir(run_dir: Path | str) -> Path:
    """Where LeRobot writes, inside a run directory. See :data:`TRAIN_SUBDIR`."""
    return Path(run_dir) / TRAIN_SUBDIR


# --------------------------------------------------------------------------- #
# The recipe
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LoraRecipe:
    """The adapter, fixed for **both** models and both training phases.

    ``STUDY.md`` fixes these four so that what separates A1 from B1 is the model
    and not the recipe. They are passed to LeRobot's ``--peft.*`` interface,
    which merges them over each policy's own default ``target_modules`` — so the
    *shape* of what is adapted stays each policy's business and the *size* of it
    is ours.

    Attributes:
        r: LoRA rank. 32.
        lora_alpha: the scaling numerator; the adapter scales by ``alpha / r``.
            16, i.e. a scale of 0.5 at r=32.
        lora_dropout: 0.05. **Not** part of LeRobot's ``PeftConfig``, which
            carries only ``target_modules``, ``full_training_modules``,
            ``method_type``, ``init_type``, ``r`` and ``lora_alpha``. It reaches
            the adapter through the *policy* config's ``lora_dropout`` instead,
            which :func:`train_argv` pins explicitly rather than inheriting —
            the default happens to be 0.05 today and an inherited default is not
            a recorded decision.
        modules_to_save: empty, spelled ``--peft.full_training_modules='[]'``.
            The default is each policy's newly-created IO projections, which is
            right when fine-tuning a base model and wrong here: both checkpoints
            are already trained on robot data, so training those layers fully
            would make the two rows differ by more than an adapter.

    ``method_type`` is ``lora`` and is not a field: a second method would be a
    different study.
    """

    r: int = 32
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    modules_to_save: tuple[str, ...] = ()

    @property
    def scale(self) -> float:
        """``alpha / r`` — how hard the adapter pushes on the frozen weights."""
        return self.lora_alpha / self.r

    def describe(self) -> str:
        """One line for the banner and the run directory."""
        return (
            f"LoRA r={self.r} alpha={self.lora_alpha} (scale {self.scale:g}) "
            f"dropout={self.lora_dropout:g} "
            f"modules_to_save={list(self.modules_to_save)}"
        )


#: The recipe. Fixed in Phase 4 and reused verbatim in Phase 7.
RECIPE = LoraRecipe()


# --------------------------------------------------------------------------- #
# What gets adapted
# --------------------------------------------------------------------------- #
#
# `STUDY.md` says of the two policies: "Both models define default targets and
# they are the same recipe in shape: the action expert's q/v projections plus the
# state and action IO projections." That is exactly true of pi0.5 and **not true
# of MolmoAct2**, and the difference is not cosmetic.
#
# pi0.5's `_get_default_peft_targets` targets
#     .*\.gemma_expert\..*\.self_attn\.(q|v)_proj  and  the IO projections
# — the action expert.
#
# MolmoAct2's targets the **vision-language model**: the transformer's and vision
# backbone's linear leaves. Its action expert is included only when
# `enable_lora_action_expert` is set, which is off by default. And LeRobot's
# generic PEFT path (`wrap_with_peft`, which is what `--peft.*` drives) freezes
# every base parameter first — so under the defaults, R1 and A1 would adapt
# MolmoAct2's VLM with its 578 M action expert **frozen solid**, while B1 adapts
# pi0.5's action expert and nothing else. Two different experiments.
#
# Turning it on through the policy config is not available: MolmoAct2 rejects
# `enable_lora_action_expert` without `enable_lora_vlm`, and `enable_lora_vlm`
# triggers the policy's own internal LoRA in `__init__` — which `wrap_with_peft`
# would then wrap a second time. So the lever is `--peft.target_modules`, and
# what it is given is MolmoAct2's own regex with the action-expert branch
# included. `tests/test_finetune.py` asserts it still equals what LeRobot builds,
# so this cannot drift silently.

#: What the adapter is attached to, as ``--adapt`` accepts it.
ADAPT_DEFAULT = "default"
ADAPT_VLM_AND_EXPERT = "vlm+expert"
ADAPT_CHOICES = (ADAPT_DEFAULT, ADAPT_VLM_AND_EXPERT)


def molmoact2_target_modules(*, action_expert: bool, prefix: str = r"model\.model") -> str:
    """MolmoAct2's LoRA target regex, action expert included or not.

    A transcription of ``MolmoAct2Policy._lora_target_modules``, which is an
    instance method and therefore not reachable before the weights are on the
    GPU — and the command line has to be built, printed and recorded before that.
    Pinned against the real one by a test rather than trusted.
    """
    leaves = "w1|w2|w3|wq|wk|wv|wo|att_proj|attn_out|ff_proj|ff_out|patch_embedding"
    vlm = rf"{prefix}\.(transformer|vision_backbone)\.(?:.*\.)?({leaves})$"
    if not action_expert:
        return vlm
    expert = (
        r"time_embed\.(1|3)|"
        r"action_embed|context_k_proj|context_v_proj|"
        r"blocks\.\d+\.self_attn\.(qkv|out_proj)|"
        r"blocks\.\d+\.cross_attn\.(q_proj|out_proj)|"
        r"blocks\.\d+\.mlp\.(up_proj|gate_proj|down_proj)|"
        r"blocks\.\d+\.modulation\.linear|"
        r"final_layer\.(modulation\.linear|linear)"
    )
    return f"({vlm}|" + rf"{prefix}\.action_expert\.({expert})$)"


def target_modules(family: str, adapt: str) -> str | None:
    """The ``--peft.target_modules`` value for one policy, or ``None`` for its default.

    ``default`` leaves each policy's own choice alone, which is what ``STUDY.md``
    prescribes literally. ``vlm+expert`` is MolmoAct2 only, and is what makes its
    adapter cover the action expert the way pi0.5's default already does.

    Raises:
        FinetuneError: for ``vlm+expert`` on pi0.5, whose default already is the
            action expert and which has no VLM branch to add to it.
    """
    if adapt not in ADAPT_CHOICES:
        raise FinetuneError(
            f"no such --adapt {adapt!r} — expected one of {', '.join(ADAPT_CHOICES)}"
        )
    if adapt == ADAPT_DEFAULT:
        return None
    if family != "molmoact2":
        raise FinetuneError(
            f"--adapt {ADAPT_VLM_AND_EXPERT} is MolmoAct2's regex; {family}'s default "
            f"targets its action expert already"
        )
    return molmoact2_target_modules(action_expert=True)


def describe_adapt(family: str, adapt: str) -> str:
    """One line saying what the adapter will and will not reach."""
    if adapt == ADAPT_VLM_AND_EXPERT:
        return "the VLM's linear leaves AND the action expert (MolmoAct2's own regex)"
    if family == "pi05":
        return "pi0.5's default: the action expert's q/v projections and the IO projections"
    return (
        "MolmoAct2's default: the VLM's linear leaves only — the 578 M action "
        "expert stays FROZEN"
    )


@dataclass(frozen=True)
class TrainingBudget:
    """A step budget, not epochs — and everything that hangs off it.

    ``STUDY.md``: *a step budget, not epochs — fixed once in Phase 4 and reused
    verbatim in Phase 7*. Epochs would make the two models' training depend on
    how many frames each demonstration happened to take.

    ``eval_steps`` and ``save_freq`` are deliberately **equal**: every checkpoint
    that exists has a held-out loss measured next to it, which is the only way
    :func:`best_checkpoint` can name one. Let them drift apart and the best loss
    belongs to a checkpoint that was never written.

    Attributes:
        steps: optimiser updates. Not epochs.
        batch_size: frames per update. Small, because the model is 5.44 B
            parameters on a 32 GB card even with the base weights frozen.
        lr: peak learning rate. **1e-4, not the checkpoint's 1e-5.** The preset
            is a full fine-tune's rate; a rank-32 adapter at scale 0.5 barely
            moves at it. Change it against a loss curve, not against this line.
        warmup: steps of linear warmup before the cosine decay.
        holdout: episodes reserved for validation, taken evenly from every
            scene. **Four**, amended from ``STUDY.md``'s ten on 2026-08-28:
            ten was set against 45 demonstrations and 26 were recorded, where it
            would hold out 38% of them. Four is 15%, and the same number has to
            be reused in Phase 7 for the two rows to be comparable.
        gradient_checkpointing: recompute activations instead of storing them.
            **On.** The alternative on this card is an out-of-memory error a few
            hundred steps in, which is the expensive way to find out.
    """

    steps: int = 20_000
    batch_size: int = 2
    lr: float = 1e-4
    warmup: int = 200
    eval_steps: int = 1_000
    save_freq: int = 1_000
    holdout: int = 4
    num_workers: int = 4
    seed: int = 1000
    gradient_checkpointing: bool = True


#: The budget, unless a flag moves it. Recorded in every run directory either way.
DEFAULT_BUDGET = TrainingBudget()


# --------------------------------------------------------------------------- #
# The rows
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Row:
    """One line of ``STUDY.md``'s results table that needs a training run.

    Attributes:
        name: ``R1``, ``A1``, ``B1``.
        profile: the run profile the fine-tuned policy is **rolled out** under,
            and therefore the lens its training data has to carry.
        cropped: whether the demonstrations need the ``optimized`` wrist crop
            applied before training. ``True`` for ``R1`` only, and it is the
            whole reason :mod:`dk1lab.recrop` exists.
        invert_gripper: what ``dk1 policy session`` must be given afterwards.
            **Off for every row here.** The demonstrations are in DK1 convention
            because that is what the robot reports, so a fine-tuned policy
            outputs DK1 convention and inverting it flips every grasp.
            ``STUDY.md`` § *The gripper convention*.
        family: the checkpoint type this row fine-tunes.
    """

    name: str
    profile: str
    cropped: bool
    invert_gripper: bool
    family: str
    summary: str


ROWS: dict[str, Row] = {
    "R1": Row(
        name="R1",
        profile=OPTIMIZED,
        cropped=True,
        invert_gripper=False,
        family="molmoact2",
        summary=(
            "MolmoAct2 on the tuned rig — the wrist crop and [limits.policy] — "
            "fine-tuned on demonstrations recorded through the full lens and "
            "cropped at training time"
        ),
    ),
    "A1": Row(
        name="A1",
        profile=COMMON,
        cropped=False,
        invert_gripper=False,
        family="molmoact2",
        summary=(
            "MolmoAct2 on the level playing field — no crop, [limits.study] — "
            "fine-tuned on the demonstrations exactly as recorded"
        ),
    ),
    "B1": Row(
        name="B1",
        profile=COMMON,
        cropped=False,
        invert_gripper=False,
        family="pi05",
        summary=(
            "pi0.5 on the level playing field, fine-tuned on the same "
            "demonstration bytes as A1 — which is what keeps the comparison clean"
        ),
    ),
}


def row(name: str) -> Row:
    """The :class:`Row` called ``name``.

    Raises:
        FinetuneError: naming what there is. A misspelled row that fell back to a
            default would train for a configuration nobody chose, and the
            checkpoint would look exactly like the one that was asked for.
    """
    try:
        return ROWS[name.strip().upper()]
    except KeyError:
        raise FinetuneError(
            f"no such row {name!r} — this study fine-tunes {', '.join(ROWS)}. "
            f"R0, A0 and B0 are zero-shot rows and have no training run."
        ) from None


# --------------------------------------------------------------------------- #
# The hold-out
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Split:
    """Which episodes train and which validate, and how to say so to LeRobot.

    Attributes:
        train: episode indices used for training.
        holdout: episode indices held out.
        order: ``train + holdout``. This is the list passed as
            ``--dataset.episodes``, and its **tail is the hold-out** —
            ``make_train_eval_datasets`` splits the last ``ceil(n * eval_split)``
            of whatever order it is handed, and ``LeRobotDataset`` stores that
            order verbatim.
        eval_split: the fraction that makes that ``ceil`` land on
            ``len(holdout)``. Half a step below the exact ratio, so a value that
            divides evenly cannot round up to one episode too many.
        scenes: the scene each held-out episode came from, for the banner.
    """

    train: tuple[int, ...]
    holdout: tuple[int, ...]
    order: tuple[int, ...]
    eval_split: float
    scenes: dict[int, int | None] = field(default_factory=dict)

    def describe(self) -> str:
        """One line: how many of each, and which scenes the hold-out spans."""
        by_scene: dict[Any, int] = {}
        for episode in self.holdout:
            by_scene[self.scenes.get(episode)] = by_scene.get(self.scenes.get(episode), 0) + 1
        spread = ", ".join(
            f"scene {key}: {count}" if key is not None else f"unlabelled: {count}"
            for key, count in sorted(by_scene.items(), key=lambda item: (item[0] is None, item[0]))
        )
        return (
            f"{len(self.train)} train, {len(self.holdout)} held out "
            f"({spread or 'no scene labels'}) — eval_split={self.eval_split:.5f}"
        )


def split_episodes(
    scenes: Mapping[int, int | None], holdout: int = DEFAULT_BUDGET.holdout
) -> Split:
    """Hold ``holdout`` episodes out, taken evenly from every scene.

    Args:
        scenes: ``{episode index: scene number or None}``, as
            :func:`episode_scenes` reads it off ``dk1_notes.jsonl``.
        holdout: how many episodes to reserve.

    Why evenly rather than the last ``holdout``: the demonstrations are recorded
    **grouped by scene**, so the tail of the dataset is one layout. A validation
    set that is entirely scene 3 measures scene 3, and the loss curve it draws
    would say nothing about whether the adapter helped the other two.

    Within a scene the picks are evenly spaced rather than the last few, because
    a session of teleoperation drifts — the last demonstrations of a block are
    the ones made by the steadiest hand, and holding out only those makes the
    validation set easier than the training set.

    Deterministic, with no seed: an evenly spaced pick over a sorted list is the
    same on every machine, and a hold-out that moved between two runs of the
    same command would make their loss curves incomparable.

    Raises:
        FinetuneError: if the hold-out would leave nothing to train on, or if it
            is not a positive number. Both are typing mistakes that would
            otherwise be found several GPU-hours later.
    """
    episodes = sorted(scenes)
    if holdout <= 0:
        raise FinetuneError(
            f"holdout must be at least 1 episode, got {holdout} — "
            f"STUDY.md fixes it at {DEFAULT_BUDGET.holdout}"
        )
    if holdout >= len(episodes):
        raise FinetuneError(
            f"holding out {holdout} of {len(episodes)} episodes leaves "
            f"{len(episodes) - holdout} to train on"
        )

    grouped: dict[Any, list[int]] = {}
    for episode in episodes:
        grouped.setdefault(scenes[episode], []).append(episode)

    # Largest-remainder over the scenes, so the hold-out is proportional and its
    # total is exactly `holdout` — a per-scene round() can miss by one either way.
    keys = sorted(grouped, key=lambda key: (key is None, key))
    exact = {key: holdout * len(grouped[key]) / len(episodes) for key in keys}
    take = {key: int(exact[key]) for key in keys}
    for key in sorted(keys, key=lambda key: (-(exact[key] - take[key]), key is None, key)):
        if sum(take.values()) >= holdout:
            break
        take[key] += 1

    held: list[int] = []
    for key in keys:
        held.extend(_evenly(grouped[key], take[key]))
    held.sort()

    train = tuple(episode for episode in episodes if episode not in set(held))
    return Split(
        train=train,
        holdout=tuple(held),
        order=tuple(train) + tuple(held),
        eval_split=_eval_split(len(episodes), len(held)),
        scenes=dict(scenes),
    )


def _evenly(items: Sequence[int], count: int) -> list[int]:
    """``count`` items spread across ``items``, endpoints included."""
    if count <= 0:
        return []
    if count >= len(items):
        return list(items)
    if count == 1:
        return [items[len(items) // 2]]
    step = (len(items) - 1) / (count - 1)
    return [items[round(index * step)] for index in range(count)]


def _eval_split(total: int, holdout: int) -> float:
    """The fraction whose ``ceil(total * f)`` is exactly ``holdout``.

    Half an episode below the exact ratio. ``10/45`` is 0.2222…, and a value
    rounded *up* anywhere — or the exact ratio landing on a whole number through
    floating point — makes ``ceil`` take eleven. ``9.5/45`` cannot.
    """
    if total <= 0 or holdout <= 0:
        return 0.0
    return (holdout - 0.5) / total


# --------------------------------------------------------------------------- #
# Reading the demonstrations' own notes
# --------------------------------------------------------------------------- #


def read_notes(root: Path | str) -> list[dict[str, Any]]:
    """Every record in a dataset's ``dk1_notes.jsonl``, in the order written.

    The file is this fork's, not LeRobot's: v3.0 has no per-episode free-form
    slot. It is what carries the scene, the run profile, the capture size and
    the codec of every episode, and it is the only reason the hold-out can be
    stratified at all.

    A malformed line is skipped rather than fatal — the file is appended to
    while the arms are live, so a truncated last line is a crash's fingerprint
    and not a reason to refuse to train on the 44 episodes before it.
    """
    path = Path(root) / "dk1_notes.jsonl"
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def episode_scenes(
    notes: Iterable[dict[str, Any]], episodes: Iterable[int] | None = None
) -> dict[int, int | None]:
    """``{episode: scene}`` from the notes, defaulting to ``None`` for the rest.

    Args:
        notes: what :func:`read_notes` returned.
        episodes: every episode the dataset actually holds. Given, it decides
            the keys — an episode with no note still has to be trained on, and
            an episode noted but discarded must not be. Omitted, the notes decide.

    A note is matched by its ``episode`` field, which
    :meth:`dk1lab.dataset.DatasetSession._write_notes` writes as the dataset's
    own episode index. ``again`` in ``dk1 teleop --record-dataset`` throws an
    episode away *before* it is written, so an index cannot be claimed twice —
    but if one ever were, the last note wins, because that is the one that
    describes the episode on disk.
    """
    from_notes: dict[int, int | None] = {}
    for record in notes:
        index = record.get("episode")
        if isinstance(index, bool) or not isinstance(index, int):
            continue
        scene = record.get("scene")
        from_notes[index] = scene if isinstance(scene, int) and not isinstance(scene, bool) else None
    if episodes is None:
        return from_notes
    return {episode: from_notes.get(episode) for episode in episodes}


# --------------------------------------------------------------------------- #
# The override repair
# --------------------------------------------------------------------------- #


def prune_overrides(
    step_keys: Iterable[str], overrides: dict[str, Any] | None
) -> tuple[dict[str, Any], list[str]]:
    """``overrides`` narrowed to the steps a saved pipeline actually has.

    Returns the kept overrides and the names dropped, so the caller can say what
    it did rather than quietly changing what the training run normalises with.

    ``lerobot_train`` builds its overrides for the pipeline shape *most* policies
    have — a ``normalizer_processor`` and an ``unnormalizer_processor`` — and
    ``PolicyProcessorPipeline`` raises rather than ignores an override that names
    no step. MolmoAct2 normalises through ``molmoact2_masked_normalizer`` keyed by
    its ``norm_tag`` and has neither, so the run dies before the first step.

    Pure: takes the step names, not a checkpoint.
    """
    keys = set(step_keys)
    kept = {name: value for name, value in (overrides or {}).items() if name in keys}
    dropped = [name for name in (overrides or {}) if name not in keys]
    return kept, dropped


def pipeline_steps(path: Path | str, filename: str) -> list[str]:
    """The registry names of one saved pipeline's steps, in order.

    JSON only — the same reading :mod:`dk1lab.checkpoint` does, and for the same
    reason: it costs milliseconds and needs no GPU.
    """
    raw = json.loads((Path(path) / filename).read_text(encoding="utf-8"))
    return [
        step.get("registry_name") or str(step.get("class", "")).rsplit(".", 1)[-1]
        for step in raw.get("steps") or []
    ]


@contextmanager
def patched(checkpoint: Path | str, *, say=None) -> Generator[None, None, None]:
    """Install the override repair around a call to LeRobot's training entry point.

    Wraps ``lerobot.scripts.lerobot_train.make_pre_post_processors`` so that the
    overrides it is handed are narrowed to the steps the checkpoint's saved
    pipelines really have. Everything else about the training run is LeRobot's,
    untouched.

    Restores the original on the way out, whatever happened, so a failed run
    leaves the module as it found it — this process may still have a
    ``dk1 policy check`` to do afterwards.

    Args:
        checkpoint: the directory whose saved pipelines decide what survives.
        say: called with one line per dropped override, or ``None`` to stay
            quiet. What was dropped changes what the run normalises with, so
            somebody has to be told.
    """
    from lerobot.scripts import lerobot_train

    original = lerobot_train.make_pre_post_processors
    pre = pipeline_steps(checkpoint, "policy_preprocessor.json")
    post = pipeline_steps(checkpoint, "policy_postprocessor.json")

    def repaired(*args: Any, **kwargs: Any):
        for name, steps in (
            ("preprocessor_overrides", pre),
            ("postprocessor_overrides", post),
        ):
            if kwargs.get(name):
                kwargs[name], dropped = prune_overrides(steps, kwargs[name])
                if dropped and say is not None:
                    say(
                        f"  dropped {name} {dropped} — the saved pipeline has no "
                        f"such step, and LeRobot raises rather than ignoring it"
                    )
        return original(*args, **kwargs)

    lerobot_train.make_pre_post_processors = repaired
    try:
        yield
    finally:
        lerobot_train.make_pre_post_processors = original


# --------------------------------------------------------------------------- #
# The command line
# --------------------------------------------------------------------------- #


def train_argv(
    *,
    dataset_root: Path | str,
    repo_id: str,
    checkpoint: Path | str,
    output_dir: Path | str,
    job_name: str,
    split: Split,
    budget: TrainingBudget = DEFAULT_BUDGET,
    recipe: LoraRecipe = RECIPE,
    image_keys: Sequence[str] = IMAGE_KEYS,
    family: str = "molmoact2",
    adapt: str = ADAPT_DEFAULT,
    wandb_project: str | None = None,
    extra: Sequence[str] = (),
) -> list[str]:
    """The ``lerobot-train`` arguments this run is, as a list.

    Written out rather than assembled from a config object so that the line
    recorded in the run directory is the line that ran, and so that a reader can
    check any one of these against ``STUDY.md`` without holding a dataclass in
    their head.

    ``--policy.image_keys`` is pinned even though the checkpoint's saved
    ``molmoact2_pack_inputs`` step already carries the order: the hazard
    ``CLAUDE.md`` names is a *training* run rebuilding the processor from a new
    dataset's features, and this is that training run.
    """
    argv = [
        f"--dataset.repo_id={repo_id}",
        f"--dataset.root={Path(dataset_root)}",
        f"--dataset.episodes={_json(list(split.order))}",
        f"--dataset.eval_split={split.eval_split:.6f}",
        f"--policy.path={Path(checkpoint)}",
        "--policy.device=cuda",
        "--policy.push_to_hub=false",
        f"--policy.optimizer_lr={budget.lr:g}",
        f"--policy.scheduler_warmup_steps={budget.warmup}",
        # The preset decays over 100 000 steps, which over a 20 000-step budget
        # is no decay at all. Match the schedule to the budget it runs under.
        f"--policy.scheduler_decay_steps={budget.steps}",
        f"--output_dir={Path(output_dir)}",
        f"--job_name={job_name}",
        f"--steps={budget.steps}",
        f"--batch_size={budget.batch_size}",
        f"--num_workers={budget.num_workers}",
        f"--seed={budget.seed}",
        # Equal on purpose: every checkpoint gets a held-out loss beside it.
        f"--eval_steps={budget.eval_steps}",
        f"--save_freq={budget.save_freq}",
        # Nothing here has a simulator to evaluate in, and STUDY.md's evaluation
        # is nine attempts on the arms.
        "--env_eval_freq=0",
        f"--peft.method_type={_METHOD}",
        f"--peft.r={recipe.r}",
        f"--peft.lora_alpha={recipe.lora_alpha}",
        f"--peft.full_training_modules={_json(list(recipe.modules_to_save))}",
    ]
    targets = target_modules(family, adapt)
    if targets is not None:
        argv.append(f"--peft.target_modules={targets}")
    if family == "molmoact2":
        argv += [
            f"--policy.image_keys={_json(list(image_keys))}",
            # LeRobot's PeftConfig has no dropout field; MolmoAct2 reads its own.
            f"--policy.lora_dropout={recipe.lora_dropout:g}",
            f"--policy.gradient_checkpointing={_bool(budget.gradient_checkpointing)}",
        ]
    argv.append(f"--wandb.enable={_bool(bool(wandb_project))}")
    if wandb_project:
        argv.append(f"--wandb.project={wandb_project}")
    argv += list(extra)
    return argv


_METHOD = "lora"


def _json(value: Any) -> str:
    """Compact JSON, which is what draccus parses a list or dict argument as."""
    return json.dumps(value, separators=(",", ":"))


def _bool(value: bool) -> str:
    return "true" if value else "false"


def run(argv: Sequence[str], *, checkpoint: Path | str, say=None) -> None:
    """Run LeRobot's training entry point with ``argv``, repairs installed.

    Goes through ``lerobot_train.train()`` rather than a config object because
    the entry point is ``@parser.wrap()``-ed and reads ``sys.argv`` — so passing
    the arguments this way is what makes the recorded command line and the
    executed one the same thing.
    """
    from lerobot.scripts import lerobot_train

    previous = sys.argv
    sys.argv = ["lerobot-train", *argv]
    try:
        with patched(checkpoint, say=say):
            lerobot_train.train()
    finally:
        sys.argv = previous


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #


def file_sha256(path: Path | str, *, chunk: int = 8 << 20) -> str:
    """SHA-256 of one file, read in chunks. ``''`` if it is not there.

    The base checkpoint's weights are 10.1 GiB, so this is a few seconds of
    disk. ``STUDY.md`` asks for the checkpoint hash in every run directory, and
    a hash taken once at the start of an overnight run is what says, months
    later, that two rows were fine-tuned from the same weights.
    """
    path = Path(path)
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def git_sha(repo: Path | str = ".") -> dict[str, Any]:
    """The commit this ran at, and whether the tree was dirty.

    A dirty tree is recorded rather than refused: the code that trains is
    allowed to be uncommitted, but a checkpoint whose provenance says
    ``clean: false`` is one whose exact code cannot be recovered, and that has
    to be visible in the file rather than remembered.
    """
    def ask(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", "-C", str(repo), *args],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    sha = ask("rev-parse", "HEAD")
    return {
        "sha": sha,
        "branch": ask("rev-parse", "--abbrev-ref", "HEAD"),
        "clean": bool(sha) and not ask("status", "--porcelain"),
    }


def write_run_dir(
    directory: Path | str,
    *,
    what: Row,
    argv: Sequence[str],
    checkpoint: Path | str,
    config_path: Path | str,
    dataset_root: Path | str,
    split: Split,
    budget: TrainingBudget = DEFAULT_BUDGET,
    recipe: LoraRecipe = RECIPE,
    notes: dict[str, Any] | None = None,
    hash_weights: bool = True,
) -> Path:
    """Write everything ``STUDY.md`` Phase 4 says a training run must record.

    Four things, and each in the form a reader can act on:

    ``dk1_run.json``
        the row, the recipe, the budget, the split, the base checkpoint and its
        hash, the git SHA, and the dataset it trained on.
    ``dk1.toml``
        a **copy** of the file in force, not a path to it. The path would be
        rewritten by the next ``dk1 find cameras``; the copy is what says which
        crop box the demonstrations carry.
    ``command.txt``
        the ``lerobot-train`` line, shell-quoted and runnable.
    ``dk1_command.txt``
        the ``dk1 policy finetune`` line that produced it, so the run can be
        repeated without reconstructing the flags.

    Written **before** training starts. A run directory that only appears when
    training succeeds is missing from exactly the runs worth explaining.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    weights = Path(checkpoint) / "model.safetensors"
    record: dict[str, Any] = {
        "row": what.name,
        "summary": what.summary,
        "profile": what.profile,
        "cropped_at_training_time": what.cropped,
        "invert_gripper_at_rollout": what.invert_gripper,
        "family": what.family,
        "started": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": str(Path(checkpoint)),
        "checkpoint_weights_bytes": weights.stat().st_size if weights.exists() else 0,
        "checkpoint_sha256": file_sha256(weights) if hash_weights else "",
        "dataset": str(Path(dataset_root)),
        "recipe": asdict(recipe) | {"method_type": _METHOD, "scale": recipe.scale},
        "budget": asdict(budget),
        "split": {
            "train": list(split.train),
            "holdout": list(split.holdout),
            "eval_split": split.eval_split,
            "scenes": {str(key): value for key, value in sorted(split.scenes.items())},
        },
        "git": git_sha(Path(config_path).parent),
        "argv": list(argv),
        "python": sys.version.split()[0],
        **(notes or {}),
    }
    (directory / "dk1_run.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    source = Path(config_path)
    if source.is_file():
        (directory / "dk1.toml").write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    (directory / "command.txt").write_text(
        "lerobot-train " + " ".join(shlex.quote(argument) for argument in argv) + "\n",
        encoding="utf-8",
    )
    (directory / "dk1_command.txt").write_text(
        " ".join(shlex.quote(argument) for argument in sys.argv) + "\n", encoding="utf-8"
    )
    return directory


def run_name(what: Row, when: datetime | None = None) -> str:
    """``<row>-<date>-<time>`` — one directory per run, ordered by name."""
    stamp = (when or datetime.now()).strftime("%Y%m%d-%H%M%S")
    return f"{what.name}-{stamp}"


# --------------------------------------------------------------------------- #
# Reading the run back
# --------------------------------------------------------------------------- #


def eval_losses(text: str) -> list[tuple[int, float]]:
    """Every ``step N: eval_loss=X`` in a training log, in the order logged.

    That line is LeRobot's own, written once per ``eval_steps`` on the main
    process. Parsing the log rather than re-running the evaluation is what makes
    :func:`best_checkpoint` free and repeatable.
    """
    found: list[tuple[int, float]] = []
    for line in text.splitlines():
        marker = line.find("step ")
        if marker < 0 or "eval_loss=" not in line:
            continue
        try:
            step = int(line[marker + 5 :].split(":", 1)[0].strip())
            loss = float(line.split("eval_loss=", 1)[1].split()[0])
        except (ValueError, IndexError):
            continue
        found.append((step, loss))
    return found


def best_checkpoint(text: str, *, available: Iterable[int] | None = None) -> tuple[int, float] | None:
    """The step with the lowest held-out loss, or ``None`` if none was measured.

    Args:
        text: the training log.
        available: the steps that actually have a checkpoint on disk. Given, a
            step with the best loss but no checkpoint is skipped rather than
            named — being told the answer is a checkpoint that does not exist is
            worse than being told the second best one that does.

    **This is not an early stop.** The budget was run to the end; this reads back
    which point of it to deploy. ``STUDY.md`` says *early-stop*, and the
    difference is worth an amendment rather than a silent reinterpretation.
    """
    losses = eval_losses(text)
    if available is not None:
        keep = set(available)
        losses = [(step, loss) for step, loss in losses if step in keep]
    if not losses:
        return None
    return min(losses, key=lambda item: (item[1], item[0]))


def checkpoint_steps(run_dir: Path | str) -> list[int]:
    """Every step that has a checkpoint under ``<run dir>/train/checkpoints``.

    LeRobot names them by zero-padded step, plus a ``last`` symlink which is
    skipped — it is a pointer, not a step.
    """
    root = train_dir(run_dir) / "checkpoints"
    if not root.is_dir():
        return []
    steps = []
    for entry in root.iterdir():
        if entry.is_symlink() or not entry.is_dir():
            continue
        try:
            steps.append(int(entry.name))
        except ValueError:
            continue
    return sorted(steps)


#: What LeRobot calls the policy directory inside a checkpoint. Restated rather
#: than imported so this module still costs no torch import; asserted against
#: ``lerobot.utils.constants.PRETRAINED_MODEL_DIR`` in ``tests/test_finetune.py``.
PRETRAINED_MODEL_DIR = "pretrained_model"


def deployable(run_dir: Path | str, step: int, total_steps: int) -> Path:
    """The directory to point ``--checkpoint`` at for one saved step.

    LeRobot writes the policy under ``checkpoints/<step>/pretrained_model``; the
    level above holds the optimiser state too and is not something a rollout can
    load. The step is zero-padded to at least six digits, and to the width of
    ``total_steps`` when that is wider — which is why the budget has to be passed
    in rather than guessed from the step.
    """
    width = max(6, len(str(total_steps)))
    return train_dir(run_dir) / "checkpoints" / f"{step:0{width}d}" / PRETRAINED_MODEL_DIR
