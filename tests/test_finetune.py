"""The LoRA fine-tune: the recipe, the hold-out, the repair, the provenance.

Four properties are worth holding here, and each of them would cost a night of
GPU time or a whole row of the study if it slipped:

* the **hold-out spans every scene**. The demonstrations are recorded grouped by
  layout, so LeRobot's own "last N episodes" split validates on scene 3 alone;
* the **override repair drops exactly the keys the saved pipeline lacks**, which
  is what stops ``lerobot-train`` dying before its first step on a checkpoint
  that normalises through its own step;
* the **episode order handed to LeRobot puts the hold-out last**, and the
  ``eval_split`` fraction rounds to exactly its size — that is the whole reason
  no patch is needed for the split;
* the **run directory is written before training**, and carries what ``STUDY.md``
  asks for.

Nothing here needs a robot, a GPU or LeRobot.
"""

from __future__ import annotations

import json
import math

import pytest

from dk1lab import finetune
from dk1lab.layout import IMAGE_KEYS


def scenes(counts: dict[int, int]) -> dict[int, int]:
    """``{episode: scene}`` recorded the way a session records it: grouped."""
    mapping: dict[int, int] = {}
    index = 0
    for scene, count in counts.items():
        for _ in range(count):
            mapping[index] = scene
            index += 1
    return mapping


# --------------------------------------------------------------------------- #
# The recipe
# --------------------------------------------------------------------------- #


def test_the_recipe_is_the_one_the_protocol_fixed():
    """r=32, alpha=16, dropout=0.05, nothing fully trained. Both models, both phases."""
    assert (finetune.RECIPE.r, finetune.RECIPE.lora_alpha) == (32, 16)
    assert finetune.RECIPE.lora_dropout == 0.05
    assert finetune.RECIPE.modules_to_save == ()
    assert finetune.RECIPE.scale == 0.5


def test_the_budget_is_steps_and_not_epochs():
    """Epochs would tie the two models' training to how long each demo happened to take."""
    assert finetune.DEFAULT_BUDGET.steps > 0
    assert not hasattr(finetune.DEFAULT_BUDGET, "epochs")


def test_a_checkpoint_exists_at_every_measured_loss():
    """Otherwise the best held-out loss belongs to a checkpoint nobody saved."""
    assert finetune.DEFAULT_BUDGET.eval_steps == finetune.DEFAULT_BUDGET.save_freq


def test_gradient_checkpointing_is_on():
    """5.44 B parameters on a 32 GB card; off, the run OOMs a few hundred steps in."""
    assert finetune.DEFAULT_BUDGET.gradient_checkpointing is True


# --------------------------------------------------------------------------- #
# The rows
# --------------------------------------------------------------------------- #


def test_only_R1_needs_the_crop_applied_at_training_time():
    """It is the one row that rolls out under `optimized`."""
    assert finetune.row("R1").cropped is True
    assert finetune.row("A1").cropped is False
    assert finetune.row("B1").cropped is False


def test_the_gripper_inversion_is_off_for_every_fine_tuned_row():
    """The demonstrations are in DK1 convention, so the weights end up speaking DK1.

    STUDY.md § *The gripper convention*: leaving the inversion on flips every
    grasp, and it does so silently — the arms move, they just never hold anything.
    """
    assert all(not what.invert_gripper for what in finetune.ROWS.values())


def test_a_zero_shot_row_is_refused_rather_than_defaulted():
    """A0 has no training run, and training something called A0 would be evidence
    about a configuration that does not exist."""
    for name in ("A0", "R0", "B0", "A2"):
        with pytest.raises(finetune.FinetuneError) as excinfo:
            finetune.row(name)
        assert "R1" in str(excinfo.value)


def test_a_row_can_be_named_in_lower_case():
    assert finetune.row("r1") is finetune.row("R1")


# --------------------------------------------------------------------------- #
# The hold-out
# --------------------------------------------------------------------------- #


def test_the_holdout_spans_every_scene():
    """The property this function exists for.

    Recorded grouped, the last ten episodes of 45 are all scene 3. A validation
    set that is one layout measures one layout, and the loss curve it draws says
    nothing about whether the adapter helped the other two.
    """
    split = finetune.split_episodes(scenes({1: 15, 2: 15, 3: 15}), holdout=10)
    held_scenes = {split.scenes[episode] for episode in split.holdout}
    assert held_scenes == {1, 2, 3}


def test_the_holdout_is_proportional_to_each_scene():
    """Ten from three equal blocks of fifteen: 4/3/3, and exactly ten in total."""
    split = finetune.split_episodes(scenes({1: 15, 2: 15, 3: 15}), holdout=10)
    counts = sorted(
        sum(1 for episode in split.holdout if split.scenes[episode] == scene)
        for scene in (1, 2, 3)
    )
    assert counts == [3, 3, 4]
    assert len(split.holdout) == 10


def test_the_holdout_is_not_the_tail_of_each_scene():
    """A teleoperation session drifts: the last demonstrations of a block are the
    steadiest, and holding out only those makes validation easier than training."""
    split = finetune.split_episodes(scenes({1: 15, 2: 15, 3: 15}), holdout=9)
    first_block = [episode for episode in split.holdout if episode < 15]
    assert min(first_block) == 0
    assert max(first_block) == 14


def test_train_and_holdout_partition_the_dataset():
    split = finetune.split_episodes(scenes({1: 15, 2: 15, 3: 15}), holdout=10)
    assert set(split.train) | set(split.holdout) == set(range(45))
    assert not set(split.train) & set(split.holdout)
    assert len(split.train) == 35


def test_the_order_handed_to_lerobot_ends_with_the_holdout():
    """This is what makes the split need no patch.

    ``make_train_eval_datasets`` takes the LAST ceil(n * eval_split) episodes of
    whatever order it is given, and ``LeRobotDataset`` stores ``episodes``
    verbatim. So the order is the instruction.
    """
    split = finetune.split_episodes(scenes({1: 15, 2: 15, 3: 15}), holdout=10)
    assert list(split.order) == list(split.train) + list(split.holdout)
    assert split.order[-10:] == split.holdout


@pytest.mark.parametrize("total,holdout", [(45, 10), (45, 9), (30, 10), (12, 3), (100, 10)])
def test_the_eval_split_fraction_rounds_to_exactly_the_holdout(total, holdout):
    """LeRobot takes ceil(n * f). One episode either way is a different experiment,
    and the failure is silent: the run trains and validates on the wrong sets."""
    fraction = finetune._eval_split(total, holdout)
    assert math.ceil(total * fraction) == holdout


def test_a_holdout_that_leaves_nothing_to_train_on_is_refused():
    with pytest.raises(finetune.FinetuneError):
        finetune.split_episodes(scenes({1: 5}), holdout=5)
    with pytest.raises(finetune.FinetuneError):
        finetune.split_episodes(scenes({1: 5}), holdout=0)


def test_the_split_is_the_same_every_time():
    """No seed, so two runs of the same command are comparable."""
    mapping = scenes({1: 15, 2: 15, 3: 15})
    assert finetune.split_episodes(mapping, 10) == finetune.split_episodes(mapping, 10)


def test_unlabelled_episodes_are_still_split():
    """A dataset with no notes must still be trainable; it just cannot be stratified."""
    split = finetune.split_episodes({index: None for index in range(20)}, holdout=4)
    assert len(split.holdout) == 4
    assert len(split.train) == 16


def test_the_split_describes_which_scenes_it_held_out():
    split = finetune.split_episodes(scenes({1: 15, 2: 15, 3: 15}), holdout=10)
    line = split.describe()
    assert "35 train, 10 held out" in line
    assert "scene 1" in line and "scene 3" in line


# --------------------------------------------------------------------------- #
# Reading the demonstrations' notes
# --------------------------------------------------------------------------- #


def test_notes_are_read_back_in_order(tmp_path):
    path = tmp_path / "dk1_notes.jsonl"
    path.write_text(
        "\n".join(json.dumps({"episode": index, "scene": 1 + index // 2}) for index in range(4))
        + "\n"
    )
    assert [record["episode"] for record in finetune.read_notes(tmp_path)] == [0, 1, 2, 3]


def test_a_truncated_last_line_does_not_lose_the_episodes_before_it():
    """The file is appended to while the arms are live, so a half-written last
    line is a crash's fingerprint — not a reason to refuse 44 good episodes."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "dk1_notes.jsonl").write_text(
            json.dumps({"episode": 0, "scene": 1}) + "\n" + '{"episode": 1, "sce'
        )
        assert len(finetune.read_notes(root)) == 1


def test_a_dataset_with_no_notes_reads_back_empty(tmp_path):
    assert finetune.read_notes(tmp_path) == []


def test_episodes_without_a_note_are_still_keys():
    """An episode on disk has to be trained on whether or not it was labelled."""
    notes = [{"episode": 0, "scene": 1}, {"episode": 2, "scene": 2}]
    assert finetune.episode_scenes(notes, range(4)) == {0: 1, 1: None, 2: 2, 3: None}


def test_a_note_without_a_scene_maps_to_none():
    assert finetune.episode_scenes([{"episode": 0}]) == {0: None}


# --------------------------------------------------------------------------- #
# The override repair
# --------------------------------------------------------------------------- #


def test_an_override_naming_a_missing_step_is_dropped():
    """MolmoAct2 normalises through `molmoact2_masked_normalizer` and has no
    `normalizer_processor`. LeRobot raises rather than ignoring the override, so
    the training run dies before its first step. Verified on the real checkpoint."""
    kept, dropped = finetune.prune_overrides(
        ["rename_observations_processor", "molmoact2_masked_normalizer", "device_processor"],
        {
            "device_processor": {"device": "cuda"},
            "normalizer_processor": {"features": {}},
            "rename_observations_processor": {"rename_map": {}},
        },
    )
    assert dropped == ["normalizer_processor"]
    assert set(kept) == {"device_processor", "rename_observations_processor"}


def test_a_pipeline_that_has_every_step_keeps_every_override():
    """The repair has to be a no-op for a policy whose pipeline does fit —
    otherwise it would quietly stop normalising pi0.5, which needs it."""
    overrides = {"normalizer_processor": {"stats": {}}, "device_processor": {"device": "cuda"}}
    kept, dropped = finetune.prune_overrides(
        ["normalizer_processor", "device_processor"], overrides
    )
    assert kept == overrides
    assert dropped == []


def test_pruning_no_overrides_is_harmless():
    assert finetune.prune_overrides(["a"], None) == ({}, [])


def test_the_saved_pipeline_steps_are_read_from_json(checkpoint_dir):
    assert finetune.pipeline_steps(checkpoint_dir, "policy_preprocessor.json") == [
        "molmoact2_state_frame_transform",
        "molmoact2_pack_inputs",
    ]


# --------------------------------------------------------------------------- #
# The command line
# --------------------------------------------------------------------------- #


@pytest.fixture
def argv(tmp_path):
    split = finetune.split_episodes(scenes({1: 15, 2: 15, 3: 15}), holdout=10)
    return finetune.train_argv(
        dataset_root=tmp_path / "demos",
        repo_id="dk1/demos",
        checkpoint=tmp_path / "ckpt",
        output_dir=tmp_path / "run",
        job_name="dk1-r1",
        split=split,
    )


def test_the_image_keys_are_pinned(argv):
    """CLAUDE.md: the hazard is a training run rebuilding the processor from a new
    dataset's features, and this is that training run."""
    line = next(item for item in argv if item.startswith("--policy.image_keys="))
    assert json.loads(line.split("=", 1)[1]) == list(IMAGE_KEYS)


def test_the_peft_arguments_carry_the_recipe(argv):
    assert "--peft.method_type=lora" in argv
    assert "--peft.r=32" in argv
    assert "--peft.lora_alpha=16" in argv
    assert "--peft.full_training_modules=[]" in argv


def test_the_dropout_goes_through_the_policy_config(argv):
    """LeRobot's PeftConfig has no dropout field; MolmoAct2's policy config does."""
    assert "--policy.lora_dropout=0.05" in argv
    assert not any(item.startswith("--peft.lora_dropout") for item in argv)


def test_the_schedule_decays_over_the_budget_it_runs_under(argv):
    """The preset decays over 100 000 steps, which over 20 000 is no decay at all."""
    steps = int(next(item for item in argv if item.startswith("--steps=")).split("=")[1])
    decay = int(
        next(item for item in argv if item.startswith("--policy.scheduler_decay_steps="))
        .split("=")[1]
    )
    assert decay == steps


def test_the_episode_order_and_the_split_fraction_travel_together(argv):
    order = json.loads(
        next(item for item in argv if item.startswith("--dataset.episodes=")).split("=", 1)[1]
    )
    fraction = float(
        next(item for item in argv if item.startswith("--dataset.eval_split=")).split("=", 1)[1]
    )
    assert math.ceil(len(order) * fraction) == 10


def test_nothing_is_pushed_to_the_hub_by_accident(argv):
    assert "--policy.push_to_hub=false" in argv


def test_pi05_gets_no_molmoact2_only_flags(tmp_path):
    """`--policy.lora_dropout` and `--policy.image_keys` are MolmoAct2's fields;
    draccus rejects an unknown one, so passing them to pi0.5 fails the run."""
    split = finetune.split_episodes(scenes({1: 4}), holdout=1)
    argv = finetune.train_argv(
        dataset_root=tmp_path,
        repo_id="dk1/demos",
        checkpoint=tmp_path,
        output_dir=tmp_path / "run",
        job_name="dk1-b1",
        split=split,
        family="pi05",
    )
    assert not any(item.startswith("--policy.image_keys") for item in argv)
    assert not any(item.startswith("--policy.lora_dropout") for item in argv)
    assert "--peft.r=32" in argv


# --------------------------------------------------------------------------- #
# The run directory
# --------------------------------------------------------------------------- #


def test_the_run_directory_carries_what_the_protocol_asks_for(tmp_path, checkpoint_dir, config_file):
    """The checkpoint hash, the dk1.toml in force, the command line, the git SHA."""
    split = finetune.split_episodes(scenes({1: 6}), holdout=2)
    argv = ["--steps=10"]
    directory = finetune.write_run_dir(
        tmp_path / "run",
        what=finetune.row("R1"),
        argv=argv,
        checkpoint=checkpoint_dir,
        config_path=config_file,
        dataset_root=tmp_path / "demos",
        split=split,
    )
    record = json.loads((directory / "dk1_run.json").read_text())
    assert record["row"] == "R1"
    assert record["checkpoint"] == str(checkpoint_dir)
    assert "checkpoint_sha256" in record
    assert record["recipe"]["r"] == 32
    assert record["split"]["holdout"] == list(split.holdout)
    assert set(record["git"]) == {"sha", "branch", "clean"}
    assert (directory / "dk1.toml").read_text() == config_file.read_text()
    assert (directory / "command.txt").read_text().startswith("lerobot-train --steps=10")


def test_the_dk1_toml_is_copied_and_not_referenced(tmp_path, checkpoint_dir, config_file):
    """A path would be rewritten by the next `dk1 find cameras`; the copy is what
    says which crop box the demonstrations carry."""
    directory = finetune.write_run_dir(
        tmp_path / "run",
        what=finetune.row("A1"),
        argv=[],
        checkpoint=checkpoint_dir,
        config_path=config_file,
        dataset_root=tmp_path,
        split=finetune.split_episodes(scenes({1: 4}), holdout=1),
        hash_weights=False,
    )
    config_file.write_text("version = 1\n")
    assert "cameras.left" in (directory / "dk1.toml").read_text()


def test_the_run_directory_records_that_the_inversion_goes_off(
    tmp_path, checkpoint_dir, config_file
):
    directory = finetune.write_run_dir(
        tmp_path / "run",
        what=finetune.row("R1"),
        argv=[],
        checkpoint=checkpoint_dir,
        config_path=config_file,
        dataset_root=tmp_path,
        split=finetune.split_episodes(scenes({1: 4}), holdout=1),
        hash_weights=False,
    )
    record = json.loads((directory / "dk1_run.json").read_text())
    assert record["invert_gripper_at_rollout"] is False
    assert record["cropped_at_training_time"] is True


def test_a_missing_weights_file_hashes_to_nothing_rather_than_raising(tmp_path):
    assert finetune.file_sha256(tmp_path / "nope.safetensors") == ""


def test_the_hash_is_of_the_bytes(tmp_path):
    import hashlib

    path = tmp_path / "model.safetensors"
    path.write_bytes(b"weights")
    assert finetune.file_sha256(path) == hashlib.sha256(b"weights").hexdigest()


# --------------------------------------------------------------------------- #
# Reading the run back
# --------------------------------------------------------------------------- #

LOG = """\
12:00:00 INFO  step 1000: eval_loss=0.4211
12:10:00 INFO  step 2000: eval_loss=0.3120
12:20:00 INFO  step 3000: eval_loss=0.3400
12:30:00 INFO  step 4000: eval_loss=0.3901
"""


def test_the_eval_losses_are_read_off_lerobots_own_line():
    assert finetune.eval_losses(LOG) == [
        (1000, 0.4211),
        (2000, 0.3120),
        (3000, 0.3400),
        (4000, 0.3901),
    ]


def test_the_best_checkpoint_is_the_lowest_held_out_loss():
    assert finetune.best_checkpoint(LOG) == (2000, 0.3120)


def test_a_best_loss_with_no_checkpoint_on_disk_is_not_offered():
    """Being told the answer is a checkpoint that does not exist is worse than
    being told the second best one that does."""
    assert finetune.best_checkpoint(LOG, available=[1000, 3000]) == (3000, 0.3400)


def test_a_log_with_no_evaluation_says_so_rather_than_guessing():
    assert finetune.best_checkpoint("step 100: loss=0.5\n") is None


def test_the_checkpoints_on_disk_are_listed_by_step(tmp_path):
    root = tmp_path / "train" / "checkpoints"
    for name in ("001000", "002000", "last"):
        (root / name).mkdir(parents=True)
    assert finetune.checkpoint_steps(tmp_path) == [1000, 2000]


def test_the_deployable_path_is_the_pretrained_model_directory(tmp_path):
    """The level above holds the optimiser state too and no rollout can load it."""
    path = finetune.deployable(tmp_path, 2000, 20_000)
    assert path.name == "pretrained_model"
    assert path.parent.name == "002000"


def test_the_pretrained_model_directory_name_still_matches_lerobots():
    """Restated in `finetune` so the module costs no torch import; if LeRobot ever
    renames it, this is where that is found out."""
    from lerobot.utils.constants import PRETRAINED_MODEL_DIR

    assert finetune.PRETRAINED_MODEL_DIR == PRETRAINED_MODEL_DIR


def test_the_step_is_padded_to_the_width_of_the_budget():
    """LeRobot pads to max(6, len(str(total_steps))), which is why the budget has
    to be passed in rather than guessed from the step."""
    assert finetune.deployable(".", 500, 1_000_000).parent.name == "0000500"


# --------------------------------------------------------------------------- #
# What the adapter attaches to
# --------------------------------------------------------------------------- #


def test_the_molmoact2_regex_is_lerobots_own():
    """Transcribed here because it is an instance method and the command line has
    to be built before the weights are on the GPU. Pinned, not trusted: if
    LeRobot changes what MolmoAct2 adapts, this is where that is found out."""
    from types import SimpleNamespace

    from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy

    for action_expert in (False, True):
        stub = SimpleNamespace(config=SimpleNamespace(enable_lora_action_expert=action_expert))
        assert MolmoAct2Policy._lora_target_modules(
            stub, prefix=r"model\.model"
        ) == finetune.molmoact2_target_modules(action_expert=action_expert)


def test_the_default_leaves_each_policys_own_choice_alone():
    """Which is what STUDY.md prescribes literally, and is not the same recipe in
    shape for the two models — see the module's own note."""
    assert finetune.target_modules("molmoact2", finetune.ADAPT_DEFAULT) is None
    assert finetune.target_modules("pi05", finetune.ADAPT_DEFAULT) is None


def test_vlm_plus_expert_reaches_the_action_expert():
    regex = finetune.target_modules("molmoact2", finetune.ADAPT_VLM_AND_EXPERT)
    assert "action_expert" in regex
    assert "vision_backbone" in regex


def test_vlm_plus_expert_is_refused_for_pi05():
    """Its default already is the action expert; MolmoAct2's regex would match
    nothing in it, and a LoRA that adapts nothing trains nothing."""
    with pytest.raises(finetune.FinetuneError):
        finetune.target_modules("pi05", finetune.ADAPT_VLM_AND_EXPERT)


def test_a_misspelled_adapt_is_refused():
    with pytest.raises(finetune.FinetuneError):
        finetune.target_modules("molmoact2", "expert")


def test_the_default_passes_no_target_modules_argument(argv):
    assert not any(item.startswith("--peft.target_modules") for item in argv)


def test_vlm_plus_expert_appears_on_the_command_line(tmp_path):
    argv = finetune.train_argv(
        dataset_root=tmp_path,
        repo_id="dk1/demos",
        checkpoint=tmp_path,
        output_dir=tmp_path / "run",
        job_name="dk1-r1",
        split=finetune.split_episodes(scenes({1: 4}), holdout=1),
        adapt=finetune.ADAPT_VLM_AND_EXPERT,
    )
    line = next(item for item in argv if item.startswith("--peft.target_modules="))
    assert "action_expert" in line


def test_what_is_adapted_is_described_in_plain_words():
    """It goes in the banner and in the run directory: 'the action expert stays
    frozen' is not something to discover from a loss curve."""
    assert "FROZEN" in finetune.describe_adapt("molmoact2", finetune.ADAPT_DEFAULT)
    assert "action expert" in finetune.describe_adapt("pi05", finetune.ADAPT_DEFAULT)
    assert "action expert" in finetune.describe_adapt(
        "molmoact2", finetune.ADAPT_VLM_AND_EXPERT
    )
