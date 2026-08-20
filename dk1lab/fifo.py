"""Serve a whole action chunk from a queue, instead of rebuilding it every tick.

The problem this exists for, measured on this cell with the real checkpoint and
the real cameras, paced at 30 Hz:

============================================  =========
``MolmoAct2PackInputsProcessorStep``           15.9 ms
the rest of the preprocessor                    0.8 ms
``select_action`` and the engine's own work     5.5 ms
the postprocessor                               0.2 ms
**per control tick, of a 33.3 ms budget**    **~22 ms**
============================================  =========

and on **29 of every 30 ticks every millisecond of it is discarded**.
:meth:`MolmoAct2Policy.select_action` looks at its own 30-deep queue first,
finds it non-empty, and returns ``popleft()`` without reading the batch at all.
So three camera views are resized to 378x378, patchified and normalised thirty
times per chunk, and twenty-nine of those are thrown away. That is what made the
rollout loop run at 27.7 Hz against a 30 Hz target.

:class:`~lerobot.rollout.inference.sync.SyncInferenceEngine` is built that way
because it drives every policy through ``select_action``, and LeRobot's own
comment block above the class says what the fix is and why upstream has not
taken it: SAC raises from ``predict_action_chunk``, ACT's temporal ensembler
lives inside ``select_action``, and the Diffusion family fills its
observation-history queues as a side effect of ``select_action``. **MolmoAct2
has none of the three** — its ``select_action`` *is* a ``predict_action_chunk``
call plus a ``deque`` — so the fix is available to us and not to upstream.

So: run the model once, postprocess the whole chunk at once, and hand out the
rows. The preprocessor runs once per chunk instead of thirty times.

Why this is the same behaviour, not merely a faster one:

- **The rows are identical.** ``select_action`` slices the same
  ``predict_action_chunk`` output to ``n_action_steps`` and pops from it in
  order. This does the same slice and the same order.
- **Postprocessing a chunk equals postprocessing its rows.** All four steps —
  clamp, unnormalise, action-frame transform, device move — are stateless and
  elementwise over the action dimension. RTC already postprocesses whole chunks
  this way, on the same pipeline object.
- **The discarded observations were discarded anyway.** The only thing lost is
  work whose result was never read.

The one policy property this *does* depend on is that actions are **absolute**.
A relative-action policy re-anchors to the current state on every call, so
serving a precomputed chunk would drift — which is upstream's fourth caveat, and
also an argument for doing this rather than against it. It is checked at
construction rather than assumed, and this checkpoint is absolute joint pose.

**This does not change the pause.** Once per chunk the loop still blocks for one
model call (~190 ms here) while the arms hold their last commanded target. That
is inherent to synchronous inference and is what ``--rtc`` exists to remove;
it is unchanged, deliberately.
"""

from __future__ import annotations

import logging
from collections import deque
from contextlib import nullcontext
from copy import copy
from typing import Any

logger = logging.getLogger(__name__)

#: Attributes copied off the engine being replaced. They are private on
#: ``SyncInferenceEngine`` and there is no public accessor; naming them in one
#: place makes the coupling to an upstream implementation detail explicit rather
#: than scattering ``_engine._task`` through the code.
_CARRIED = (
    "_policy",
    "_preprocessor",
    "_postprocessor",
    "_dataset_features",
    "_ordered_action_keys",
    "_task",
    "_device",
    "_robot_type",
)


def _relative_action_steps(preprocessor: Any) -> list[str]:
    """Any pipeline step that makes actions relative to the current state.

    Matched by class name because the classes live in different modules across
    LeRobot versions and importing them all to ``isinstance`` against would make
    this file fail to import rather than fail to find one.
    """
    steps = getattr(preprocessor, "steps", ()) or ()
    return [type(s).__name__ for s in steps if "Relative" in type(s).__name__]


def build(engine: Any) -> Any:
    """A :class:`ChunkFIFOInferenceEngine` standing in for ``engine``.

    Takes the already-built sync engine rather than the pieces, so there is no
    second place that has to know how a policy, its pipelines and this cell's
    action key order are wired together. Everything is carried across by
    reference — the same policy object, the same two pipeline objects — so a
    gripper inversion already applied to them is still applied, and a trace
    attached afterwards still wraps the things that run.
    """
    from lerobot.rollout.inference.sync import SyncInferenceEngine

    if not isinstance(engine, SyncInferenceEngine):
        raise TypeError(
            f"the chunk FIFO replaces a SyncInferenceEngine, not a {type(engine).__name__}. "
            "Under --rtc the chunk is already served whole and there is nothing to fix."
        )
    return ChunkFIFOInferenceEngine(**{name.lstrip("_"): getattr(engine, name) for name in _CARRIED})


class ChunkFIFOInferenceEngine:
    """Synchronous inference that runs the policy once per chunk, not once per tick.

    A drop-in for ``SyncInferenceEngine``: same ``InferenceEngine`` lifecycle,
    same ``get_action`` contract, same tensor out. It is deliberately **not** a
    subclass — the only method it would inherit is the one it replaces, and
    inheriting the rest would suggest the two share a per-tick path that they do
    not.

    Args:
        policy: the loaded policy. Must produce absolute actions.
        preprocessor, postprocessor: the pipelines the sync engine held, by
            reference, so anything patched onto them still applies.
        dataset_features: what ``make_robot_action`` names the action slots from.
        ordered_action_keys: this cell's key order, from ``dk1lab.layout``.
        task: the instruction handed to the policy.
        device: where the policy lives.
        robot_type: passed through to ``prepare_observation_for_inference``.
    """

    def __init__(
        self,
        *,
        policy: Any,
        preprocessor: Any,
        postprocessor: Any,
        dataset_features: dict,
        ordered_action_keys: list[str],
        task: str,
        device: Any,
        robot_type: str,
    ) -> None:
        import torch

        relative = _relative_action_steps(preprocessor)
        if relative:
            raise ValueError(
                f"the chunk FIFO serves precomputed actions, so it is only correct for a policy "
                f"with absolute actions; this pipeline has {', '.join(relative)}. Use the plain "
                f"sync engine, which re-anchors every tick."
            )
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._dataset_features = dataset_features
        self._ordered_action_keys = ordered_action_keys
        self._task = task
        self._device = device if isinstance(device, torch.device) else torch.device(device or "cpu")
        self._robot_type = robot_type
        self._fifo: deque = deque()

    # -- lifecycle ---------------------------------------------------------- #

    def start(self) -> None:
        """No background resources; put the policy in eval mode once, here.

        ``select_action`` calls ``self.eval()`` on every call, which walks 1737
        submodules of a 7B model for 1.8 ms to set a flag that is already set.
        Doing it once at the start of the run is the whole of what that achieved.
        """
        self._policy.eval()
        logger.info(
            "ChunkFIFOInferenceEngine started — one model call per %d-step chunk, "
            "not one pipeline pass per tick",
            self._chunk_steps(),
        )

    def stop(self) -> None:
        """No background resources to stop."""
        logger.info("ChunkFIFOInferenceEngine stopped")

    def reset(self) -> None:
        """Drop the queued chunk and reset the policy and both pipelines.

        The local queue has to go with the policy's own: a reset means the
        episode boundary moved, and actions computed for the old one describe a
        situation that no longer exists.
        """
        self._fifo.clear()
        self._policy.reset()
        self._preprocessor.reset()
        self._postprocessor.reset()
        self._policy.eval()
        logger.info("Resetting chunk FIFO inference state (policy + processors + queue)")

    # -- the InferenceEngine optional hooks --------------------------------- #

    def notify_observation(self, observation: dict) -> None:
        """No-op: this engine is driven by ``get_action``, like the sync one."""

    def pause(self) -> None:
        """No-op."""

    def resume(self) -> None:
        """No-op."""

    # -- actions ------------------------------------------------------------ #

    def _chunk_steps(self) -> int:
        """How many rows of a chunk to serve — the policy's own ``n_action_steps``."""
        return int(getattr(self._policy.config, "n_action_steps", 0) or 0)

    def get_action(self, obs_frame: dict | None) -> Any:
        """The next action: from the queue, or from one model call that fills it.

        Returns ``None`` only when there is nothing queued *and* no observation
        to compute from, which is the same condition the sync engine returns
        ``None`` under.
        """
        if self._fifo:
            return self._fifo.popleft()
        if obs_frame is None:
            return None
        self._fill(obs_frame)
        return self._fifo.popleft() if self._fifo else None

    def _fill(self, obs_frame: dict) -> None:
        """Run the model once and queue every row of the chunk it returned."""
        import torch
        from lerobot.policies.utils import make_robot_action, prepare_observation_for_inference

        if getattr(self._policy, "_rtc_enabled", lambda: False)():
            raise RuntimeError(
                "the policy has RTC enabled, so predict_action_chunk expects RTC's delay and "
                "previous-chunk arguments. Run the RTC engine, or build the FIFO on a "
                "policy configured for sync inference."
            )

        # A shallow copy for the same reason the sync engine takes one: the
        # caller rebuilds obs_frame per tick, so nothing else reads these values.
        observation = copy(obs_frame)
        autocast_ctx = (
            torch.autocast(device_type=self._device.type)
            if self._device.type == "cuda" and self._policy.config.use_amp
            else nullcontext()
        )
        with torch.inference_mode(), autocast_ctx:
            observation = prepare_observation_for_inference(
                observation, self._device, self._task, self._robot_type
            )
            observation = self._preprocessor(observation)
            chunk = self._policy.predict_action_chunk(observation)
            steps = self._chunk_steps()
            if steps:
                chunk = chunk[:, :steps]
            processed = self._postprocessor(chunk)

        # ``[1, steps, dim]`` -> one row per tick. ``.cpu()`` once for the whole
        # chunk rather than once per tick: each call synchronises the device.
        rows = processed.squeeze(0).cpu()
        for row in rows:
            action_dict = make_robot_action(row, self._dataset_features)
            self._fifo.append(
                torch.tensor([action_dict[key] for key in self._ordered_action_keys])
            )

    # -- reporting ---------------------------------------------------------- #

    @property
    def queued(self) -> int:
        """Rows left to serve before the next model call. For a banner or a test."""
        return len(self._fifo)
