"""The model's-eye view of a live observation, for watching during teleoperation.

``dk1 teleop --display`` already streams the three cameras to Rerun — that is
LeRobot's own ``log_rerun_data``, and it shows the robot-side view: the frames
the cell produces, crop included. This module adds the *other* picture, the one
the policy would actually be handed, so both can be watched side by side while a
human drives the arms.

**Why that is a different picture.** Between the camera and the model sit a
rename, a fixed key order, a channel-layout change and a resize to a square
378x378. None of it is visible in the robot-side view, and a mistake in any of it
looks exactly like a correct rollout right up until the policy misbehaves. The
only way to see it is to run the real preprocessor and look at what comes out —
which is what this does, using the same checkpoint pipelines a rollout builds.

**It needs no model and no GPU.** ``make_pre_post_processors`` reads the saved
``policy_preprocessor.json`` and the HF image processor; the 11 GB of weights are
never touched. Building one costs ~0.6 s on the CPU.

**It never costs the control loop anything.** One pass through the preprocessor
costs ~11 ms and the teleop loop's whole budget at 60 Hz is 16.7 ms, so doing it
inline would drop ticks every time it ran. Instead the probe hands the
observation to a background thread and returns immediately — the same shape as
``OpenCVCamera``'s own read thread — and the loop pays a dict copy. The thread
samples one observation in :data:`DEFAULT_EVERY` and **drops** anything that
arrives while it is busy, because this is a picture for a human to look at and a
stale frame is worth less than a fast loop.

**It attaches by wrapping, never by replacing** — the same rule
:mod:`dk1lab.trace` follows. :class:`ModelInputProbe` proxies the observation
processor LeRobot's ``teleop_loop`` already calls, passes its result through
untouched, and logs as a side effect. The control loop itself stays upstream's.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from .layout import CAMERA_NAMES, IMAGE_KEYS
from .trace import model_input_images

logger = logging.getLogger(__name__)

#: Sample one tick in this many. 12 is 5 Hz at the 60 Hz teleop default — fast
#: enough to see a wrist camera swing as you move the arm, slow enough that the
#: worker thread is idle most of the time rather than pegging a core.
DEFAULT_EVERY = 12

#: What the probe calls the task when logging. The preprocessor tokenises an
#: instruction whether or not one matters, and the images do not depend on it.
DEFAULT_TASK = "teleoperation"


def build_preprocessor(checkpoint: str, *, width: int, height: int) -> tuple[Any, Any]:
    """The checkpoint's real input pipeline, and the feature dict that feeds it.

    On the CPU and in float32: nothing here runs a model, and putting the
    display on the GPU would make watching a teleop session compete with
    whatever else is using the card.
    """
    from lerobot.policies.factory import make_pre_post_processors

    from .policy import dataset_features, policy_config

    config = policy_config(checkpoint, device="cpu", dtype="float32", invert_gripper=False)
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=config.pretrained_path,
        preprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    return preprocessor, dataset_features(width=width, height=height)


class ModelInputProbe:
    """An observation processor that also logs the model's-eye view to Rerun.

    Call :meth:`start` before the loop and :meth:`stop` after it. Between them
    every call passes the observation straight through and, one call in
    ``every``, hands a copy to the worker.

    Args:
        inner: the processor to wrap — whatever ``make_default_processors``
            returned. Its result is passed through untouched.
        preprocessor: the policy input pipeline, from :func:`build_preprocessor`.
        features: the matching feature dict.
        task: the instruction to tokenise. Does not affect the images.
        every: sample one observation in this many.
    """

    def __init__(
        self,
        inner: Any,
        preprocessor: Any,
        features: Any,
        *,
        task: str = DEFAULT_TASK,
        every: int = DEFAULT_EVERY,
    ) -> None:
        self.inner = inner
        self.preprocessor = preprocessor
        self.features = features
        self.task = task
        self.every = max(1, int(every))
        self.ticks = 0
        self.logged = 0
        self.dropped = 0
        #: Set when a pass raises, which switches the probe off for the rest of
        #: the run. A display that fails once will fail every tick, and a warning
        #: per tick at 60 Hz would bury the teleop readout it is printed next to.
        self.failed: str | None = None

        self._pending: Any | None = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None

    # -- lifecycle ---------------------------------------------------------- #

    def start(self) -> None:
        """Start the worker. Idempotent."""
        if self._worker is not None:
            return
        self._stop.clear()
        self._worker = threading.Thread(target=self._run, name="dk1-modelview", daemon=True)
        self._worker.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Stop the worker and wait for it. Safe to call without :meth:`start`."""
        self._stop.set()
        self._wake.set()
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.join(timeout=timeout)

    def __enter__(self) -> ModelInputProbe:
        self.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.stop()

    # -- the loop's half ---------------------------------------------------- #

    def __call__(self, observation: Any) -> Any:
        """Pass the observation through; every ``every`` ticks, offer it to the worker.

        The dict copy is shallow, which is the right thing: ``OpenCVCamera``
        publishes each frame as a **new** array rather than writing into the one
        it handed out, so the reference the worker holds stays the frame the loop
        saw and no pixels are copied on the control loop.
        """
        result = self.inner(observation)
        self.ticks += 1
        if self.failed is None and self._worker is not None and self.ticks % self.every == 0:
            with self._lock:
                if self._pending is not None:
                    self.dropped += 1
                self._pending = dict(observation)
            self._wake.set()
        return result

    # -- the worker's half -------------------------------------------------- #

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(timeout=0.2)
            self._wake.clear()
            with self._lock:
                observation, self._pending = self._pending, None
            if observation is not None and not self._stop.is_set():
                self._log(observation)

    def _log(self, observation: Any) -> None:
        try:
            import rerun as rr
            from lerobot.utils.constants import OBS_STR
            from lerobot.utils.feature_utils import build_dataset_frame

            if not all(name in observation for name in CAMERA_NAMES):
                return  # no cameras attached; nothing to show
            frame = build_dataset_frame(self.features, observation, OBS_STR)
            frame["task"] = self.task
            images = model_input_images(self.preprocessor(frame))
            for name, image in images.items():
                rr.log(f"policy_input/{name}", rr.Image(image))
            self.logged += len(images)
        except Exception as exc:  # noqa: BLE001 - a display must never stop teleop
            self.failed = str(exc)
            logger.warning("model-input view disabled after an error: %s", exc)


def pin_blueprint(camera_names: tuple[str, ...] = CAMERA_NAMES) -> bool:
    """Lay Rerun out with the model's-eye views included, and stop LeRobot's.

    ``log_rerun_data`` builds a blueprint from the first observation it sees and
    caches it on itself. That blueprint is an explicit grid of views, so anything
    it does not know about — ``policy_input/*``, which is logged from here and
    never passes through it — gets no view and is invisible until the operator
    builds one by hand in the viewer.

    Filling the cache first is what stops that: ``_ensure_blueprint`` returns
    early when the attribute is already set, so a layout sent here is the one
    that survives. It is a coupling to an upstream implementation detail and is
    written down as such; the alternative is a flag that silently shows nothing.

    Returns:
        Whether the blueprint was sent. ``False`` if Rerun is not installed, is
        not initialised, or upstream's cache attribute has gone — in every case
        the views still log and Rerun falls back to its own automatic layout,
        which is a worse arrangement of the right data rather than a failure.
    """
    try:
        import rerun as rr
        import rerun.blueprint as rrb
        from lerobot.utils import rerun_visualization
    except ImportError:  # pragma: no cover - display is opt-in
        return False

    log_rerun_data = getattr(rerun_visualization, "log_rerun_data", None)
    if log_rerun_data is None or not hasattr(log_rerun_data, "blueprint"):
        return False

    model_paths = [f"policy_input/{key.rsplit('.', 1)[-1]}" for key in IMAGE_KEYS]
    robot_paths = [f"observation.{name}" for name in camera_names]
    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                *(rrb.Spatial2DView(origin=path, name=f"robot {path}") for path in robot_paths)
            ),
            rrb.Horizontal(
                *(rrb.Spatial2DView(origin=path, name=f"model {path}") for path in model_paths)
            ),
        )
    )
    try:
        rr.send_blueprint(blueprint)
    except Exception as exc:  # noqa: BLE001 - a layout must never stop teleop
        logger.warning("could not send the Rerun layout: %s", exc)
        return False
    log_rerun_data.blueprint = blueprint
    return True
