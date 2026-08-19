"""Serve the rollout checkpoint over the MolmoAct2 ``/act`` HTTP protocol.

This exists so that `sai-prasanna/molmoact2`'s `sim_eval` can drive **the exact
policy the arms run** without a line of it being modified. `sim_eval` is a pure
HTTP client: it posts three camera frames, a 14-D state and an instruction to an
``/act`` endpoint, and executes the action chunk that comes back. It never
imports a model. So the clean way to ask "is the policy the fault?" is to answer
that endpoint from here.

The alternative — the repo's own ``examples/yam/host_server_yam.py`` — loads
``allenai/MolmoAct2-BimanualYAM`` through `transformers`' ``predict_action``.
Same weights, but a different code path from the rollout, plus a fresh ~20 GB
download. This serves ``[policy]`` from ``dk1.toml``: the same directory,
the same ``policy_preprocessor.json`` / ``policy_postprocessor.json``, the same
normalisation. If the sim works and the arms do not, the model is exonerated and
everything left is ours.

**Two deliberate differences from the rollout, both structural:**

*No RTC.* The rollout runs inference in a background thread and blends
consecutive chunks, because it has a 30 Hz deadline to hold. `sim_eval` is
synchronous — it blocks on the response and then steps the simulator — so there
is no deadline, no latency to compensate and no seam to blend. RTC is real-time
scheduling around the policy, not part of it. Timing conclusions therefore do
not transfer in either direction; behavioural ones do.

*No gripper inversion.* The wire protocol **is** the YAM convention: `sim_eval`'s
own ``yam_state_adapter`` sends ``grip in [0,1] (1=open, 0=closed)`` and its
``yam_action_adapter`` maps ``1.0`` back to ManiSkill's ``-1.0`` = open. That is
the third independent statement of the convention behind ``--invert-gripper``.
The checkpoint speaks it natively, so nothing needs flipping here — and that is
useful in itself: **the sim run tests the policy with the DK1's gripper sign out
of the picture entirely.** If it grasps in sim and not on the arms, the sign
moves back up the suspect list. ``--invert-gripper`` exists for symmetry and is
almost certainly wrong to use here.

Wire protocol, matching ``examples/yam/host_server_yam.py`` exactly:

    GET  /act   -> {"status": "ok", ...}
    POST /act   -> json_numpy body
        in:  top_cam/left_cam/right_cam (H,W,3) uint8 RGB, instruction str,
             state (14,) float32, num_steps int (optional)
        out: {"actions": (N, 14) float32, "dt_ms": float}
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

# Imported at module scope, not inside build_app, and that is load-bearing:
# `from __future__ import annotations` turns every annotation into a string, and
# FastAPI resolves ``request: Request`` against the *module* globals. Imported
# locally, the name does not resolve, FastAPI decides ``request`` must be a query
# parameter, and every POST comes back 422 "Field required". These are small pure
# Python packages; torch and lerobot stay lazy, which is what actually matters.
import json_numpy
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from .layout import ACTION_KEYS, CAMERA_NAMES, DOF

# NOTE: ``json_numpy.patch()`` is deliberately NOT called. It monkeypatches the
# stdlib ``json`` module process-wide, and LeRobot's own config loading passes an
# ``object_hook`` that returns a ``SimpleNamespace`` — which json_numpy's hook
# then tries to subscript, so merely importing ``lerobot.policies`` dies with
# ``TypeError: argument of type 'types.SimpleNamespace' is not iterable``. The
# repo's server can afford the patch because it never imports LeRobot. Calling
# ``json_numpy.dumps`` / ``loads`` explicitly needs no patch and is what this
# module does; `sim_eval` patches on its own side, in its own process.

logger = logging.getLogger(__name__)

#: The port `sim_eval`'s YAM schema defaults to. Matching it means the documented
#: `--remote-url http://localhost:8202/act` works with nothing overridden.
DEFAULT_PORT: int = 8202

#: Flow-matching integration steps. The repo's server defaults to 10; LeRobot's
#: policy defaults to 8, and 8 is what the arms run — so 8 is the default here,
#: because "the same policy as the robot" is the whole point. ``num_steps`` in
#: the request body overrides it, which is how to check that it does not matter.
DEFAULT_FLOW_STEPS: int | None = None


@dataclass
class ActServer:
    """A loaded policy behind the ``/act`` contract. Holds the GPU, not the robot.

    No ``/dev`` node is opened and no motor is energised: this is the same class
    of command as ``dk1 policy smoke``. It is safe to run with the cell powered
    down, which is the point — the sim needs no hardware.
    """

    policy: Any
    preprocessor: Any
    postprocessor: Any
    features: dict
    device: str
    task_override: str | None = None
    #: CUDA-graph capture in the action expert is not re-entrant, and `sim_eval`
    #: is single-threaded anyway. Serialising is free and removes a whole class
    #: of "it worked yesterday".
    _lock: threading.Lock = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self._lock is None:
            self._lock = threading.Lock()

    def predict(
        self,
        *,
        images: dict[str, Any],
        state: Any,
        instruction: str,
        num_steps: int | None = None,
    ) -> Any:
        """One action chunk, ``(chunk, 14)`` float32, in the YAM wire convention.

        Walks exactly the path :func:`dk1lab.policy.prewarm` walks — dataset
        frame, ``prepare_observation_for_inference``, the policy preprocessor,
        ``predict_action_chunk``, the policy postprocessor. The whole chunk is
        returned rather than one action: `sim_eval` buffers it and executes
        ``--n-action-steps`` of it per server call.
        """
        import numpy as np
        import torch
        from lerobot.policies.utils import prepare_observation_for_inference
        from lerobot.utils.constants import OBS_STR
        from lerobot.utils.feature_utils import build_dataset_frame

        vector = np.asarray(state, dtype=np.float32).reshape(-1)
        if vector.shape != (DOF,):
            raise ValueError(f"state must be ({DOF},), got {tuple(vector.shape)}")

        task = self.task_override or instruction
        values: dict[str, Any] = dict(zip(ACTION_KEYS, (float(v) for v in vector), strict=True))
        values.update(images)

        frame = build_dataset_frame(self.features, values, OBS_STR)
        batch = prepare_observation_for_inference(
            frame, torch.device(self.device), task, "sim_eval"
        )
        batch["task"] = [task]

        kwargs: dict[str, Any] = {}
        if num_steps:
            kwargs["num_steps"] = int(num_steps)

        with self._lock, torch.inference_mode():
            chunk = self.policy.predict_action_chunk(self.preprocessor(batch), **kwargs)
            actions = self.postprocessor(chunk)

        array = actions.detach().to(dtype=torch.float32, device="cpu").numpy()
        while array.ndim > 2:
            array = array[0]
        return np.asarray(array, dtype=np.float32)


def build_server(
    checkpoint: str,
    *,
    device: str = "cuda",
    dtype: str = "bfloat16",
    width: int = 640,
    height: int = 360,
    invert_gripper: bool = False,
    task: str | None = None,
) -> ActServer:
    """Load the checkpoint and its saved pipelines. **No robot, no /dev, no motion.**

    Deliberately the same loading path as :func:`dk1lab.policy.smoke`: the policy
    config from :func:`~dk1lab.policy.policy_config` (which is what overrides the
    checkpoint's baked-in ``"device": "cpu"``), the pipelines rebuilt from the
    checkpoint's own JSON by ``make_pre_post_processors``, and
    :func:`~dk1lab.policy.freeze_for_inference`. Nothing here is a sim-specific
    reimplementation — if it were, a difference between sim and hardware could
    just be this file.
    """
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    from .policy import (
        apply_gripper_inversion,
        dataset_features,
        freeze_for_inference,
        policy_config,
    )

    config = policy_config(
        checkpoint, device=device, dtype=dtype, invert_gripper=invert_gripper
    )
    logger.info("loading %s ...", config.pretrained_path)
    policy = get_policy_class(config.type).from_pretrained(
        config.pretrained_path, config=config
    )
    policy = policy.to(device)
    policy.eval()
    freeze_for_inference(policy)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=config.pretrained_path,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    if invert_gripper:
        apply_gripper_inversion(preprocessor, postprocessor)

    return ActServer(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        features=dataset_features(width=width, height=height),
        device=device,
        task_override=task,
    )


def build_app(server: ActServer, *, checkpoint: str) -> FastAPI:
    """The FastAPI app. Shapes and field names follow the repo's server exactly."""
    import numpy as np

    app = FastAPI(title="dk1 MolmoAct2 /act server", version="0.1.0")

    def error(status: int, message: str) -> Response:
        return Response(
            content=json_numpy.dumps({"error": message}),
            status_code=status,
            media_type="application/json",
        )

    @app.get("/act")
    async def health() -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "checkpoint": str(checkpoint),
                "norm_tag": "yam_dual_molmoact2",
                "device": server.device,
                "num_cameras": len(CAMERA_NAMES),
                "state_dim": DOF,
                "served_by": "dk1lab.serve",
            }
        )

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    @app.post("/act")
    async def act(request: Request) -> Response:
        try:
            payload = json_numpy.loads((await request.body()).decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - a bad body is the client's fault
            return error(400, f"failed to decode json_numpy body: {exc}")

        try:
            images = {name: to_uint8_rgb(payload[f"{name}_cam"]) for name in CAMERA_NAMES}
            instruction = str(payload["instruction"])
            state = payload["state"]
        except KeyError as exc:
            return error(400, f"missing required field: {exc}")
        except ValueError as exc:
            return error(400, str(exc))

        start = time.perf_counter()
        try:
            actions = server.predict(
                images=images,
                state=state,
                instruction=instruction,
                num_steps=payload.get("num_steps") or DEFAULT_FLOW_STEPS,
            )
        except Exception as exc:  # noqa: BLE001 - report, do not take the server down
            logger.exception("inference failed")
            return error(500, f"inference failed: {exc}")
        dt_ms = (time.perf_counter() - start) * 1000.0

        logger.info(
            "act: %s -> %s in %.0f ms  | first %s",
            np.asarray(state, dtype=np.float32).shape,
            actions.shape,
            dt_ms,
            " ".join(f"{v:+.3f}" for v in actions[0][:4]),
        )
        return Response(
            content=json_numpy.dumps({"actions": actions, "dt_ms": dt_ms}),
            media_type="application/json",
        )

    return app


def to_uint8_rgb(image: Any) -> Any:
    """An HWC uint8 RGB frame, from whatever `sim_eval` put on the wire.

    `sim_eval` already does this on its side (``common._to_uint8``), so in
    practice this is a no-op — but it is the boundary of a process, and a silent
    float image would reach the model as near-black rather than fail.
    """
    import numpy as np

    array = np.asarray(image)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"image must be (H, W, 3), got {tuple(array.shape)}")
    if array.dtype != np.uint8:
        scale = 255.0 if float(array.max(initial=0.0)) <= 1.0 else 1.0
        array = np.clip(array * scale, 0, 255).astype(np.uint8)
    return array


def serve(
    checkpoint: str,
    *,
    host: str = "0.0.0.0",
    port: int = DEFAULT_PORT,
    device: str = "cuda",
    dtype: str = "bfloat16",
    width: int = 640,
    height: int = 360,
    invert_gripper: bool = False,
    task: str | None = None,
    warmup: bool = True,
) -> None:
    """Load the policy and block, serving ``/act``. Ctrl-C stops it.

    Args:
        warmup: run one inference on a black frame before listening. The first
            call pays model warmup and CUDA-graph capture — measured at ~950 ms
            here against a steady state of ~170 — and `sim_eval`'s client has a
            60 s timeout, so this is politeness rather than necessity. It also
            fails loudly at startup instead of on the first episode.
    """
    import numpy as np
    import uvicorn

    server = build_server(
        checkpoint,
        device=device,
        dtype=dtype,
        width=width,
        height=height,
        invert_gripper=invert_gripper,
        task=task,
    )

    if warmup:
        blank = np.zeros((height, width, 3), dtype=np.uint8)
        start = time.perf_counter()
        chunk = server.predict(
            images=dict.fromkeys(CAMERA_NAMES, blank),
            state=np.zeros(DOF, dtype=np.float32),
            instruction="warmup",
        )
        logger.info(
            "warmup: %s chunk in %.0f ms", chunk.shape, (time.perf_counter() - start) * 1000.0
        )

    app = build_app(server, checkpoint=checkpoint)
    logger.info("listening on %s:%d  (POST /act)", host, port)
    uvicorn.run(app, host=host, port=port, log_level="warning")
