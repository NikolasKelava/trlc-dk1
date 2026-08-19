"""The ``/act`` server: the wire contract, and nothing beyond it.

No GPU and no checkpoint — :class:`~dk1lab.serve.ActServer` is subclassed with a
fake ``predict``, so what is tested here is exactly what `sim_eval` depends on:
the field names, the shapes, the status codes and the encoding. Whether the model
is any good is not a question this file can ask.
"""

from __future__ import annotations

import json_numpy
import numpy as np
import pytest
from fastapi.testclient import TestClient

from dk1lab.layout import ACTION_KEYS, CAMERA_NAMES, DOF
from dk1lab.serve import DEFAULT_PORT, ActServer, build_app, to_uint8_rgb

CHUNK = 30


class FakeServer(ActServer):
    """Records what it was asked for and answers with a recognisable chunk."""

    calls: list = []

    def predict(self, *, images, state, instruction, num_steps=None):
        self.calls.append(
            {
                "images": images,
                "state": np.asarray(state, dtype=np.float32),
                "instruction": instruction,
                "num_steps": num_steps,
            }
        )
        return np.arange(CHUNK * DOF, dtype=np.float32).reshape(CHUNK, DOF)


def client() -> tuple[TestClient, FakeServer]:
    server = FakeServer(
        policy=None, preprocessor=None, postprocessor=None, features={}, device="cpu"
    )
    server.calls = []
    return TestClient(build_app(server, checkpoint="a/checkpoint")), server


def frame(value: int = 0, size: tuple[int, int] = (8, 12)) -> np.ndarray:
    return np.full((*size, 3), value, dtype=np.uint8)


def body(**overrides) -> str:
    payload = {
        "top_cam": frame(1),
        "left_cam": frame(2),
        "right_cam": frame(3),
        "instruction": "put everything into the box",
        "state": np.zeros(DOF, dtype=np.float32),
    }
    payload.update(overrides)
    return json_numpy.dumps(payload)


# --------------------------------------------------------------------------- #
# The contract sim_eval depends on
# --------------------------------------------------------------------------- #


def test_the_health_endpoint_reports_what_this_server_is():
    api, _ = client()
    info = api.get("/act").json()
    assert info["status"] == "ok"
    assert info["state_dim"] == DOF
    assert info["num_cameras"] == len(CAMERA_NAMES)
    assert info["norm_tag"] == "yam_dual_molmoact2"
    assert info["checkpoint"] == "a/checkpoint"


def test_a_well_formed_request_returns_an_action_chunk():
    api, _ = client()
    response = api.post("/act", content=body())
    assert response.status_code == 200
    data = json_numpy.loads(response.text)
    assert np.asarray(data["actions"]).shape == (CHUNK, DOF)
    assert data["dt_ms"] >= 0.0


def test_the_three_cameras_arrive_under_this_cells_own_names():
    """``top_cam`` on the wire, ``top`` in the frame — the mapping is one place."""
    api, server = client()
    api.post("/act", content=body())
    assert list(server.calls[0]["images"]) == list(CAMERA_NAMES)
    assert int(server.calls[0]["images"]["top"][0, 0, 0]) == 1
    assert int(server.calls[0]["images"]["right"][0, 0, 0]) == 3


def test_the_state_arrives_as_fourteen_floats():
    api, server = client()
    api.post("/act", content=body(state=np.arange(DOF, dtype=np.float32)))
    assert server.calls[0]["state"] == pytest.approx(np.arange(DOF))
    assert len(ACTION_KEYS) == DOF


def test_the_instruction_is_passed_through_untouched():
    api, server = client()
    api.post("/act", content=body(instruction="pick up the red dice"))
    assert server.calls[0]["instruction"] == "pick up the red dice"


def test_num_steps_is_optional_and_forwarded_when_given():
    api, server = client()
    api.post("/act", content=body())
    assert server.calls[0]["num_steps"] is None
    api.post("/act", content=body(num_steps=10))
    assert server.calls[1]["num_steps"] == 10


def test_the_default_port_matches_the_one_sim_eval_documents():
    """`--remote-url http://localhost:8202/act` has to work unmodified."""
    assert DEFAULT_PORT == 8202


# --------------------------------------------------------------------------- #
# Failures are reported, not crashed on
# --------------------------------------------------------------------------- #


def test_a_missing_field_is_a_400_not_a_500():
    api, _ = client()
    response = api.post("/act", content=json_numpy.dumps({"instruction": "x"}))
    assert response.status_code == 400
    assert "missing required field" in json_numpy.loads(response.text)["error"]


def test_an_undecodable_body_is_a_400():
    api, _ = client()
    assert api.post("/act", content="not json at all").status_code == 400


def test_an_image_of_the_wrong_shape_is_a_400():
    api, _ = client()
    response = api.post("/act", content=body(top_cam=np.zeros((8, 12), dtype=np.uint8)))
    assert response.status_code == 400
    assert "(H, W, 3)" in json_numpy.loads(response.text)["error"]


def test_an_inference_failure_is_a_500_and_the_server_stays_up():
    api, server = client()

    def explode(**_kwargs):
        raise RuntimeError("CUDA fell over")

    server.predict = explode
    assert api.post("/act", content=body()).status_code == 500
    server.predict = FakeServer.predict.__get__(server)
    assert api.post("/act", content=body()).status_code == 200


# --------------------------------------------------------------------------- #
# Image normalisation at the process boundary
# --------------------------------------------------------------------------- #


def test_a_uint8_frame_passes_through_unchanged():
    source = frame(200)
    assert to_uint8_rgb(source) is not None
    assert int(to_uint8_rgb(source)[0, 0, 0]) == 200


def test_a_float_frame_in_zero_to_one_is_scaled_rather_than_silently_blackened():
    """`sim_eval` already sends uint8; a float one must not reach the model as near-black."""
    array = to_uint8_rgb(np.full((4, 6, 3), 0.5, dtype=np.float32))
    assert array.dtype == np.uint8
    assert int(array[0, 0, 0]) == 127


def test_a_float_frame_already_in_zero_to_255_is_not_scaled_again():
    array = to_uint8_rgb(np.full((4, 6, 3), 200.0, dtype=np.float32))
    assert int(array[0, 0, 0]) == 200


def test_a_batched_frame_loses_its_batch_dimension():
    assert to_uint8_rgb(np.zeros((1, 4, 6, 3), dtype=np.uint8)).shape == (4, 6, 3)


def test_something_that_is_not_an_image_is_rejected_by_shape():
    with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
        to_uint8_rgb(np.zeros((4, 6, 4), dtype=np.uint8))


# --------------------------------------------------------------------------- #
# The stdlib json module must survive importing this
# --------------------------------------------------------------------------- #


def test_importing_the_server_does_not_patch_the_stdlib_json_module():
    """``json_numpy.patch()`` process-wide breaks importing lerobot. See dk1lab/serve.py."""
    import json
    from types import SimpleNamespace

    import dk1lab.serve  # noqa: F401

    decoded = json.loads('{"a": 1}', object_hook=lambda d: SimpleNamespace(**d))
    assert decoded.a == 1
