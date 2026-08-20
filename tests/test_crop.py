"""The cropping camera: right pixels, right shape, right refusals.

No hardware. ``_postprocess_image`` is the whole of what this subclass adds, and
it is a pure function of a frame, so it is driven directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from dk1lab.crop import CroppedOpenCVCamera, CroppedOpenCVCameraConfig
from dk1lab.fov import FOVError, crop_box

OURS, WRIST = 105.0, 87.0


def camera(**overrides) -> CroppedOpenCVCamera:
    settings = dict(
        index_or_path="/dev/null",
        width=640,
        height=360,
        fps=30,
        source_hfov_deg=OURS,
        target_hfov_deg=WRIST,
    )
    settings.update(overrides)
    return CroppedOpenCVCamera(CroppedOpenCVCameraConfig(**settings))


def gradient(width: int = 640, height: int = 360) -> np.ndarray:
    """A left-to-right ramp, so which columns survived a crop is legible.

    Monotonic and non-wrapping over uint8, which a bare column index is not.
    """
    columns = (np.arange(width) * 255 // (width - 1)).astype(np.uint8)
    return np.repeat(np.tile(columns, (height, 1))[..., None], 3, axis=2)


def ramp_value(column: int, width: int = 640) -> int:
    """What :func:`gradient` puts in a given column."""
    return column * 255 // (width - 1)


def test_the_output_keeps_the_configured_shape():
    """The whole point of resizing back: nothing downstream has to change."""
    out = camera()._postprocess_image(np.zeros((360, 640, 3), np.uint8))
    assert out.shape == (360, 640, 3)


def test_only_the_centre_survives():
    cam = camera()
    box = crop_box(640, 360, OURS, WRIST)
    frame = np.zeros((360, 640, 3), np.uint8)
    box.apply(frame)[:] = 255
    assert (cam._postprocess_image(frame) == 255).all()


def test_the_edges_of_the_frame_are_gone():
    """The columns the crop drops must not reappear anywhere in the output."""
    out = camera()._postprocess_image(gradient())
    box = crop_box(640, 360, OURS, WRIST)
    assert out[180, 0, 0] == pytest.approx(ramp_value(box.x), abs=2)
    assert out[180, -1, 0] == pytest.approx(ramp_value(box.x + box.width - 1), abs=2)
    # and the frame's own edges, which the crop drops, are nowhere in it
    assert out[180, 0, 0] > ramp_value(0) + 2
    assert out[180, -1, 0] < ramp_value(639) - 2


def test_the_result_is_a_fresh_contiguous_buffer_not_a_view():
    """Video encoders and torch.from_numpy both assume contiguous memory."""
    frame = gradient()
    out = camera()._postprocess_image(frame)
    assert out.flags["C_CONTIGUOUS"]
    assert not np.shares_memory(out, frame)


def test_an_identical_field_of_view_is_a_pass_through():
    frame = gradient()
    out = camera(target_hfov_deg=OURS)._postprocess_image(frame)
    assert np.array_equal(out, frame)


def test_the_crop_it_reports_is_the_crop_it_does():
    cam = camera()
    assert (cam.crop.width, cam.crop.height) == (467, 263)
    assert cam.achieved_hfov_deg == pytest.approx(87.1, abs=0.05)
    assert cam.achieved_vfov_deg == pytest.approx(56.3, abs=0.05)
    assert "467x263" in cam.describe_crop()


def test_rotation_by_a_half_turn_is_allowed():
    """All three of this cell's cameras are mounted upside down."""
    cam = camera(rotation=180)
    frame = gradient()
    out = cam._postprocess_image(frame)
    # 180 degrees first, then the crop: the ramp comes back reversed, so the
    # output runs bright-to-dark where the uncropped one runs dark-to-bright.
    assert out[180, 0, 0] > out[180, -1, 0]
    assert camera()._postprocess_image(frame)[180, 0, 0] < out[180, 0, 0]


def test_rotation_by_a_quarter_turn_is_refused():
    """A quarter turn puts the sensor's vertical axis across the output's width."""
    with pytest.raises(FOVError, match="horizontal field of view"):
        camera(rotation=90)


def test_half_a_configuration_is_refused():
    with pytest.raises(FOVError, match="needs both"):
        CroppedOpenCVCameraConfig(index_or_path="/dev/null", width=640, height=360, fps=30)


def test_a_target_wider_than_the_lens_is_refused():
    with pytest.raises(FOVError, match="only narrows"):
        camera(source_hfov_deg=WRIST, target_hfov_deg=OURS).crop


def test_the_config_registers_under_its_own_type_name():
    """So `--robot.cameras={type: opencv_cropped, ...}` parses."""
    assert CroppedOpenCVCameraConfig(
        index_or_path="/dev/null",
        width=640,
        height=360,
        fps=30,
        source_hfov_deg=OURS,
        target_hfov_deg=WRIST,
    ).type == "opencv_cropped"


def test_lerobot_can_build_the_camera_from_the_config():
    """`make_cameras_from_configs` has no branch for us and falls through to a
    class-name lookup in the package holding the config's module — dk1lab."""
    from lerobot.cameras.utils import make_cameras_from_configs

    built = make_cameras_from_configs(
        {
            "left": CroppedOpenCVCameraConfig(
                index_or_path="/dev/null",
                width=640,
                height=360,
                fps=30,
                source_hfov_deg=OURS,
                target_hfov_deg=WRIST,
            )
        }
    )
    assert isinstance(built["left"], CroppedOpenCVCamera)


# --------------------------------------------------------------------------- #
# The hand-tuned adjustments, through the camera
# --------------------------------------------------------------------------- #


def test_the_adjustments_reach_the_box():
    cam = camera(crop_inset=6, crop_shift_y=-20)
    assert (cam.crop.width, cam.crop.height) == (455, 256)
    assert cam.crop.shift_y == -20
    assert cam.achieved_hfov_deg == pytest.approx(85.6, abs=0.05)


def test_shifting_up_keeps_the_top_of_the_frame_and_drops_the_bottom():
    """A row that was inside the unshifted box's top edge must now be inside it,
    and one near the bottom edge must have fallen out."""
    plain, moved = camera(), camera(crop_shift_y=-20)
    assert moved.crop.y == plain.crop.y - 20
    assert moved.crop.y + moved.crop.height == plain.crop.y + plain.crop.height - 20


def test_the_output_shape_survives_every_adjustment():
    out = camera(crop_inset=6, crop_shift_x=5, crop_shift_y=-20)._postprocess_image(
        np.zeros((360, 640, 3), np.uint8)
    )
    assert out.shape == (360, 640, 3)


def test_a_shifted_box_takes_its_pixels_from_the_shifted_place():
    cam = camera(crop_shift_y=-20)
    frame = np.zeros((360, 640, 3), np.uint8)
    cam.crop.apply(frame)[:] = 255
    assert (cam._postprocess_image(frame) == 255).all()
    # and the unshifted camera, given the same frame, sees black at the bottom
    assert camera()._postprocess_image(frame)[-1, 320, 0] < 255


def test_the_tuned_wrist_crop_clears_the_models_input_size_at_720p():
    """378x378 is what MolmoAct2 resizes to; a smaller crop invents detail."""
    cam = camera(width=1280, height=720, crop_inset=6, crop_shift_y=-20)
    assert cam.crop.width >= 378
    assert cam.crop.height >= 378


def test_the_geometry_is_the_same_at_both_capture_resolutions():
    """What you approve in teleop is what the policy gets in a rollout."""
    small = camera(width=640, height=360, crop_inset=6, crop_shift_y=-20)
    large = camera(width=1280, height=720, crop_inset=6, crop_shift_y=-20)
    assert small.achieved_hfov_deg == pytest.approx(large.achieved_hfov_deg, abs=0.1)
    assert small.achieved_vfov_deg == pytest.approx(large.achieved_vfov_deg, abs=0.1)
    assert large.crop.shift_y == 2 * small.crop.shift_y
