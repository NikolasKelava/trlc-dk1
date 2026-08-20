"""Field-of-view arithmetic: the crop that turns our lens into the trained one.

No cv2, no LeRobot — this is the half of the crop that is pure geometry.
"""

from __future__ import annotations

import math

import pytest

from dk1lab.fov import (
    REFERENCE_WIDTH,
    FOVError,
    crop_box,
    describe,
    hfov_from_scale,
    hfov_scale,
    vfov,
)

#: Our Innomaker U30CAM-4K-S1, per its user manual.
OURS = 105.0
#: The BimanualYAM checkpoint's wrist views (RealSense D405) and top view (D435i).
WRIST, TOP = 87.0, 69.4


def test_scale_is_the_ratio_of_half_angle_tangents():
    """The pinhole relation the checkpoint's own intrinsics are built from."""
    assert hfov_scale(OURS, WRIST) == pytest.approx(
        math.tan(math.radians(WRIST / 2)) / math.tan(math.radians(OURS / 2))
    )


def test_scale_round_trips_through_hfov_from_scale():
    assert hfov_from_scale(OURS, hfov_scale(OURS, WRIST)) == pytest.approx(WRIST)


def test_the_same_field_of_view_needs_no_crop():
    assert hfov_scale(OURS, OURS) == pytest.approx(1.0)
    assert crop_box(640, 360, OURS, OURS).is_full_frame


def test_a_narrower_lens_cannot_be_widened():
    """The one case cropping cannot fix, so it must not pretend to."""
    with pytest.raises(FOVError, match="only narrows"):
        crop_box(640, 360, WRIST, OURS)


def test_the_wrist_crop_is_the_box_dk1_toml_documents():
    box = crop_box(640, 360, OURS, WRIST)
    assert (box.width, box.height) == (467, 263)
    assert (box.x, box.y) == (86, 48)


def test_the_tuned_wrist_box_is_the_one_dk1_toml_configures():
    """inset 6, lifted 20 — at the 1280x720 the policy actually captures."""
    box = crop_box(1280, 720, OURS, WRIST, inset=6, shift_y=-20)
    assert (box.width, box.height) == (909, 511)
    assert (box.x, box.y) == (185, 64)


def test_the_tuned_box_still_clears_the_models_input_size():
    """378 rows is what MolmoAct2 resizes to; fewer means inventing detail."""
    box = crop_box(1280, 720, OURS, WRIST, inset=6, shift_y=-20)
    assert box.height >= 378
    assert box.width >= 378


def test_the_achieved_field_of_view_is_slightly_wider_than_asked():
    """Rounding always takes the larger box — never show the policy less."""
    box = crop_box(640, 360, OURS, WRIST)
    got = hfov_from_scale(OURS, box.width / 640)
    assert got >= WRIST
    assert got == pytest.approx(WRIST, abs=0.5)


def test_the_crop_lands_on_the_trained_vertical_field_of_view_too():
    """The sim sets fy = fx, so matching HFOV at 16:9 matches VFOV as well."""
    box = crop_box(640, 360, OURS, WRIST)
    got = vfov(hfov_from_scale(OURS, box.width / 640), box.width, box.height)
    assert got == pytest.approx(vfov(WRIST, 640, 360), abs=0.5)


def test_the_box_fits_inside_the_frame():
    box = crop_box(640, 360, OURS, TOP)
    assert box.x >= 0 and box.y >= 0
    assert box.x + box.width <= 640
    assert box.y + box.height <= 360


def test_the_box_is_centred_to_within_the_odd_pixel():
    box = crop_box(640, 360, OURS, WRIST)
    assert abs((640 - box.width) - 2 * box.x) <= 1
    assert abs((360 - box.height) - 2 * box.y) <= 1


def test_the_top_crop_would_fall_below_the_models_input_size():
    """Why dk1.toml leaves the overhead view alone: 192 rows < 224."""
    box = crop_box(640, 360, OURS, TOP)
    assert box.height < 224


def test_apply_slices_the_named_window():
    numpy = pytest.importorskip("numpy")
    frame = numpy.zeros((360, 640, 3), numpy.uint8)
    box = crop_box(640, 360, OURS, WRIST)
    box.apply(frame)[:] = 255
    assert frame[box.y, box.x, 0] == 255
    assert frame[0, 0, 0] == 0
    assert frame[-1, -1, 0] == 0


@pytest.mark.parametrize("bad", [0.0, 180.0, -5.0, True, "87"])
def test_a_field_of_view_outside_zero_to_one_eighty_is_rejected(bad):
    with pytest.raises(FOVError):
        hfov_scale(OURS, bad)


def test_describe_names_the_box_and_what_it_achieves_not_what_was_asked():
    line = describe(640, 360, OURS, WRIST)
    assert "467x263" in line
    assert "87.1 deg H" in line
    assert "asked 87" in line


# --------------------------------------------------------------------------- #
# The hand-tuned adjustments: inset and shift
# --------------------------------------------------------------------------- #


def test_the_offsets_are_quoted_at_the_reference_width_and_scale():
    """The whole point: identical geometry at every capture resolution, so what
    you approve in teleop is what the policy is handed in a rollout."""
    small = crop_box(REFERENCE_WIDTH, 360, OURS, WRIST, inset=6, shift_y=-20)
    large = crop_box(2 * REFERENCE_WIDTH, 720, OURS, WRIST, inset=6, shift_y=-20)
    assert large.width == pytest.approx(2 * small.width, abs=2)
    assert large.shift_y == pytest.approx(2 * small.shift_y, abs=2)
    assert hfov_from_scale(OURS, small.width / small.frame_width) == pytest.approx(
        hfov_from_scale(OURS, large.width / large.frame_width), abs=0.1
    )


def test_inset_takes_six_pixels_off_each_side():
    plain = crop_box(640, 360, OURS, WRIST)
    inset = crop_box(640, 360, OURS, WRIST, inset=6)
    assert inset.width == plain.width - 12


def test_inset_keeps_the_frames_aspect_ratio():
    """A box that did not would be stretched anisotropically on the way out —
    the exact distortion this module exists to remove."""
    box = crop_box(1280, 720, OURS, WRIST, inset=6)
    assert box.width / box.height == pytest.approx(1280 / 720, rel=2e-3)


def test_inset_narrows_the_field_of_view():
    plain = hfov_from_scale(OURS, crop_box(640, 360, OURS, WRIST).width / 640)
    inset = hfov_from_scale(OURS, crop_box(640, 360, OURS, WRIST, inset=6).width / 640)
    assert inset < plain


def test_a_negative_shift_moves_the_box_up():
    """Up = see more above the lens's centre line, lose the same off the bottom."""
    plain = crop_box(640, 360, OURS, WRIST)
    up = crop_box(640, 360, OURS, WRIST, shift_y=-20)
    assert up.y == plain.y - 20
    assert up.shift_y == -20


def test_a_positive_shift_moves_the_box_down():
    assert crop_box(640, 360, OURS, WRIST, shift_y=20).shift_y == 20


def test_shifting_does_not_change_the_field_of_view():
    """It re-aims the box; only inset and the target angle resize it."""
    plain = crop_box(640, 360, OURS, WRIST)
    moved = crop_box(640, 360, OURS, WRIST, shift_y=-20, shift_x=10)
    assert (moved.width, moved.height) == (plain.width, plain.height)


def test_a_shift_off_the_sensor_is_clamped_not_raised():
    """Retuning a number beats refusing to produce a picture mid-rollout."""
    box = crop_box(640, 360, OURS, WRIST, shift_y=-9999)
    assert box.y == 0
    assert box.y + box.height <= 360
    assert crop_box(640, 360, OURS, WRIST, shift_x=9999).x + box.width <= 640


def test_the_box_reports_the_offset_it_actually_achieved():
    """Clamped or not, shift_y is readable — so a silently clamped shift shows."""
    assert crop_box(640, 360, OURS, WRIST, shift_y=-9999).shift_y > -9999


def test_a_full_frame_box_has_nowhere_to_shift_to():
    """It fills the sensor, so a shift clamps to zero and it stays a pass-through
    rather than sliding off and blacking out an edge."""
    assert crop_box(640, 360, OURS, OURS).is_full_frame
    assert crop_box(640, 360, OURS, OURS, shift_y=-20).is_full_frame
    # but an inset box can be shifted, and is no longer full frame
    assert not crop_box(640, 360, OURS, OURS, inset=6).is_full_frame
    assert not crop_box(640, 360, OURS, WRIST).is_full_frame


def test_describe_names_the_adjustments_when_there_are_any():
    line = describe(1280, 720, OURS, WRIST, inset=6, shift_y=-20)
    assert "inset 6" in line and "shift +0,-20" in line
    assert "inset" not in describe(1280, 720, OURS, WRIST)
