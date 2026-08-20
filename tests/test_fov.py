"""Field-of-view arithmetic: the crop that turns our lens into the trained one.

No cv2, no LeRobot — this is the half of the crop that is pure geometry.
"""

from __future__ import annotations

import math

import pytest

from dk1lab.fov import (
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
