"""Capture-mode probing: does a camera really offer the configured profile?

Parsing is exercised against output recorded from a real Innomaker U30CAM-4K, so
these tests need no camera.
"""

from __future__ import annotations

import pytest

from dk1lab.discovery.formats import (
    Mode,
    ProbeError,
    check_profiles,
    find_mode,
    nearest_aspect,
    parse_formats_ext,
    probe,
)

# Trimmed from `v4l2-ctl -d /dev/video0 --list-formats-ext` on the DK1 cell.
# 3840x2160 is deliberately kept: the device advertises it twice under MJPG,
# which is what the rate-merging in the parser exists for.
INNOMAKER = """ioctl: VIDIOC_ENUM_FMT
	Type: Video Capture

	[0]: 'MJPG' (Motion-JPEG, compressed)
		Size: Discrete 3840x2160
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 640x360
			Interval: Discrete 0.017s (60.000 fps)
			Interval: Discrete 0.020s (50.000 fps)
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 640x480
			Interval: Discrete 0.017s (60.000 fps)
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 1280x720
			Interval: Discrete 0.017s (60.000 fps)
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 3840x2160
			Interval: Discrete 0.067s (15.000 fps)
	[1]: 'YUYV' (YUYV 4:2:2)
		Size: Discrete 640x360
			Interval: Discrete 0.017s (60.000 fps)
		Size: Discrete 1280x720
			Interval: Discrete 0.033s (30.000 fps)
"""


@pytest.fixture
def modes():
    return parse_formats_ext(INNOMAKER)


def test_both_pixel_formats_are_found(modes):
    assert {m.fourcc for m in modes} == {"MJPG", "YUYV"}


def test_a_size_repeated_under_one_format_merges_its_rates(modes):
    """The 4K entry is advertised twice; it must not become two modes."""
    uhd = [m for m in modes if m.fourcc == "MJPG" and m.width == 3840]
    assert len(uhd) == 1
    assert uhd[0].fps == (30.0, 15.0)


def test_the_same_size_under_two_formats_stays_two_modes(modes):
    assert len([m for m in modes if m.width == 640 and m.height == 360]) == 2


def test_rates_are_kept_per_size(modes):
    mode = find_mode(modes, fourcc="MJPG", width=640, height=360, fps=30)
    assert mode is not None
    assert mode.fps == (60.0, 50.0, 30.0)


def test_the_policy_profile_is_offered(modes):
    """640x360 MJPG@30 is what [capture.policy] asserts. Verified on hardware."""
    assert find_mode(modes, fourcc="MJPG", width=640, height=360, fps=30) is not None


def test_the_teleop_profile_is_offered(modes):
    assert find_mode(modes, fourcc="MJPG", width=1280, height=720, fps=60) is not None


def test_a_size_offered_only_under_another_format_does_not_match(modes):
    """YUYV 640x480 is absent, so asking for it must not fall through to MJPG."""
    assert find_mode(modes, fourcc="YUYV", width=640, height=480, fps=30) is None


def test_an_unadvertised_rate_does_not_match(modes):
    assert find_mode(modes, fourcc="MJPG", width=640, height=480, fps=50) is None


def test_fps_tolerance_absorbs_the_drivers_rounding():
    mode = Mode("MJPG", 640, 360, (29.97,))
    assert mode.supports_fps(30)
    assert not mode.supports_fps(60)


def test_nearest_aspect_prefers_the_matching_ratio(modes):
    ranked = nearest_aspect(modes, fourcc="MJPG", aspect=16 / 9)
    assert (ranked[0].width, ranked[0].height) == (640, 360)
    assert all(m.fourcc == "MJPG" for m in ranked)


def test_nearest_aspect_offers_the_43_fallback_when_169_is_wanted():
    """The fallback [capture.policy] would take if 640x360 vanished."""
    only_43 = parse_formats_ext(
        "\t[0]: 'MJPG' (x)\n"
        "\t\tSize: Discrete 640x480\n"
        "\t\t\tInterval: Discrete 0.033s (30.000 fps)\n"
    )
    ranked = nearest_aspect(only_43, fourcc="MJPG", aspect=16 / 9)
    assert (ranked[0].width, ranked[0].height) == (640, 480)


def test_empty_output_parses_to_nothing_rather_than_raising():
    assert parse_formats_ext("") == []


def test_a_size_before_any_format_header_is_ignored():
    assert parse_formats_ext("\t\tSize: Discrete 640x360\n") == []


def test_probe_reports_a_missing_v4l2_ctl_as_a_probe_error(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: None)
    with pytest.raises(ProbeError, match="v4l2-ctl"):
        probe("/dev/video0")


# --------------------------------------------------------------------------- #
# Checking whole capture profiles
# --------------------------------------------------------------------------- #


class FakeProfile:
    """Structurally what ``check_profiles`` needs from a CaptureProfile."""

    def __init__(self, width, height, fps, fourcc="MJPG"):
        self.width, self.height, self.fps, self.fourcc = width, height, fps, fourcc


def test_both_dk1_profiles_pass_against_the_real_camera_output(modes):
    checks = check_profiles(
        modes,
        {
            "policy": FakeProfile(640, 360, 30),
            "teleop": FakeProfile(1280, 720, 60),
        },
    )
    assert [c.profile for c in checks] == ["policy", "teleop"]
    assert all(c.ok for c in checks)


def test_an_unavailable_profile_is_reported_with_fallbacks(modes):
    (check,) = check_profiles(modes, {"policy": FakeProfile(1920, 1080, 30)})
    assert not check.ok
    assert check.matched is None
    assert check.alternatives  # something to fall back to
    assert check.wanted == "1920x1080 MJPG@30"


def test_a_169_fallback_is_not_flagged_as_an_aspect_mismatch(modes):
    """1920x1080 is unavailable but 640x360 has the same ratio — no warning."""
    (check,) = check_profiles(modes, {"policy": FakeProfile(1920, 1080, 30)})
    assert check.aspect_gap() is None


def test_a_43_fallback_is_flagged_because_the_resize_would_stretch():
    only_43 = parse_formats_ext(
        "\t[0]: 'MJPG' (x)\n"
        "\t\tSize: Discrete 640x480\n"
        "\t\t\tInterval: Discrete 0.033s (30.000 fps)\n"
    )
    (check,) = check_profiles(only_43, {"policy": FakeProfile(640, 360, 30)})
    assert not check.ok
    assert check.aspect_gap() == pytest.approx(16 / 9 - 4 / 3)


def test_a_satisfied_profile_reports_no_aspect_gap_and_no_alternatives(modes):
    (check,) = check_profiles(modes, {"policy": FakeProfile(640, 360, 30)})
    assert check.ok
    assert check.alternatives == ()
    assert check.aspect_gap() is None


def test_a_camera_advertising_nothing_useful_yields_no_alternatives():
    (check,) = check_profiles([], {"policy": FakeProfile(640, 360, 30)})
    assert not check.ok
    assert check.alternatives == ()
    assert check.aspect_gap() is None
