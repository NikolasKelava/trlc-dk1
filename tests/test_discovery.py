"""Device discovery, driven entirely by faked device listings."""

from __future__ import annotations

import pytest

from dk1lab.config import ArmPorts
from dk1lab.discovery.arms import ARM_SEQUENCE, DiscoveryError, detect_removed, find_arms
from dk1lab.discovery.cameras import CameraCandidate, candidates_from_names

# --------------------------------------------------------------------------- #
# Arms
# --------------------------------------------------------------------------- #


def test_detect_removed_finds_the_missing_port():
    before = {"/dev/ttyACM0", "/dev/ttyACM1", "/dev/ttyACM2"}
    after = {"/dev/ttyACM0", "/dev/ttyACM2"}
    assert detect_removed(before, after, what="left follower arm") == "/dev/ttyACM1"


def test_detect_removed_refuses_when_nothing_disappeared():
    ports = {"/dev/ttyACM0"}
    with pytest.raises(DiscoveryError, match="No serial port disappeared"):
        detect_removed(ports, ports, what="left follower arm")


def test_detect_removed_refuses_when_several_disappeared():
    """Two at once is ambiguous; guessing would write a wrong port."""
    before = {"/dev/ttyACM0", "/dev/ttyACM1", "/dev/ttyACM2"}
    after = {"/dev/ttyACM0"}
    with pytest.raises(DiscoveryError, match="2 serial ports disappeared"):
        detect_removed(before, after, what="left follower arm")


def test_detect_removed_ignores_ports_that_appeared():
    """Something else plugged in mid-scan must not confuse the comparison."""
    before = {"/dev/ttyACM0", "/dev/ttyACM1"}
    after = {"/dev/ttyACM0", "/dev/ttyACM9"}
    assert detect_removed(before, after, what="x") == "/dev/ttyACM1"


class FakeDevices:
    """A /dev that unplugs one scripted port per round."""

    def __init__(self, ports: list[str], order: list[str]) -> None:
        self.all = list(ports)
        self.order = list(order)
        self.calls = 0

    def __call__(self) -> set[str]:
        # Odd-numbered calls are the "after unplugging" listing.
        self.calls += 1
        if self.calls % 2 == 0:
            missing = self.order[(self.calls // 2) - 1]
            return set(self.all) - {missing}
        return set(self.all)


def test_find_arms_assigns_every_arm_in_order():
    ports = ["/dev/ttyACM0", "/dev/ttyACM1", "/dev/ttyACM2", "/dev/ttyACM3"]
    # The order the operator will be asked to unplug them in.
    unplug_order = ["/dev/ttyACM1", "/dev/ttyACM3", "/dev/ttyACM0", "/dev/ttyACM2"]
    lister = FakeDevices(ports, unplug_order)

    result = find_arms(
        prompt=lambda _: "",
        announce=lambda _: None,
        lister=lister,
        sleep=lambda _: None,
    )

    assert result == {
        "follower": ArmPorts(left="/dev/ttyACM1", right="/dev/ttyACM3"),
        "leader": ArmPorts(left="/dev/ttyACM0", right="/dev/ttyACM2"),
    }


def test_find_arms_asks_about_each_arm_exactly_once():
    ports = ["/dev/ttyACM0", "/dev/ttyACM1", "/dev/ttyACM2", "/dev/ttyACM3"]
    lister = FakeDevices(ports, ["/dev/ttyACM1", "/dev/ttyACM3", "/dev/ttyACM0", "/dev/ttyACM2"])
    asked: list[str] = []

    find_arms(
        prompt=lambda message: asked.append(message) or "",
        announce=lambda _: None,
        lister=lister,
        sleep=lambda _: None,
    )

    unplug_prompts = [m for m in asked if "Unplug" in m]
    assert len(unplug_prompts) == len(ARM_SEQUENCE) == 4
    for role, side in ARM_SEQUENCE:
        assert any(f"{side} {role} arm" in m for m in unplug_prompts)


def test_find_arms_refuses_a_duplicate_assignment():
    """A device that had not finished re-enumerating must not be written."""
    ports = ["/dev/ttyACM0", "/dev/ttyACM1"]
    lister = FakeDevices(ports, ["/dev/ttyACM1", "/dev/ttyACM1", "/dev/ttyACM0", "/dev/ttyACM0"])
    with pytest.raises(DiscoveryError, match="both resolved to"):
        find_arms(
            prompt=lambda _: "",
            announce=lambda _: None,
            lister=lister,
            sleep=lambda _: None,
        )


def test_find_arms_writes_nothing_on_failure(config_file):
    """A failed scan must leave the config exactly as it was."""
    from dk1lab.config import load

    before_text = config_file.read_text()
    before = load(config_file)
    ports = ["/dev/ttyACM0"]
    lister = FakeDevices(ports, ["/dev/ttyACM0"] * 4)
    with pytest.raises(DiscoveryError):
        find_arms(prompt=lambda _: "", announce=lambda _: None, lister=lister, sleep=lambda _: None)
    assert config_file.read_text() == before_text
    assert load(config_file).follower == before.follower


# --------------------------------------------------------------------------- #
# Cameras
# --------------------------------------------------------------------------- #

# A real listing from this machine: three cameras, each exposing a video node and
# a metadata node, each under both the usb- and usbv3- spellings. Twelve entries
# for three cameras.
REAL_LISTING = [
    "pci-0000:00:14.0-usb-0:10.1:1.0-video-index0",
    "pci-0000:00:14.0-usb-0:10.1:1.0-video-index1",
    "pci-0000:00:14.0-usb-0:4.3:1.0-video-index0",
    "pci-0000:00:14.0-usb-0:4.3:1.0-video-index1",
    "pci-0000:00:14.0-usb-0:4.4:1.0-video-index0",
    "pci-0000:00:14.0-usb-0:4.4:1.0-video-index1",
    "pci-0000:00:14.0-usbv3-0:10.1:1.0-video-index0",
    "pci-0000:00:14.0-usbv3-0:10.1:1.0-video-index1",
    "pci-0000:00:14.0-usbv3-0:4.3:1.0-video-index0",
    "pci-0000:00:14.0-usbv3-0:4.3:1.0-video-index1",
    "pci-0000:00:14.0-usbv3-0:4.4:1.0-video-index0",
    "pci-0000:00:14.0-usbv3-0:4.4:1.0-video-index1",
]


def test_twelve_nodes_resolve_to_three_cameras():
    """Each camera exposes four nodes; the operator must be shown three."""
    assert len(candidates_from_names(REAL_LISTING)) == 3


def test_metadata_nodes_are_excluded():
    """-index1 carries UVC metadata, not frames — opening it yields nothing."""
    for candidate in candidates_from_names(REAL_LISTING):
        assert candidate.by_path.endswith("-video-index0")


def test_the_usbv3_duplicate_spelling_is_collapsed():
    for candidate in candidates_from_names(REAL_LISTING):
        assert "-usbv3-" not in candidate.by_path


def test_a_usbv3_only_device_is_still_offered():
    """Collapsing duplicates must not drop a camera that has no usb- spelling."""
    listing = REAL_LISTING + ["pci-0000:00:14.0-usbv3-0:9.9:1.0-video-index0"]
    ports = {c.hub_port for c in candidates_from_names(listing)}
    assert ports == {"10.1", "4.3", "4.4", "9.9"}


def test_hub_port_is_extracted_for_display():
    ports = sorted(c.hub_port for c in candidates_from_names(REAL_LISTING))
    assert ports == ["10.1", "4.3", "4.4"]


def test_candidates_are_absolute_device_paths(tmp_path):
    candidates = candidates_from_names(REAL_LISTING, base=tmp_path)
    for candidate in candidates:
        assert candidate.by_path.startswith(str(tmp_path))


def test_empty_listing_yields_nothing():
    assert candidates_from_names([]) == []


def test_hub_port_of_an_unparseable_name_is_marked_unknown():
    assert CameraCandidate(by_path="/dev/v4l/by-path/weird").hub_port == "?"
