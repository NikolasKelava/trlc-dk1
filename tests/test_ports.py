"""USB identity of the arm serial ports: what a port is, without opening it.

Built from identity recorded off the DK1 cell, so none of this needs hardware.
"""

from __future__ import annotations

from dk1lab.discovery.ports import SerialPort, describe, group_by_role, is_usb, role_conflicts

# Recorded from `python -m serial.tools.list_ports -v` on the DK1 host.
FOLLOWER_LEFT = SerialPort(
    "/dev/ttyACM1", 0x2E88, 0x4603, "00000000050C", "1-5.1:1.0", "CDC Device"
)
FOLLOWER_RIGHT = SerialPort(
    "/dev/ttyACM3", 0x2E88, 0x4603, "00000000050C", "1-5.2:1.0", "CDC Device"
)
LEADER_LEFT = SerialPort(
    "/dev/ttyACM0", 0x1A86, 0x55D3, "5A68010481", "1-4.2:1.0", "USB Single Serial"
)
LEADER_RIGHT = SerialPort(
    "/dev/ttyACM2", 0x1A86, 0x55D3, "5A68010285", "1-4.3:1.0", "USB Single Serial"
)
MOTHERBOARD = SerialPort("/dev/ttyS0")

ALL = [LEADER_LEFT, FOLLOWER_LEFT, LEADER_RIGHT, FOLLOWER_RIGHT]

#: What dk1.toml says, keyed as DK1Config.arm_ports() renders it.
CONFIGURED = {
    "follower_left": "/dev/ttyACM1",
    "follower_right": "/dev/ttyACM3",
    "leader_left": "/dev/ttyACM0",
    "leader_right": "/dev/ttyACM2",
}


def test_the_damiao_can_adapter_is_a_follower():
    assert FOLLOWER_LEFT.role == "follower"


def test_the_dynamixel_adapter_is_a_leader():
    assert LEADER_LEFT.role == "leader"


def test_an_unrecognised_adapter_has_no_role():
    assert SerialPort("/dev/ttyACM9", 0x0403, 0x6001).role is None


def test_a_motherboard_port_is_neither_usb_nor_an_arm():
    assert not is_usb(MOTHERBOARD)
    assert MOTHERBOARD.role is None
    assert MOTHERBOARD.usb_id == "-"


def test_usb_id_is_rendered_as_vendor_colon_product():
    assert FOLLOWER_LEFT.usb_id == "2e88:4603"


def test_hub_port_is_extracted_from_the_pyserial_location():
    assert FOLLOWER_LEFT.hub_port == "5.1"
    assert LEADER_RIGHT.hub_port == "4.3"


def test_hub_port_of_a_port_with_no_location_is_marked_unknown():
    assert MOTHERBOARD.hub_port == "?"


def test_grouping_splits_the_four_arms_two_and_two():
    grouped = group_by_role(ALL)
    assert len(grouped["follower"]) == 2
    assert len(grouped["leader"]) == 2
    assert grouped[None] == []


def test_the_two_followers_are_indistinguishable_by_serial():
    """Which is why they must be told apart by hub port, not by /dev/serial/by-id."""
    assert FOLLOWER_LEFT.serial_number == FOLLOWER_RIGHT.serial_number


def test_the_two_leaders_do_have_distinct_serials():
    assert LEADER_LEFT.serial_number != LEADER_RIGHT.serial_number


def test_the_configured_ports_agree_with_what_the_adapters_are():
    """The check that settles the old repo's ports.toml / robot-ports.txt clash."""
    assert role_conflicts(CONFIGURED, ALL) == []


def test_the_contradicting_claim_is_caught():
    """robot-ports.txt said follower.left = ttyACM2, which is a leader adapter."""
    wrong = {**CONFIGURED, "follower_left": "/dev/ttyACM2"}
    (conflict,) = role_conflicts(wrong, ALL)
    assert "follower_left" in conflict
    assert "leader adapter" in conflict
    assert "1a86:55d3" in conflict


def test_every_conflicting_assignment_is_reported_not_just_the_first():
    swapped = {
        "follower_left": "/dev/ttyACM0",
        "follower_right": "/dev/ttyACM2",
        "leader_left": "/dev/ttyACM1",
        "leader_right": "/dev/ttyACM3",
    }
    assert len(role_conflicts(swapped, ALL)) == 4


def test_swapping_sides_within_a_pair_is_not_a_conflict():
    """USB identity cannot see left from right — only unplugging can."""
    sides_swapped = {
        **CONFIGURED,
        "follower_left": "/dev/ttyACM3",
        "follower_right": "/dev/ttyACM1",
    }
    assert role_conflicts(sides_swapped, ALL) == []


def test_an_unknown_adapter_does_not_block_a_run():
    unknown = SerialPort("/dev/ttyACM9", 0x0403, 0x6001)
    assert role_conflicts({"follower_left": "/dev/ttyACM9"}, [unknown]) == []


def test_a_port_that_is_not_present_at_all_is_not_a_conflict():
    assert role_conflicts({"follower_left": "/dev/ttyACM7"}, ALL) == []


def test_describe_names_each_port_its_role_and_its_config_label():
    text = "\n".join(describe(ALL, CONFIGURED))
    assert "/dev/ttyACM1" in text
    assert "2e88:4603" in text
    assert "follower" in text
    assert "[dk1.toml: follower_left]" in text
    assert "does not match" not in text


def test_describe_flags_a_configured_port_whose_adapter_disagrees():
    text = "\n".join(describe(ALL, {**CONFIGURED, "follower_left": "/dev/ttyACM2"}))
    assert "does not match this adapter" in text


def test_describe_works_with_no_config_to_compare_against():
    text = "\n".join(describe(ALL))
    assert "dk1.toml" not in text
    assert "/dev/ttyACM0" in text
