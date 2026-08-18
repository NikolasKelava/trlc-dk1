# TRLC-DK1 — bimanual operation and MolmoAct2

This is a fork of [robot-learning-co/trlc-dk1](https://github.com/robot-learning-co/trlc-dk1)
set up to operate a bimanual DK1 cell — two leader arms, two follower arms, three
USB cameras — and to evaluate and fine-tune the MolmoAct2 VLA policy on it.

Everything this fork adds lives in the `dk1lab` package and is reached through a
single `dk1` command. Upstream's files — the CAD, the URDF, the LeRobot plugin
classes in `lerobot_robot_trlc_dk1/`, the motor and impedance stack in
`trlc_dk1_control/` — are left untouched so the fork stays rebaseable.

## Documents

| | |
| --- | --- |
| [setup.md](setup.md) | Clone, install, find the devices. Start here. |
| [configuration.md](configuration.md) | `dk1.toml`: what is in it and how it is written. |
| [safety.md](safety.md) | What causes motion, and what does not. Read before connecting. |

Workflow documents — recording, zero-shot evaluation, fine-tuning, deployment —
land with the commands they describe.

## What exists today

Built and tested:

- `dk1 config show` / `dk1 config check` — inspect and validate the device config
- `dk1 config cameras-arg` — the `--robot.cameras` argument, for raw lerobot commands
- `dk1 find arms` — identify the four serial ports by unplugging each in turn
- `dk1 find cameras` — list the attached cameras against what the config expects
- `bi_dk1_follower_safe` — the bimanual follower with a working joint speed limit

Not built yet: teleoperation, recording, and the MolmoAct2 evaluation path.

## A note on what is verified

This fork inherits a body of reasoning from an earlier attempt at the same
project. Some of it was confirmed on hardware and some of it was never run at
all, and the two were not distinguished. Here they are:

**Verified on hardware.** The LeRobot plugin classes and the DM4310/DM4340 motor,
impedance and gravity-compensation stack. Bimanual teleoperation, with and
without cameras. That the three Innomaker U30CAM-4K cameras all report serial
`20010101` and therefore cannot be addressed by `/dev/v4l/by-id`; that MJPG is
required because YUYV at 720p60 exceeds the UVC bandwidth allocation; that all
three are currently mounted upside down.

**Not verified.** Everything about MolmoAct2 on this robot. No dataset has been
recorded, no fine-tune completed, and no policy has ever driven these arms.
Claims carried over from the earlier attempt are marked `UNVERIFIED` where they
appear.
