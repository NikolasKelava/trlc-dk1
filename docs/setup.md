# Setup

## Install

```bash
git clone https://github.com/NikolasKelava/trlc-dk1.git
cd trlc-dk1
uv venv --python 3.12
uv pip install -e ".[dev]"
```

This installs the upstream LeRobot plugin (`lerobot_robot_trlc_dk1`), the motor
and impedance stack (`trlc_dk1_control`), and this fork's additions (`dk1lab`),
all editable, plus the `dk1` command.

Check it:

```bash
uv run dk1 --help
uv run pytest          # 100+ tests, none of which need hardware
```

## Find the devices

Every device this cell uses is recorded in one file, `dk1.toml`. It is tracked in
the repo, so a fresh clone already has this machine's values — but device nodes
move when things are replugged, so verify before trusting them:

```bash
uv run dk1 config check
```

That validates the file and confirms every configured port and camera is present
right now. It opens nothing and energises nothing.

If an arm has moved:

```bash
uv run dk1 find arms
```

You will be asked to unplug one arm at a time; the command watches which
`/dev/ttyACM*` node disappears. Again, nothing is opened or energised — this only
reads the contents of `/dev`.

If a camera has moved:

```bash
uv run dk1 find cameras
```

This lists the cameras attached right now next to what `dk1.toml` expects, and
exits non-zero if a configured camera is missing.

Each of these rewrites **only its own section** of `dk1.toml`. Running
`find arms` cannot disturb the camera settings, and vice versa. See
[configuration.md](configuration.md).

## Then

Read [safety.md](safety.md) before running anything that connects to the
follower arms. Teleoperation is the next thing to be built.
