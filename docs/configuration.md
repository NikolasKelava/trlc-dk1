# `dk1.toml`

One file records every device this cell uses: the four arm serial ports, the
three camera device nodes, their rotations, and the capture profiles. Nothing
else in the repo hardcodes a port or a `/dev` path.

It is tracked in git, so the values in it are also a record of how this cell was
wired at any point in its history.

```bash
uv run dk1 config show     # what is configured
uv run dk1 config check    # ...and is it all plugged in right now
```

## Sections

### `[arms.follower]` and `[arms.leader]`

One `/dev/ttyACM*` per arm. `ttyACM` numbering is assigned in enumeration order,
so it can change across a reboot or a replug — `dk1 find arms` re-derives it.

Validation rejects a config where two arms share a port. That is always a
discovery mistake, never a valid setup.

### `[cameras.top]`, `[cameras.left]`, `[cameras.right]`

The names are not free choices. The MolmoAct2 BimanualYAM checkpoint's image
keys are `observation.images.{top,left,right}`, LeRobot derives those from the
robot's camera keys, and a mismatch fails the rollout context's visual-feature
check. Validation rejects both a missing name and an unexpected one.

Cameras are addressed by `/dev/v4l/by-path`, which encodes the physical USB hub
port and is stable as long as a camera stays in its socket. The alternatives do
not work here:

- `/dev/videoN` is assigned in enumeration order and moves.
- `/dev/v4l/by-id` is unusable — all three cameras report serial `20010101`, so
  only one of them wins the symlink. Verified: `by-id` lists a single entry for
  three cameras.

`rotation` is per camera, one of 0/90/180/270. All three are currently mounted
upside down, hence 180 throughout.

### `[capture.*]`

Device identity is shared between uses; resolution is not. Teleoperation wants a
big picture to look at, the policy wants the aspect ratio it was trained on, so
each profile sets its own and they cannot drift apart from the device they refer
to.

`fourcc` is `MJPG` everywhere and should stay that way: YUYV at 720p60 needs
about 884 Mb/s and the uvc driver fails to allocate the bandwidth, so reads fail
immediately.

## Writes are surgical

`dk1 find arms` rewrites **only** `[arms.follower]` and `[arms.leader]`.
`dk1 find cameras` will rewrite **only** `[cameras.*]`. Each leaves every other
section, and every comment, byte-identical, and each writes atomically via a
temp file so an interrupted run cannot truncate the config.

This is the single most important invariant in `dk1lab.config`, and it has
dedicated tests in both directions. The previous iteration of this project lost
its entire camera section to a port-discovery run that rewrote the file
wholesale, and every script that depended on it broke at once.

## Validation

`dk1.toml` is validated in full every time it is loaded, and a bad file fails
immediately with a message naming the offending key — rather than surfacing much
later as a camera that opened the wrong device. `dk1 config check` additionally
asserts every configured node exists right now, and reports *all* the missing
ones in one pass rather than stopping at the first.
