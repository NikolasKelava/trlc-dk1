# Safety

Read this before running anything that connects to the follower arms.

## Connecting is not passive

`connect()` on a DK1 follower **energises every motor** and **self-zeroes both
grippers by driving them closed until they stall**. That last part is a
calibration routine — the gripper spins closed, the controller watches for a
torque spike, and sets zero there — and it happens every single time you connect,
before any of your code runs.

So: clear the workspace, keep hands and cables clear of the grippers, and expect
the arms to become stiff and hold position the moment a command connects.

Every `dk1` command that connects says so in its `--help` and warns again on
stderr immediately before it acts. Commands that only read `/dev` — `config
show`, `config check`, `find arms`, `find cameras` — connect to nothing.

## Stopping never moves the arms

Return-to-home is opt-in and off by default, everywhere.

LeRobot's rollout defaults to sweeping the arms back to their startup pose over
several seconds during teardown (`return_to_initial_position` defaults to true).
That is *motion triggered by pressing stop*, which is the opposite of what you
want when you stopped because something was going wrong. This fork defaults it
off. When you stop, the arms stay exactly where they are.

`SafeBiDK1Follower.disconnect()` commands no pose, and there is a test asserting
it never will.

## The joint speed limit

`SafeBiDK1Follower` (`--robot.type=bi_dk1_follower_safe`) enforces a joint
slew-rate limit in **both** control modes.

This matters because upstream's `joint_velocity_scaling` does not. It scales the
velocity argument of `control_Pos_Vel`, which is only reached in `pos_vel` mode.
In `impedance` mode — the mode the bimanual follower runs by default, and
therefore the mode every rollout actually used — `send_action` calls
`DK1Robot.command_joint_pos`, which writes the target straight into the shared
command buffer. The 250 Hz server loop clamps position limits and torque, but
nothing anywhere in that path limits rate. The knob was silently dead.

The limit is expressed in **rad/s**, not per command, so it means the same thing
at 30 Hz behind a policy and at 200 Hz behind teleoperation:

```
--robot.max_joint_rate 0.2     # rad/s, roughly 11 deg/s. The default.
--robot.max_gripper_rate 1.0   # normalised units/s
--robot.max_lag 0.15           # rad the command may lead the measurement
```

Three details are deliberate:

- **The first command after connecting holds position.** There is no previous
  timestamp to measure elapsed time against, so no travel is granted; motion
  begins one control period later. Connecting cannot lurch.
- **The ramp advances from the previous command, not the measured position.**
  Clamping to the measurement deadlocks under stiction — a small error produces
  too little torque to break friction, the arm does not move, and the setpoint
  never advances. Ramping the command lets error, and therefore torque, build
  until the joint breaks free.
- **`max_lag` caps how far the command may lead the measurement.** Without it a
  blocked arm winds the setpoint up arbitrarily far and lunges when freed.

Set `--robot.max_joint_rate` to nothing at your own risk; there is a test
asserting that disabling it really does send raw targets.

## What the limiter is not

It is not an emergency stop. It bounds how fast the arms can move; it does not
decide whether they should. Keep the hardware e-stop in reach whenever a policy
is driving, and remember that a keyboard stop needs a focused terminal, a live
key listener and a responsive loop — all three of which can fail exactly when
you need them.
