# CRASH.md — the machine froze, and it was the firmware

> **CLOSED 2026-08-27.** A **BIOS update from F6 (2025-12-01) to F8a
> (2026-07-29)** on the Gigabyte Z790 EAGLE AX stopped it. The machine has been
> stable since. Six freezes in three days, none after.
>
> **Nothing in this repository was ever the cause, and no change here fixed it.**
> Read § *The firmware's own verdict* before reopening anything: the platform
> recorded three FATAL Intel SoC internal error records at the moment of the
> crash, which is why nothing ever appeared in the kernel log.
>
> This file stays as the record — the method, the instrumentation, and the six
> hypotheses that turned out wrong. If the machine ever freezes again, start at
> § *How it was found*, not at the top.

## How it was found

Six freezes produced nothing: no journal entry, no panic, no `Xid`, no USB error.
Four things broke that open, in this order, and they are the ones to repeat:

1. **A ping and an SSH session from a second machine.** Every other instrument —
   the log, the telemetry, the PSU sensor, Magic SysRq — reaches you through a
   device on the suspect hardware. A ping does not. It is also what made the NIC
   busy enough to log `NETDEV WATCHDOG`, the first fingerprint of any kind.
2. **`journalctl --list-boots` plus `sar`.** The boot list showed **five**
   freezes, not the two anyone remembered, and the ten-minute CPU history showed
   **two of them happened on an idle machine**. That killed "our workload causes
   it" in a single command.
3. **`/sys/firmware/acpi/tables/data/BERT`.** The answer had been sitting there
   since the second crash. The kernel prints `BERT: Skipped 1 error records` and
   moves on, because it refuses to dump records over 1024 bytes; ours was 3424.
   **Always read this file after an unexplained reset**, before the next one
   overwrites it. Add `bert_print_all` to the kernel command line to have it
   printed automatically.
4. **Decoding the record rather than trusting the summary line.** It named three
   FATAL Intel SoC error records — a hardware fault, below the operating system.

The lesson worth keeping: *the absence of an OS-level error is itself evidence.*
Six clean kernel logs did not mean nothing was wrong; they meant the fault was
underneath the thing writing the logs.

---


**What it was.** Six freezes in three days on an **i9-14900K** / Gigabyte Z790
EAGLE AX. The platform firmware recorded **three FATAL Intel SoC internal error
records** at the moment of each load-related crash (§ *The firmware's own
verdict*) — a fault below the operating system, which is why six kernel logs
were clean.

**What fixed it.** A BIOS update, **F6 -> F8a**. Stable since 2026-08-27. The
CPU was not replaced and no setting in this repository was changed.

**What was never established**, and does not need to be now: whether the
underlying part is degraded silicon (the acknowledged 13th/14th-gen defect, for
which F8a's microcode and power limits are the vendor mitigation) or a firmware
power-management bug F8a happened to fix. If freezes return, that question
becomes live again and § *The machine* is where it starts.

This file is the standing account and the brief for whoever picks it up. The
first section is what a next session should read as its instructions; the rest
is evidence, so nothing gets re-measured.

---

## The brief, as it stood while this was open (historical)

You are picking up an unsolved hard-freeze on the machine that runs this cell.
Read `CLAUDE.md` for the project, then this file. Then:

1. **First message of the session: ask Nikolas for the two things only he can
   set up** (§ *Make it leave a trace*) — the `sudo` sysctls, and **an SSH
   session plus a ping from a second machine**. The second matters more. Magic
   SysRq arrives over a USB keyboard on the same controller as the cameras and
   the CAN adapters, so it cannot tell a dead kernel from a dead xHCI; a ping
   from a laptop can.
2. **Do not run anything that moves the arms without asking Nikolas.** The
   investigation below is designed so that most of it needs no motor at all.
3. **Start with the fault that reproduces, which is not ours** (§ *The plan*
   step 1): the machine does not wake from a blank screen, with none of this
   code running. Everything else in this file needs a freeze that nobody can
   provoke on purpose. The point is to find which subsystem is involved — GPU,
   USB, CAN adapters, or none of them — not to guess at a fix.
4. **Make the next freeze leave evidence** before provoking one (§ *Make it
   leave a trace*). Everything in that section except `sysrq` and `kdump` is
   already implemented and on by default; the kernel-side parts need `sudo` and
   are Nikolas's to run — hand him the commands.
5. When a run does freeze, the first thing to read is
   `uv run dk1 doctor report`, which prints the last second the machine was
   alive, followed by `journalctl -b -1 -k -n 200`.
6. Write what you find **into this file**, dated, including the hypotheses that
   turned out wrong. Half of the value here is not re-testing them.

Two things Nikolas suspects and neither is ruled out: **something in LeRobot**,
and **a thread we share with the desktop (Plasma/KDE)**. Take both seriously,
but note the shape of the problem: ordinary user-space code — Python threads,
CPU load, a busy GUI — cannot normally take a Linux machine down. What can is a
*kernel driver*, and this workload leans on three at once: `nvidia`, `uvcvideo`
(three 1280x720 MJPG streams), and the USB-CAN adapters — the last two on the
**only** xHCI controller on the board, which also carries the keyboard and
mouse. A user-space bug that *triggers* a driver bug looks exactly like this.
The question to answer first is **which driver**.

And note that the machine freezes with **none of this code running at all**
(§ *The freeze that has nothing to do with us*) — which may mean LeRobot and the
cameras were never the story.

**And ask rather than infer.** This file has already had to be corrected once
because a session guessed at which conditions a crash happened under instead of
asking Nikolas. He is at the machine and he remembers; the table above is only
worth as much as its accuracy.

---

## What happened

| when | what was running | Rerun | model / CUDA | how it ended |
| --- | --- | --- | --- | --- |
| 2026-08-25 ~17:58 | `dk1 policy session --study A0`, 8th attempt, dataset recording, SVT-AV1 on the CPU | **no** | yes | machine dead, hard reset |
| 2026-08-26 ~11:23 | `dk1 policy session --study A0`, scene 3 attempt 1, dataset recording, NVENC | **no** | yes | machine dead, hard reset |
| earlier, once or twice | `dk1 teleop`, "after a few minutes" | **yes** | no | machine dead |
| 2026-08-26 | `dk1 teleop` with the Rerun viewer open | **yes** | no | froze within a few minutes: teleoperation never worked at all, everything down, the display showing a stale image |

**Roughly twenty minutes into a session; a few minutes into teleoperation.**

**There is no reproducer.** Earlier versions of this file called teleop-with-Rerun
"the fastest known reproducer" and built the plan on it. Nikolas has since tried
to recreate it deliberately, **with the Rerun viewer open and without it, and the
machine did not freeze either time** (2026-08-26). Every step below that says
"run it until it freezes" therefore has no expected time-to-failure, and a
survival is weak evidence. Read them as *exclusions under load*, not as bisection
against a reliable trigger.

**The one thing that does reproduce is not ours at all:** when the screen blanks
or the machine sleeps, it does not come back (§ *The freeze that has nothing to
do with us*).

### What is common to all four, and what is not

This is the part to reason from, and an earlier version of this file got it
wrong by guessing instead of asking.

| | in the session crashes | in the teleop crashes | common? |
| --- | --- | --- | --- |
| the three USB cameras streaming 1280x720 MJPG | yes | yes (teleop attaches them by default) | **yes** |
| the four USB-CAN adapters, arms energised | yes | yes | **yes** |
| one xHCI controller carrying both | yes | yes | **yes** |
| Plasma/KDE on X11, kernel 7.0.0-30, nvidia 580.173.02 | yes | yes | **yes** |
| the Rerun viewer (wgpu/Vulkan) | **no** | yes | no |
| CUDA, MolmoAct2 resident, NVENC | yes | **no** | no |
| the dataset recorder, PNG cache, video encoding | yes | **no** | no |

So **neither GPU consumer is common** — the sessions had CUDA and no Rerun, the
teleop runs had Rerun and no CUDA. Either the GPU is involved through something
both share (the display stack itself, the driver, the card), or it is not
involved at all and the common factor is elsewhere. What *is* common to every
freeze: **the USB tree** — three 720p MJPG streams and four CDC-ACM CAN adapters
on one xHCI controller — **the desktop**, and **this kernel and driver pair**.

A detail that points the same way: the cameras produce a steady trickle of
`Corrupt JPEG data: N extraneous bytes before marker 0xd7`. That is benign in
itself, and it also means the USB video stream is *marginal* rather than clean.

**Answered 2026-08-26 by Nikolas: yes, the cameras were attached in the teleop
runs that froze.** So the camera row above is confirmed, not inferred.

### What it looked like, and what that rules out (2026-08-26, from Nikolas)

**The fans kept spinning. The screen updated once more — one warning line — and
then nothing: mouse and keyboard moved nothing, and only the reset button
worked.**

That is not a power cut. Power was still being delivered and the machine was
still executing something long enough to paint a line of text. It is the
signature of a **GPU / display-driver wedge**: the last thing the console
manages to print, then a desktop that never repaints. Whether the *kernel* was
also dead is the next question.

So: **the PSU theory is demoted** (keep the +12 V column, it is free, but stop
leading with it).

**But this section previously used the dead keyboard and mouse to promote the
GPU, and that inference does not hold** (see the next section): they are on the
same USB controller as the cameras and the CAN adapters, so a wedged xHCI kills
them exactly as thoroughly as a wedged kernel does. The one thing the account
still establishes is that *something* was executing when the last line was
painted. Which subsystem stopped first is open.

### The freeze that finally left a fingerprint (2026-08-26 15:39)

The first freeze run with the instrumentation on, and the first that named
anything. `dk1 teleop --display --duration 3600`, idle-sleep masked, ping running
from a laptop. **It froze 7 min 30 s in, at the moment Nikolas began actually
moving the arms**, and — new — **the machine rebooted itself**. No hard reset.
There is no hardware watchdog on this board (`/dev/watchdog*` absent,
systemd's `RuntimeWatchdogUSec=0`), so an unattended reboot means the kernel
almost certainly **panicked** — which is what the sysctls set earlier that day
were for.

The sequence, from three independent clocks:

| time | what |
| --- | --- |
| 15:38:53.795 | last telemetry sample written and fsynced. It reads the PSU over **USB** and the GPU via `nvidia-smi` |
| ~15:38:58 | the NIC's transmit queue stops completing (dated backwards from the watchdog's own 5247 ms) |
| 15:39:03 | `r8169 0000:04:00.0 enp4s0: NETDEV WATCHDOG: CPU: 23: transmit queue 0 timed out 5247 ms` |
| 15:39:08 | the same again, 5001 ms, from a different CPU. Then nothing |
| 15:40:13 | the machine boots by itself |

**Three subsystems on two different buses stopped inside fifteen seconds** — the
USB-attached PSU sensor, the PCIe NIC, then the kernel. That is the shape of a
whole-machine collapse, not of one driver failing.

What the telemetry says it was **not**:

- **not power.** 266 W total at the end against a 362 W peak on a 1200 W supply,
  and the +12 V rail held 12.031–12.046 V throughout. **The PSU hypothesis is
  now closed**, not merely demoted.
- **not heat.** GPU 50 °C and 83 W; CPU 72–79 °C; PSU 54 °C.
- **not memory or IO.** 59.2 GB free, IO stall 0.0.
- nothing ramps. The last five samples are indistinguishable from the first.

Two hypotheses this kills:

- **PCIe ASPM.** Tempting, because `r8169 ... can't disable ASPM` appears right
  after the watchdog. But that line appears **at every boot**, and the firmware
  already disables ASPM globally: `ACPI FADT declares the system doesn't support
  PCIe ASPM, so disable it`. The line at 15:39:03 is the driver **resetting the
  NIC** in response to the timeout and re-running its own ASPM check. Symptom of
  the recovery, not a cause.
- **The NIC as the culprit.** `NETDEV WATCHDOG` appears in **no other boot** —
  but no other freeze had a ping running, so the NIC was idle and could not time
  out. It is the instrument that fired, not necessarily the thing that broke.
  Keep the ping running for exactly this reason.

**Our own `logs/*-teleop.log` contributed nothing**: the last line is the third
camera connecting at 15:30:48, because nothing in the teleop loop logs at INFO.
The `.jsonl` did all the work. A once-a-second tick counter in the telemetry
context would say whether the control loop was still turning at 15:38:53.

### The firmware's own verdict: three FATAL SoC errors (2026-08-26)

**The kernel had been hiding the answer at every boot since 2026-08-25.**
`BERT: Total records found: 1` / `Skipped 1 error records` appears at the boot
following three of the freezes; the kernel skips BERT records over 1024 bytes to
avoid flooding the log, and this one is 3424. The record is still readable at
**`/sys/firmware/acpi/tables/data/BERT`** (root only) until the next reset
overwrites it. Decoded:

```
Generic Error Status Block, data_length 3424, error_severity = 1 (FATAL)
  entry 1  Firmware Error Record Reference   FATAL   2592 bytes
  entry 2  Firmware Error Record Reference   FATAL    544 bytes
  entry 3  Firmware Error Record Reference   FATAL     72 bytes
  all three: SOC Firmware Error Record (Type2), rev 2,
             record GUID 8f87f311-c998-4d9e-a0c4-6065518c4f6d  (Intel Crashlog)
```

**Three FATAL Intel SoC internal error records**, written by the platform, with
an opaque Intel-format payload. That is the processor's own crash telemetry,
captured by firmware at a point where Linux was no longer running. It explains
every negative result in this file at once: no journal entry, no panic, no
pstore record, no `Xid`, no `xhci` error, and a machine that resets itself or
sits with its fans spinning. **The operating system never saw the fault because
the operating system was already gone.**

### The machine: an i9-14900K (2026-08-26)

```
Intel(R) Core(TM) i9-14900K      family 6, model 183 (Raptor Lake)
microcode  0x133  (BIOS ships 0x12F; the kernel updates early to 0x133)
Gigabyte Z790 EAGLE AX, BIOS F6 (2025-12-01)
62 GB non-ECC RAM  —  `EDAC ie31200: No ECC support`
```

This is the CPU family covered by Intel's acknowledged 13th/14th-generation
desktop instability defect, and the i9 is its worst-affected part. The reported
signature is what this machine does: **random hard freezes with no operating
system error**, at light load as often as heavy, getting more frequent over
time. Microcode 0x133 is past all of Intel's mitigations (0x125 / 0x129 /
0x12B) — but those **prevent further degradation, they do not repair silicon
that has already degraded**. Intel's remedy for an already-affected part is
replacement, under a warranty they extended to five years.

Two details that fit uncomfortably well:

- **The freezes happen at low CPU load** — 0.57 %, 1.28 %, 4.7 %, and today's
  teleop run. The Vmin-shift failure mode is an *undervolt* instability: the
  part misbehaves when it is asked for little, not when it is asked for much.
  A synthetic all-core stress test may well pass and prove nothing.
- **The RAM is non-ECC**, so a memory error would also be invisible to Linux and
  visible only to firmware. That is the main alternative explanation and it has
  not been excluded.

**This is not proven, and the alternatives are real**: memory instability (an
XMP/EXPO profile), VRM or power delivery, or a board fault would all produce
fatal SoC records too. What *is* proven is that the fault is below the operating
system, and that no amount of work on `dk1lab`, LeRobot, the cameras or the CAN
adapters can fix it.

### The boot records: five freezes, and two of them on an idle machine (2026-08-26)

Read out of `journalctl --list-boots` and the ten-minute `sar` history. **Five of
the last seven boots ended without a clean shutdown**, not the two this file
recorded, and the workload at the time is not what anyone assumed:

| boot ended | CPU over the preceding 20 min | what the journal shows | what was running |
| --- | --- | --- | --- |
| 2026-08-25 13:21:46 | **0.57 %** | stops mid-line, no shutdown | **nothing of ours.** Last `dk1` activity an hour earlier; last journal line is Firefox |
| 2026-08-25 17:20:42 | **1.28 %** | stops mid-line, no shutdown | **nothing of ours** |
| 2026-08-25 17:58:35 | 12–16 % | stops mid-line, no shutdown | the `--study A0` session (the crash this file was opened for) |
| 2026-08-26 11:23:23 | 4.4–4.9 % | stops mid-line, no shutdown | the `--study A0` session |
| 2026-08-26 12:39:41 | **0.25 %** | **`PM: suspend entry (deep)`, then nothing** | nothing of ours — idle auto-suspend |

Three conclusions, and the first one reframes the whole investigation:

- **Our workload is not the common factor.** It is present in two of the five
  and absent in three. A machine that freezes at 0.57 % CPU with no cameras
  open, no CUDA context and no CAN adapters in use is not being taken down by
  LeRobot, the recorder, or the control loop. Those may still *raise the rate* —
  five freezes in two days is more than idle alone would explain — but they
  cannot be the cause.
- **The 2026-08-26 12:39 event is fully explained and is a different fault**:
  logind logged `The system will suspend now!`, the whole sleep sequence ran,
  `nvidia-suspend.service` finished, `PM: suspend entry (deep)` was written, and
  the machine never resumed. **Suspend-to-RAM is broken on this machine.** It is
  what Nikolas describes as "the screen blanks, then later it powers off".
- **The other four attempted no suspend at all** — no `The system will suspend
  now!`, no `PM: suspend entry`. So idle auto-suspend does not explain them, and
  they remain unexplained.

Also checked and clean in all five: no `Xid`, no `NVRM` error (the only NVRM
line is the driver's load banner), no `xhci` error, no USB disconnect, no OOM.
The only `uvcvideo` messages are the harmless focus-control probes emitted when
a camera is opened — **not** an error trickle, and the `Corrupt JPEG data`
warnings this file mentions come from libjpeg in userspace, not the kernel.

### Idle sleep is on, and nothing inhibits it (2026-08-26)

KDE's `powermanagementprofilesrc` on AC: **dim at 5 min, screen off at 10 min,
suspend-to-RAM at 15 min.** Nothing in `dk1lab` calls `systemd-inhibit`, so a
policy session that the operator watches without touching the keyboard will hit
all three timers. Given that suspend-to-RAM does not resume on this machine,
that is a live hazard during any run — independent of whether it caused the four
unexplained freezes.

### The freeze that has nothing to do with us (2026-08-26, from Nikolas)

**When the screen blanks or the machine goes to sleep, it does not come back** —
either the display never wakes or the whole machine never wakes. No cameras, no
CAN adapters, no CUDA, no LeRobot, none of our code. Just the desktop, the
nvidia driver, and power management.

This is the most important thing anyone has said about this fault, for three
reasons:

- it is the **only thing that reproduces on demand**, and it costs minutes;
- it needs **no arms, no permission and no hardware risk**;
- if it is the same fault, then everything else in this file — the USB tree, the
  cameras, the recorder, LeRobot — is a coincidence, and the answer is in the
  nvidia driver's display power management on kernel 7.0 / 580.173.02.

**Tested 2026-08-26, and the cheap version of it came back negative.** With an
SSH session and a ping running from a laptop, `xset dpms force off` blanked the
screen and the keyboard woke it immediately; ping never dropped. So forcing DPMS
off and on is not the fault. The failure needs the real idle path — and the boot
records show what that path does: it suspends, and does not come back.

### The USB tree, as it is wired today (verified 2026-08-26)

There is **one** USB controller on this board — Intel Raptor Lake xHCI at PCI
`00:14.0`. "Bus 001" and "Bus 002" are its USB-2 and USB-3 root hubs, not two
controllers. Everything hangs off it:

```
00:14.0  xHCI  (the only USB controller on the board)
 |
 +- usb2 (5 Gbps)                     +- usb1 (480M / 12M)
 |   hub 2-4  -> camera 2-4.3         |   hub 1-4  -> CAN 1-4.2, CAN 1-4.3   (leaders, 1a86:55d3)
 |            -> camera 2-4.4         |   hub 1-5  -> CAN 1-5.1, CAN 1-5.2   (followers, 2e88:4603)
 |   hub 2-10 -> camera 2-10.1        |   1-6  KEYBOARD      (Logitech K280e)
 |                                    |   1-8  MOUSE         (Logitech M500s)
 |                                    |   1-9.4 Corsair HX1200i PSU monitor
 |                                    |   1-14 Bluetooth
```

Three consequences, and the first one changes how the evidence reads:

- **The keyboard and the mouse are on the same controller as the workload.**
  So "mouse and keyboard moved nothing" is equally well explained by an xHCI
  wedge as by a dead kernel or a wedged GPU, and it does **not** promote the GPU
  the way this file previously argued. Worse, it disarms the SysRq test: Magic
  SysRq arrives over that keyboard, on that controller. **A SysRq that does
  nothing does not prove the kernel is dead.** The test needs an out-of-band
  channel — see § *Make it leave a trace*.
- **The PSU telemetry is USB too** (`1-9.4`, `corsairpsu` hwmon). A `.jsonl`
  that stops does not distinguish a dead machine from a dead USB controller.
- **"Move the cameras to another controller" is not available.** Front panel and
  rear ports are all `00:14.0`; only a PCIe USB card would separate them.
  Confirmed with Nikolas: top camera front, other two rear — already split
  across ports, and it made no difference.

Still worth trying as diagnostics: **cameras at 640x360** instead of 720p (a
quarter of the bandwidth — it is `[capture.policy]` in `dk1.toml`, and it changes
what the policy sees, so it is a test and not a setting to leave), and **USB
autosuspend off** — the three cameras and both USB-3 hubs are currently
`power/control = auto`, the CAN adapters are `on`. A suspend/resume race in
`uvcvideo` is a real class of hang and costs nothing to exclude.

### What the logs say: nothing, and that is a finding

For both crashes the systemd journal for that boot **simply stops** mid-second,
with no shutdown records after it. Checked and absent:

- no OOM kill, no `Killed process`, no `Out of memory` (memory was 8% used);
- no kernel oops, panic, `BUG:` or `WARNING:`;
- no `NVRM`/`Xid` message, no thermal or throttling message;
- no USB disconnect, no `xhci` error;
- no `mce`, no EDAC, no hardware error report.

`sar` was sampling every ten minutes and shows nothing unusual either: CPU 4–8%
average, memory 8%, IO 106 MB/s while the PNG cache was being written and near
zero in the last interval.

A journal that ends mid-line means the kernel never got to write anything to
disk. Combined with what Nikolas saw — fans still running, one last line painted
— the reading is a **hard hang** rather than a power loss: something kept the
kernel from ever reaching the journal again.

### The rig, for whoever is reading this cold

- **RTX 5090**, driver 580.173.02, kernel **7.0.0-30-generic**, Plasma/KDE on X11.
- **Corsair HX1200i** PSU, and it is monitored: `corsairpsu` hwmon exposes total
  watts, the +12 V rail's volts and amps, and PSU temperature. This is the
  single most useful sensor on the machine for this problem, and it is now
  sampled every second.
- 3x Innomaker U30CAM 1280x720 MJPG cameras on one USB controller, 4x USB-CAN
  adapters (followers `2e88:4603`, leaders `1a86:55d3`).
- 62 GB RAM, `powersave` governor, no swap pressure.

---

## Make it leave a trace

**Already implemented and on by default** (2026-08-26), so the next freeze is
not another blank page:

| | |
| --- | --- |
| `logs/<time>-<what>.log` | every log record from `dk1lab` at DEBUG and `lerobot` at INFO, **fsynced line by line**, with tracebacks. `--no-log` turns it off. `dk1lab/logs.py` |
| `logs/<time>-<what>.jsonl` | PSU total power, +12 V volts and amps, PSU temperature, CPU package temperature, GPU temperature/power/utilisation/memory/clock, load, free memory, IO stall — **once a second, each line fsynced**. `--no-telemetry` turns it off. `dk1lab/telemetry.py` |
| `dk1 doctor watch` | the same sampler, standalone, for anything that is not ours: `uv run dk1 doctor watch --label teleop-test` |
| `dk1 doctor report` | reads the newest file back: extremes, whether it ends with a `stop` event, and the last samples. **A file with no `stop` event ends where the machine did** |

The telemetry also carries `episode_start` / `episode_end` events, so a freeze
can be placed against what the operator was doing.

### An out-of-band channel, and why it is not optional

**A second machine with an SSH session open is the most valuable instrument in
this investigation**, and it is free. Every other instrument here — the log, the
telemetry `.jsonl`, the PSU sensor, Magic SysRq — reaches you through a device
that is on the suspect controller, or through a disk the kernel may never write
to again. A ping from a laptop is not.

`sshd` on this machine is socket-activated and listening (`ssh.socket` enabled,
port 22 open); the machine is `134.2.169.74` on `enp4s0`. Before provoking
anything, from the laptop:

```bash
ping -D 134.2.169.74 | tee ~/ping.log     # -D timestamps every reply
ssh nikolas@134.2.169.74                  # and in it: while true; do date; sleep 1; done
```

Then a freeze reads directly, and this is the split the SysRq test was supposed
to give and cannot:

| ping | ssh session | reading |
| --- | --- | --- |
| replies | shell still ticking | the kernel is fine. USB and/or the display stack wedged — and you can run `dmesg`, `ps`, `nvidia-smi` *during* the freeze |
| replies | shell dead | kernel alive, but userspace or IO is blocked |
| stops | dead | the kernel is gone. Only then is this a true lock-up |

If the shell lives, the freeze is diagnosable in real time and everything below
becomes much cheaper.

**Still to do, and it needs `sudo` — hand these to Nikolas.** Note the
differences from the version this file used to carry: `kernel.sysrq` was already
**176** (sync + remount-ro + reboot) so `s` and `b` always worked and only the
`w`/`l` *dumps* were missing; `nmi_watchdog` is on but `hardlockup_panic` and
`softlockup_panic` were **0**, so a wedged CPU printed one line and carried on
being wedged — plausibly the one line Nikolas saw painted. And `sysctl -w` does
not survive a hard reset, which this machine gets, so it goes in a file.

```fish
# fish has no heredocs — this is the fish-safe form
printf '%s\n' \
  '# Freeze investigation — see CRASH.md. Remove when closed.' \
  'kernel.sysrq = 1' \
  'kernel.hardlockup_panic = 1' \
  'kernel.softlockup_panic = 1' \
  'kernel.hung_task_panic = 1' \
  'kernel.hung_task_timeout_secs = 60' \
  'kernel.panic_on_oops = 1' \
  'kernel.panic = 0' \
  | sudo tee /etc/sysctl.d/99-crash-debug.conf
sudo sysctl --system | grep -A8 99-crash-debug
```

`kernel.panic = 0` halts on panic rather than rebooting, so the panic text stays
on the console to photograph; flip it to `30` once pstore is known to capture.

```bash
# Somewhere for a panic to be written. pstore is mounted on this board already.
sudo ls -la /sys/fs/pstore        # a panic would leave dmesg-* here
sudo dmesg | grep -iE 'pstore|erst'
# netconsole to a laptop is the fallback if pstore turns out to be a no-op:
#   sudo modprobe netconsole netconsole=6666@134.2.169.74/,6666@<laptop-ip>/<mac>

# Watch the GPU independently of anything we write, at 1 Hz, to a file:
nvidia-smi --query-gpu=timestamp,temperature.gpu,power.draw,clocks.sm,utilization.gpu \
  --format=csv -l 1 > ~/gpu.csv
```

One caveat on the console: the kernel command line carries `quiet splash`, so a
panic message may never reach the screen. pstore, netconsole or the SSH session
are the channels to trust — not the monitor.

---

## The plan

Reordered 2026-08-26 on two facts that arrived after the first draft: **there is
no reproducer under load** (Nikolas could not recreate the teleop freeze either
way), and **there is one that needs none of our code** — the machine does not
wake from a blank screen. A plan built on "run it until it freezes" cannot work
when nothing reliably freezes, so the reproducible fault goes first and the
load tests become exclusions rather than bisections.

**Before any of it: open the SSH session and the ping from a laptop**
(§ *An out-of-band channel*). Without that, a freeze produces the same blank page
as the last three did. Run `dk1 doctor watch --label <step>` alongside each step
in a second terminal, and read `dk1 doctor report` afterwards either way.

### First, the fault that reproduces

1. **The blank-screen non-wake, instrumented.** No arms, no cameras, no CUDA, no
   permission needed. With ping and SSH live from the laptop, force the display
   off (`xset dpms force off`, and separately `systemctl suspend`), wait, then
   try to wake it. What you learn:
   - **ping replies and the SSH shell responds** -> the kernel is fine and this is
     the nvidia display stack. Grab `dmesg`, `nvidia-smi`, and `Xorg.0.log`
     *while it is wedged* — that is evidence no previous freeze produced.
   - **ping stops** -> a genuine whole-machine lock-up reachable in one minute,
     which is the best experimental handle this investigation could have.
   Either result is worth more than any step below, and it costs ten minutes.
2. **Is it the same fault?** Only step 1's answer tells us. If the blank-screen
   hang leaves the kernel alive and the session freezes do not (or vice versa),
   they are two faults and this file needs to be split. If they look the same,
   stop testing cameras and go to the driver.

### Then, exclusions under load — no expected time-to-failure

Each runs half an hour (teleop substitute) or an hour (session substitute).
Surviving proves little on its own now; freezing proves a lot.

3. **The cameras, alone, with Rerun.** Three 720p MJPG streams read at 30 Hz and
   logged to a Rerun viewer, **no arms, no model, no CUDA**. Write it as a small
   script against `dk1lab.crop.CroppedOpenCVCamera` and `rr.log`; do not use
   `dk1 teleop`, which needs the arms.
4. **The cameras, alone, without Rerun.** Same script, no viewer — separates
   "USB video" from "USB video plus the display stack".
5. **The cameras at 640x360**, and separately **with autosuspend off** — the
   three cameras and both USB-3 hubs are `power/control = auto` today.
6. **The GPU alone.** `dk1 policy run --sim --no-view --duration 1800`: full
   inference at 30 Hz, no camera, no motor, no viewer.
7. **NVENC alone.** Encode the leftover PNG cache in
   `study/rollouts/A0-crashed/images/` in a loop with `h264_nvenc` — far more
   NVENC in ten minutes than a session does in an hour.
8. **The CAN adapters alone.** Arms connected and energised, nothing commanded.
   **Needs Nikolas's permission and his hand near the e-stop.**
9. **Teleoperation with Rerun**, only if 3–8 all survive. Note it did *not*
   freeze when tried deliberately, so this is no longer a reproducer.
10. **A short row on the real cell**, to prove the recorder end to end before
    another nine-attempt session — three attempts, one per scene:

    ```
    dk1 policy session --study TEST --attempts 1 --profile common --duration 60 \
      --record-dataset --dataset-dir study/rollouts/TEST --vcodec libsvtav1
    ```

    Then `dk1 doctor report`, and check the dataset opens:
    `python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset as D; d=D.resume('dk1/test', root='study/rollouts/TEST'); print(d.num_episodes, d.num_frames)"`.
    `--vcodec libsvtav1` is deliberate here: slow, and the only encoder that has
    ever produced readable video on this cell.

Cheap mitigations worth trying *as diagnostics*, one at a time:

- **Cut the cameras to 640x360**, or **turn off USB autosuspend** on them.
  Moving them to another controller is **not available** — there is only one on
  this board (§ *The USB tree*); it would need a PCIe USB card.
- **Cap the GPU:** `sudo nvidia-smi -pl 400`. Demoted by the fans-still-spinning
  observation, but still cheap: if capping makes the freezes stop, the answer is
  power after all.
- **Try a plain session on TTY without Plasma** (Ctrl+Alt+F3, log in, run it).
  The cheapest test of Nikolas's KDE suspicion, and it costs one session.

Things to *record* while doing all this: how long each ran, and what
`dk1 doctor report` said afterwards. Add them to the table below.

| date | experiment | duration | outcome |
| --- | --- | --- | --- |
| 2026-08-26 | teleoperation **with the Rerun viewer open** | a few minutes | **froze.** Teleoperation never worked at all; everything down, display stale |
| 2026-08-26 | deliberate attempt to recreate that freeze, **with and without the Rerun viewer** | — | **did not freeze.** So it is not a reproducer, and the plan was reordered around that |
| 2026-08-26 15:30 | `dk1 teleop --display --duration 3600`, sleep masked, ping from a laptop, telemetry on | **7 min 30 s** | **froze**, at the moment the arms were first moved in earnest — and **rebooted itself**. First freeze to name anything: `r8169 NETDEV WATCHDOG`. § *The freeze that finally left a fingerprint* |
| 2026-08-26 | 3 episodes through the recorder, three 1280x720 cameras, NVENC, **10 GB held on the GPU** by torch to imitate a resident policy | 300 frames each | **saved clean**: 900 frames readable, 3 episodes in the metadata, no leftover frames, no failures. So the encode failure needs something this does not have — real cameras, a real inference load, or 120 s episodes |

---

## The other fault of 2026-08-26, and it is ours

The same session **recorded nothing at all**, and that is separate from the
freeze. The dataset directory holds one episode's PNG cache (7.8 GB), an
unfinished data file, `total_episodes: 0`, and three empty temporary directories
timestamped 11:08:47 — the signature of `save_episode` raising inside the video
encode. Five more attempts were then run and scored against a dataset that stayed
empty.

Three things have been fixed since (all in `dk1lab/dataset.py` and
`dk1lab/cli/policy_cmds.py`):

- the traceback now goes to the session log, where it survives a reset;
- a failed write is announced **in red, after that attempt**, saying the score
  will have no frames behind it — it is not one line among many any more;
- the episode buffer is cleared after a failure, so one bad encode no longer
  poisons every episode after it.

**The cause of that encode failure is still unknown.** Re-running the exact same
encode afterwards, on the same 3 538 real frames with the same encoder settings,
**succeeded in 48.8 s**. So it is not the codec settings and not the frame count.
What differs in the live case: three cameras encoding at once, and a CUDA
context holding MolmoAct2 while NVENC starts. First suspicion for the next
session — **concurrent NVENC sessions while CUDA is resident** — and the log file
will now name the exception.

---

## The starved ticks

Every episode of the 2026-08-26 session reported **7–15 starved ticks** — ticks
where the chunk queue had nothing to serve and the arms held their last command.
Out of ~3 550 ticks that is 0.2–0.4%, and the motion looked fine, but the
comparable measurement in `DIAGNOSTICS.md` § *The fix* is **zero starved ticks
over 335 chunks** — measured **without dataset recording**.

The obvious suspect is the difference between those two runs: the async image
writer, compressing and writing ~90 PNGs a second (~40 MB/s) while the inference
worker wants the CPU. Testable without the arms in the sim, and worth an entry
in `DIAGNOSTICS.md` once measured:

```
dk1 policy run --sim --no-view --duration 300                     # baseline
dk1 policy run --sim --no-view --duration 300 --record-dataset --dataset-dir /tmp/ds1
dk1 policy run --sim --no-view --duration 300 --record-dataset --dataset-dir /tmp/ds2 --stream-video
```

If recording is what starves the queue, `--stream-video` (no PNG cache at all)
is the interesting comparison, and its own cost is measured in
`DIAGNOSTICS.md` § *Recording: the episode that took minutes to save*.

---

## What is already ruled out

Do not spend time on these again without a reason:

- **Memory.** 8% used, no OOM killer entry, 62 GB installed, no swap thrash.
- **Disk space.** 1.3 TB free at the time.
- **The recorder's own durability** as a *cause* — it is a victim. The data-loss
  side of the first crash is fixed and tested (`DIAGNOSTICS.md` § *Recording: the
  crash that ate seven episodes*).
- **The codec settings.** The exact failing encode succeeds when re-run.
- **A power cut / PSU shutdown**, as the *whole* story: the fans kept running
  and the screen painted one more line. Something was still executing.
- **The dataset recorder as a cause of the freeze** — three episodes with NVENC
  and 10 GB of CUDA resident save cleanly (see the table).
- **CPU thermal throttling**, as far as ten-minute `sar` averages can say: 4–8%
  average CPU. Not conclusive at one-second resolution, which is why the
  telemetry now exists.
