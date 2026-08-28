# DIAGNOSTICS.md — the record behind CLAUDE.md

`CLAUDE.md` states what is true now. This file is why: the measurements, the
faults that were chased and closed, the hypotheses that turned out wrong, and
the methodology that produced the numbers. Nothing here is needed to work on
this repo day to day — read it when you are about to re-measure something, undo
something, or repeat an experiment, because most of what looks like an open
question below has already been answered once.

Each section is kept at the length its evidence needs. Where a number decided
something, the number is here; where a hypothesis was discarded, it is here so
it does not get proposed again.

**Order matters.** The rollouts on the arms are in sequence and each one's fault
is diagnosed in the timing or geometry section it points to.

---

## The rollouts on the arms

Six so far, all "pick up the marker/dice", none scored.

### The first rollout, and what was wrong with it

The first `dk1 policy run` ("pick up the marker") moved, reached toward the
marker, and actuated the grippers — but juddered, hesitated, and stalled for
seconds at a time. The log carried both `Record loop is running slower` and
`Indexes diff is not equal to real delay`. Raising `--execution-horizon` to 30
made it smoother and *more* confused.

**The cause was RTC's prefix blending collapsing, and it was our number that
collapsed it.** RTC merges each new chunk with the unexecuted tail of the
previous one, weighting the first `delay` steps at 1.0 — the steps that went by
while the GPU was thinking — then ramping linearly to 0 at `execution_horizon`.
The ramp is the whole mechanism. `DEFAULT_EXECUTION_HORIZON` was **10**, chosen
from the smoke test's 172 ms, which is ~5 ticks at 30 Hz.

But 172 ms is the **`select_action` path**. A rollout runs `predict_action_chunk`
through RTC, which is a different, slower path — measured here at **324 ms**,
almost exactly **10 ticks**. So `delay >= execution_horizon`, the ramp had zero
width, and `get_prefix_weights` degenerated to a step function: ten steps pinned
rigidly to the old chunk, then twenty unconstrained, meeting at a discontinuity
about six times a second. That is the judder, and it is reproducible in one line
against LeRobot's own code (`tests/test_policy.py`).

`--execution-horizon 30` then pins *every* step at weight 1.0, so the new chunk
is dragged onto the old one for its whole length and the policy stops reacting to
what it sees. Smoother and more confused is exactly the predicted symptom.

Three things made it worse:

- **The latency tracker is poisoned by the warmup call.** `LatencyTracker.max()`
  is a running maximum that never decays, and the exclusion for warmup samples is
  gated on `use_torch_compile`, which is `False` here. So the first inference —
  measured at 511 ms with a cold model — set the delay for the entire run at 16
  ticks instead of 10. `dk1lab.policy.prewarm` now runs one inference before the
  RTC thread starts.
- **RTC ran with autograd on for nothing.** Its guidance step runs the action
  expert under `torch.enable_grad()`, so with the parameters still flagged
  trainable every flow step built and kept a full autograd graph. That graph is
  never used: `denoise_step` computes `v_t` *before* calling
  `x_t.requires_grad_(True)`, so the only differentiable path is
  `x1_t = x_t - time * v_t`, an identity in `x_t`, and the recovered correction
  equals the `grad_outputs` handed in. `freeze_for_inference` drops it: 324 ms →
  **272 ms**, action chunks **bit-identical** under a fixed seed.
- **Two serial round-trips per tick.** `SafeBiDK1Follower.send_action` called
  `measured_positions()` for the limiter's anti-windup clamp, a second full read
  of all 12 motors in a tick that had already read them. It now reuses the
  reading from `get_observation()` when it is younger than 15 ms, and falls back
  to a real read otherwise, so a caller that does not observe every tick is
  unaffected.

Measured on this machine, GPU only, no robot involved:

| | |
| --- | --- |
| backbone (VLM) forward alone | 55 ms |
| `predict_action_chunk`, non-RTC, 8 flow steps | 168 ms |
| `predict_action_chunk`, **RTC**, 8 flow steps | 324 ms |
| the same, after `freeze_for_inference` | **271 ms** = 9 ticks at 30 Hz |
| 30 Hz control loop, while RTC infers in a background thread | 29.9 Hz, p95 33.5 ms |

That last row matters: **the control loop is not GIL-starved by inference**, so
the async-inference server in LeRobot's `async` docs addresses a real problem
that this cell does not have. RTC is the in-process answer to the same problem
and it was already the default; it was simply mis-tuned.

`DEFAULT_EXECUTION_HORIZON` is now **20** — a 9-tick delay leaves an 11-step
blend, and 20 also matches the steady-state leftover length (`chunk - delay`), so
the previous chunk is used whole and never zero-padded. `dk1 policy smoke` now
measures the RTC path as well as the sync one and prints the delay it implies,
and both `smoke` and the `run` banner refuse to stay quiet when the blend is
thinner than `MIN_RTC_BLEND_STEPS`. Measuring only the path you are not going to
run is what caused this.

### The second rollout: the judder is gone, the stall is not

Re-run on the arms with the fixes above, plus the home sweep another session
added. Reported: **it judders much less, still stalls, and reacts noticeably
less to the object being moved.** Its log carried two warnings, roughly one pair
per twenty debug lines:

```
lerobot.policies.rtc.action_queue: Indexes diff is not equal to real delay.
                                   indexes_diff=10, real_delay=27
trlc_dk1_control.robot:            No command for 0.50 s — holding last commanded target.
```

Those two lines are the same event from both ends, and together they say the
whole thing. Reading them against `rollout/inference/rtc.py`:

- `real_delay` is **not** the latency tracker. It is `ceil(new_latency / period)`
  for *this* chunk, computed fresh each iteration. `real_delay=27` therefore
  means one chunk cost **27 ticks — 900 ms** of wall time, against the 271 ms
  measured on the bench. Something in situ costs 3.3× the bench number, and
  nothing measured so far explains it.
- `_replace_actions_queue` then discards `real_delay` actions from the front of
  the 30-step chunk, because they describe time that has already passed. 30 − 27
  leaves **three actions**: 100 ms of motion, then 900 ms of nothing. That is the
  stall, structurally — not a hiccup but the steady state.
- `indexes_diff=10` is how many actions the control loop actually consumed while
  the chunk computed. Ten consumed over twenty-seven ticks means **seventeen
  ticks with nothing to send**, which is exactly what the motor chain reports as
  `No command for 0.50 s`.
- And `delay=27` is also passed into `predict_action_chunk`, so RTC's prefix
  weights are 1.0 across the entire execution horizon of 20. The new chunk is
  pinned to the old one for its whole usable length. **That is why it reacts less
  to the object moving** — it is the `--execution-horizon 30` failure mode
  arriving through a different door. Less judder and less reactivity is the
  matching pair of symptoms, and both follow from the one number.

So there is one root cause with two faces, and it is not a tuning constant this
time: **a chunk that takes 900 ms cannot drive a 30-step chunk at 30 Hz.** RTC
needs inference well under `chunk / fps` = 1 s, with room to spare. Band-aids
(trim by consumption instead of wall clock, cap the delay) all amount to
commanding stale targets. The fix is to find the missing 620 ms.

`dk1lab/trace.py` exists to find it, and it is the deliverable of this round
rather than a guess at the answer. It times the three things the RTC thread
does — `predict_action_chunk`, the preprocessor, the postprocessor — and
subtracts them from RTC's own per-chunk figure. The remainder is time the thread
was not running at all. The two outcomes point opposite ways:

| what the trace shows | what it means |
| --- | --- |
| `model_ms` ≈ 900 | the GPU path really is that slow live; look at the model |
| `model_ms` ≈ 270, `other` ≈ 620 | the RTC thread is losing the CPU to the control loop's camera and serial work. No model tuning helps; per-tick work has to come down |

The second is the more likely, and it would also revise the "the control loop is
not GIL-starved by inference" measurement in the table above — that was taken
with no robot attached, so it measured the loop while nothing was decoding three
MJPG streams or talking to two CAN adapters.

**Also still unexplained, and needing the arms:** whether the gripper opens at
the right moment. The direction is still inferred rather than observed changing,
which is why it is now a flag rather than a default.

### The third rollout: smooth, tracking, and misaligned

Run on the arms 2026-08-20, after raising `[limits.policy]` to
`max_joint_rate = 1.0` / `max_lag = 0.4` and defaulting `dk1 policy run` to
`--sync`. Reported by Nikolas, task "pick up the dice":

- **Motion is very smooth.** The judder is gone and stayed gone.
- **It does not really stall any more.** The `--sync` change did what the trace
  predicted it would: with `n_action_steps = 30` the engine executes the whole
  chunk instead of letting RTC discard 27 of every 30 rows.
- **It tracks the dice and moves toward it.** Visual servoing works — the policy
  is grounded on the object and drives the arms at it.
- **It misaligns the gripper with the dice.** This is now the failure.

So the timing faults are closed and the remaining problem is **spatial**, which
is a different class of bug and points somewhere specific: the cameras. The
checkpoint was trained on a top view at 69.4° HFOV and wrist views at 87°
(D435i / D405 in the colleague's sim). Our Innomaker U30CAM-4K match neither.
A policy that has learned "the gripper is aligned when the object appears *here*,
at *this* size" will be systematically off if the lens maps the world onto the
sensor differently than in training — which is exactly a consistent misalignment
on top of correct tracking.

**The next step is therefore to crop the camera images to the trained field of
view**, not to touch the policy. That is now done for the wrist views and is
described in *The camera crop*, below: ours are 105°, comfortably wider than the
87° the wrists were trained at, so the crop is possible. It has not yet driven a
rollout, so it is a built change and not a result.

Two things still not established by this run, and worth stating: nothing was
**scored** (no success/failure count over labelled attempts), and the gripper
inversion was still never *watched* to open and close correctly on the arms.

### The fourth rollout: the crop helped

Run on the arms 2026-08-20 with the wrist crop (87.1°, no inset, no shift) and
`[capture.policy]` still at 640×360. Reported by Nikolas:

- **Alignment is better.** The spatial failure the third rollout ended on is
  reduced, which is the first evidence that the field-of-view mismatch was
  really part of it.
- **The gripper hesitates to close until it is in a good position.** That is a
  *behaviour*, not a fault — and it is the first time the gripper channel has
  been watched doing something sensible on the arms, which the inversion
  question has been waiting for. It is still not a controlled test of the
  inversion, and `--invert-gripper` is still off by default.

Not scored, again. But the crop is now a change with evidence behind it rather
than an argument, and the follow-up work in this round — the retune, the capture
raise, and the corrected `--display-policy-input` — all comes from it.

### The fifth rollout: the freeze becomes visible

Run 2026-08-21 with the crop retune (inset 6, view lifted 40), the capture at
1280x720 and the blocking chunk FIFO. The arms froze for ~310 ms once a second.
That is *The freeze*, below, and it was the last fault this fork could cause.

### The sixth rollout: the loop is clean, and what is left is the policy

Run 2026-08-21 with the async chunk FIFO. 29.9 Hz over 335 chunks, zero starved
ticks, the only blocking tick the cold start, and the home sweep on the arms for
the first time (8.5 s, worst joint 0.028 rad off). Numbers in *The fix*, below.

Nikolas then read the verdict off `--display`'s per-joint panels directly: the
policy's own plan is not a smooth trajectory, and the command and the
measurement follow it. So the roughness is in the model's output, not in
anything between the model and the motor. The right arm was reported not to pick
anything up, which nothing here explains.

---

## Timing: how a chunk reaches the arms

### The stall: what LeRobot does with a chunk, and why `--sync` is now the default

Traced through LeRobot 0.6.1. `interpolation_multiplier` is 1, so there is no
interpolation — one chunk row per tick:

```
RTC thread:   predict_action_chunk -> 30x14 -> postprocessor
              -> ActionQueue.merge: blend with the previous tail,
                 DISCARD real_delay rows from the front
control loop: engine.get_action() -> one 14-D row -> ActionInterpolator
              -> robot_action_processor
              -> SafeBiDK1Follower.send_action   <- the limiter
              -> command_joint_pos -> 250 Hz impedance server
```

**LeRobot does not clip actions anywhere.** The processor pipeline has no clamp
on the action path; absolute joint angles pass through untouched. Everything that
changes the numbers is RTC's blend, our limiter, or upstream's position/torque
clamps. And when the queue is empty, `rollout/strategies/core.py:297` returns
`None` and **nothing is sent at all** — the motor chain holds its last target and
warns after `command_timeout_s = 0.5`. That is the stall, exactly.

The fix is not a tuning constant. `n_action_steps` is 30 on this checkpoint and
`select_action` serves from a 30-deep queue, so the **sync** engine executes the
whole chunk and then blocks for one model call — which is byte-for-byte the
arrangement `sim_eval` uses, and `sim_eval` scores 3/3. RTC, at the measured
900 ms in-situ latency, discarded 27 of every 30 actions and delivered 100 ms of
motion per second. Sync pays one visible pause per chunk and delivers all 30.

`dk1 policy run` therefore defaults to `--sync`. `--rtc` is still there and is
the right answer once in-situ inference is well under `chunk / fps` = 1 s; the
620 ms of unexplained in-situ latency is still unexplained, and `dk1lab/trace.py`
is still the instrument for it. Sync makes it non-blocking rather than solved.

`--duration` now defaults to **180 s**, because the policy is slow: ~30 s before
it does much, 54 s for a successful sim episode. A 30 s rollout cannot succeed.

### The 27.7 Hz loop: the capture raise is innocent, the per-tick engine was not

Two problems were open here. Both are now closed: the trace is fixed, and the
loop's cost is diagnosed **and** fixed in `dk1lab/fifo.py`. Nothing in this round
has been on the arms.

#### B (fixed): `--trace` no longer lies under `--sync`

The old accounting cut a record at the **postprocessor**, and the sync engine
runs the postprocessor every tick — so a 180 s run reported `chunk 4390`, one
per tick, each handed RTC's `real_delay` of 0 as its total. Hence
`0 ms = 0 ticks` and `other -160`: three timers minus a total of nothing.

A chunk boundary under sync is **the tick that actually ran the model**, and
nothing else is. MolmoAct2's `select_action` calls `predict_action_chunk` only
when its own 30-deep queue is empty, so wrapping `predict_action_chunk` — which
`dk1lab/trace.py` already did — is enough to detect one. `RolloutTrace` now
folds sync ticks into a `SyncWindow` between two inference calls and measures
what actually matters at 27.7 Hz: **where each tick goes**.

| | |
| --- | --- |
| `wall_ms` | measured wall clock across the window, not inferred from a delay |
| `ticks` | ticks in it — 30 here: one inference, 29 cached |
| `model_ms` | the one real forward pass, at the head of the window |
| `pre_ms` / `post_ms` / `select_ms` | per **cached** tick, median |
| `outside_ms` | tick period minus the engine call: everything else in the loop |

The inference tick is reported separately as `infer_ms` rather than averaged in,
because it carries the whole forward pass and would hide the per-tick cost the
other 29 are paying. `unaccounted_ms` under sync is wall clock minus **every**
engine call, so it cannot go negative. `total_ticks`, `consumed`, `queue_after`
and `starved` are RTC readings and are simply absent. `trace.close()` cuts the
window still open when the run ends, so the last chunk is not dropped.

Sample, from the real engine and the real cameras with no robot attached:

```
chunk   2    1156 ms over  30 ticks  = 25.9 Hz  (pause 186 ms, model 118)
  per cached tick   33.4 ms = 29.9 Hz  (pre 17.5 · select 5.6 · post 0.2 · loop 10.2)
```

#### A: it is not the capture resolution — measured, paired, three rounds

The suspicion was that raising `[capture.policy]` to 1280x720 added ~5 ms to
every tick, because the sync engine re-runs the whole preprocessor per tick.
**It did not.** The paired A/B — one model load, one process, cameras swapped
between 720p and 360p and back, three rounds, 145 timed cached ticks each:

| | engine per cached tick | `pack_inputs` |
| --- | --- | --- |
| 1280x720 | 22.72 ms | 15.90 ms |
| 640x360 | 21.83 ms | 15.86 ms |
| **difference** | **+0.89 ms** | **+0.04 ms** |

Round-to-round variance is ~1.5 ms, i.e. **larger than the effect**. So
reverting the capture would buy under a millisecond of the ~3 ms needed, and
would cost the policy the thing the raise was for: at 640x360 the tuned wrist
box is 455x256 and 256 rows get upsampled 1.5x into a 378-row model input.
**Do not revert it.** That option is closed.

The 6.1 / 11.0 ms figures this section previously carried are not reproducible
and were measured some other way. Two confounds that produce exactly that kind
of error, both hit while chasing this:

- **A flat-out loop and a paced one disagree about the same function.** Run
  back-to-back, `pack_inputs` costs 7.5 ms at 360p and 14.0 at 720p — resolution
  matters and the effect is large. Paced at 30 Hz, which is what a rollout does,
  both sit at ~15.5 ms. Benchmark the duty cycle you are going to run.
- **Separate processes drift.** Run one resolution per process, the engine reads
  23.2 vs 20.0 ms and the story looks confirmed. Feed both to the *same* loaded
  engine and the gap collapses to 0.89 ms. The 3 ms was between-run variance.

Not the GIL: `thread_time` equals wall time inside `pack_inputs` to within
0.06 ms with all three camera decode threads running, so it is real CPU burn in
the calling thread, not waiting behind the cameras. Not core placement either —
pinning to a P-core changes nothing (the machine is a 14900K on the `powersave`
governor, so this was worth ruling out).

#### A: what it actually is — 22 ms of engine on a 33.3 ms tick, thrown away 29 times in 30

Measured with the real `SyncInferenceEngine`, the real bf16 checkpoint and the
real cameras, paced at 30 Hz, **no robot attached**:

| per cached tick | |
| --- | --- |
| `MolmoAct2PackInputsProcessorStep` | **15.9 ms** |
| the rest of the preprocessor (rename, batch, normalise, clamp, device) | 0.8 ms |
| `select_action` and the engine's own per-tick work | **5.5 ms** |
| postprocessor (clamp, unnormalise, frame transform, device) | 0.2 ms |
| **the inference engine, total** | **~22 ms of a 33.3 ms budget** |

and once per chunk, a **118 ms** model call inside a **~190 ms** engine call.

On 29 of those 30 ticks **every millisecond of it is discarded**:
`select_action` looks at its queue first, finds it non-empty, and returns
`popleft()` without reading the batch at all. So the three camera views are
resized to 378x378, patchified and normalised, and thrown away, thirty times per
chunk instead of once.

Two smaller findings inside that 5.5 ms: `select_action` calls `self.eval()` on
every call, and walking 1737 submodules of a 7B model costs **1.83 ms** each
time; the remainder is `prepare_observation_for_inference`, the `copy`, and
`action.squeeze(0).cpu()` synchronising the device for 14 floats. The
"29 calls in 30 cost ~12 ms" line recorded earlier in this file is a *total* for
29 calls; per call it is ~5.5 ms, and only ~0.4 ms of that is the queue.

The arithmetic against the in-situ 36.1 ms tick: the engine is ~22 ms of it, so
the robot reads, the limiter and the dataset write are ~14 ms. Remove the engine
from the 29 cached ticks and the tick is ~14.5 ms — 30 Hz with room to spare.
Revert the capture instead and it is ~35.2 ms, still 28.4 Hz. Only one of those
two is a fix.

#### The fix: `dk1lab/fifo.py` — the blocking chunk FIFO

`ChunkFIFOInferenceEngine` is what LeRobot's own comment block above
`SyncInferenceEngine` sketches: run the model once, postprocess the whole chunk
at once, queue the rows, hand out one per tick. It is a drop-in for the sync
engine — same `InferenceEngine` lifecycle, same `get_action` contract — and
deliberately **not a subclass**, because the only method it would inherit is the
one it replaces.

**What it is worth**, measured on the real checkpoint and the real cameras,
paced at 30 Hz, no robot, paired in one process with the order alternated:

| | engine cost on a cached tick | the model tick |
| --- | --- | --- |
| `SyncInferenceEngine` | 23.2 ms | ~186 ms |
| `ChunkFIFOInferenceEngine` | **0.02 ms** | ~193 ms |

0.02 ms is a `deque.popleft()`. That takes **23.2 ms out of 29 ticks in every
30**; the in-situ tick of 36.1 ms becomes ~13 ms against a 33.3 ms budget, so
the loop stops being the binding constraint rather than merely clearing the bar.
The once-per-chunk pause is unchanged and is meant to be: it is inherent to
synchronous inference and is what `--rtc` exists to remove.

**The actions are bit-identical.** Thirty rows from each engine, same frozen
observation, same seed, the real bf16 checkpoint: max absolute difference
`0.000e+00` on all 14 channels. That is the equivalence claim tested directly
rather than argued.

Why the equivalence holds, and what it rests on:

- `select_action` slices the same `predict_action_chunk` output to
  `n_action_steps` and pops it in order; this does the same slice, same order.
- All four postprocessor steps — clamp, unnormalise, action-frame transform,
  device move — are stateless and elementwise, so postprocessing a chunk equals
  postprocessing its rows. RTC already does exactly this on the same pipeline.
- The observations dropped were dropped anyway: on a cached tick `select_action`
  never reads the batch.
- It requires **absolute** actions. A relative-action policy re-anchors to the
  current state every call, so a precomputed chunk would drift — upstream's own
  fourth caveat. `dk1lab/fifo.py` **checks** for a relative step in the pipeline
  and raises rather than assuming; ours is absolute joint pose.
- Upstream's other three blockers are SAC (raises from `predict_action_chunk`),
  ACT (ensembler inside `select_action`) and the Diffusion family (obs-history
  queues filled as a side effect). MolmoAct2 has none of them.

A bonus, since `select_action` is no longer on the per-tick path: `self.eval()`
now runs once in `start()` instead of walking 1737 submodules every tick.

Worth noting as corroboration: **`dk1lab/serve.py` has always worked this way** —
it calls `predict_action_chunk`, postprocesses the chunk whole and returns it,
because that is what the `/act` protocol is. That is the server that scored 3/3
in ManiSkill. So the arrangement the FIFO brings to the rollout is the one our
only successful evaluation of this checkpoint already ran under.

**Wiring.** `policy.use_chunk_fifo(ctx)` replaces `ctx.policy.inference` between
`build_context` and `strategy.setup` — `BaseStrategy._init_engine` keeps
whatever it finds there, so that attribute is the whole of the seam and nothing
upstream is touched. It runs **after** `build_context` so `prewarm` has already
built the CUDA graph, and **before** the trace attaches so the trace wraps the
engine that will actually be driven. The pipelines are carried across by
reference, so a gripper inversion already applied to them still applies.

`--fifo` is **on by default** for `dk1 policy run` and `dk1 policy dryrun`, and
is a silent no-op under `--rtc`, which already serves chunks whole. `--no-fifo`
exists to measure the difference on the same rollout, not for ordinary use.

**It ran on the arms, and it did what it promised and no more.** The engine cost
per tick went to `select 0.0`. What it did *not* touch is the pause, and that is
the next section.

The machine's CPU governor stays on `powersave` by request. It was ruled out as
a cause here.

### The freeze: the loop waits for one model call per chunk, and that is the whole warning

Reported from the arms 2026-08-21, with the FIFO on. One trace line and one
LeRobot warning:

```
chunk 144    1297 ms over  30 ticks  = 23.1 Hz  (pause 313 ms, model 220)
  per cached tick   34.1 ms = 29.4 Hz  (pre 32.4 · select 0.0 · post 46.2 · loop 34.0)
WARNING lerobot.rollout.strategies.base: Record loop is running slower (3.4 Hz)
```

**The warning is one tick.** `BaseStrategy.run` measures a single loop iteration
and warns when it exceeds `1/fps`; `1/dt = 3.4 Hz` is a **294 ms tick**, and the
only tick that costs that is the one that ran the model. It fires **once per
chunk**, about once a second, and it is not task-dependent, not the crop and not
the capture resolution. It is also not new — the same warning is in the first
rollout's log, at commit `2fab41c`, before the crop existed.

Confirmed by reproduction rather than by argument: a fake engine that sleeps
220 ms once per 30 ticks and 0 ms otherwise, in an **empty loop with no robot at
all**, prints

```
chunk   0    1268 ms over  30 ticks  = 23.7 Hz  (pause 299 ms, model 220)
  per cached tick   33.4 ms = 29.9 Hz  (pre 32.6 · select 0.0 · post 46.4 · loop 33.4)
```

which is the arms' line to within the robot's own per-tick work. Everything in
it follows from the pause.

**Two of those numbers were the trace lying, and that is now fixed.**
`pre 32.4 · post 46.2` are per *chunk* — the FIFO runs the pipelines once per
chunk, and `RolloutTrace` was keeping the last value it saw and stamping it onto
every tick. 79 ms of pipeline inside a 34 ms tick is impossible on its face.
`loop 34.0` was also not work: it was `period − engine`, which counts
`precise_sleep`. The empty fake loop reports `loop 33.4`. So the cached ticks
were **never** the problem — Nikolas confirmed the run's warnings only ever
quote 3–4.5 Hz, never ~29, which settles it: work per tick was under budget.

**What the pause does to the arms**, all three from the one fact:

- a freeze of ~310 ms, once a second — the stop-and-go;
- the chunk **plays back 23% slow**, 30 rows meant for 1000 ms spread over 1297;
- the policy **re-plans only every 1.3 s**, and each new chunk is anchored on the
  *measured* pose, which lags the commanded one (speed limit plus impedance
  compliance). The longer the gap, the more lag has accumulated, and the bigger
  the correction when the plan lands — with nothing blending the seam. That is
  the rough, fast trajectory change.

Note `post 46 ms`: the postprocessor's last step is `device_processor` to cpu, so
that number was the **CUDA sync**, not CPU work. Model and post are one number,
~266 ms of GPU. There is no waste to reclaim there.

### The fix: `AsyncChunkFIFOInferenceEngine`

Compute the next chunk on a worker thread while the current one is still being
served. `dk1lab/fifo.py` gains a second engine; nothing upstream is touched and
the blocking one stays, behind `--blocking-fifo`, so the two can be compared.

Per tick the control loop publishes its observation, serves one row, and — when
the queue falls to `replan_at` — wakes the worker. When a chunk lands it is
**spliced**: `ceil(latency / period)` rows describe time already spent and are
dropped, the rest replace the queue, cross-faded over `blend` rows.

**Measured on this machine, real bf16 weights, real preprocessor, paced at
30 Hz, synthetic frames, no robot:**

| | loop rate | ticks that ran a model call | starved ticks |
| --- | --- | --- | --- |
| `--blocking-fifo` | 25.8 Hz | 17 in 20 s, worst 201 ms | 0 |
| **async (default)** | **29.7 Hz** | **1** (the cold start) | **0** |

and per chunk, async: `plan 196 ms old = 6 rows dropped, queue 9 -> 24,
blended 4`, one chunk every 15 ticks. So the plan the arms execute is ~200 ms
old instead of the blocking engine's 1300 ms, and the queue never came close to
dry — 9 rows (300 ms) still in hand every time a chunk landed.

Design decisions worth keeping:

- **Drop by wall clock, not by rows consumed.** A chunk is a plan parameterised
  by time, and time is what passed. `ceil` rather than `round` so the first row
  served is never one the arms have already gone past.
- **`replan_at` defaults to 15**, half a chunk. At a 310 ms in-situ latency that
  leaves ~190 ms of margin and lands a plan every ~510 ms. Higher is fresher
  *and* safer (queue at splice is `replan_at − latency_ticks`), at the cost of
  GPU duty and of blending a larger fraction of what gets executed. The ceiling
  is `30 − latency_ticks` ≈ 21, past which it runs back to back like RTC.
- **`blend` defaults to 4** rows, 133 ms. It is RTC's prefix weighting done on
  the postprocessed actions instead of inside the flow model — much simpler, and
  it does not require inference to fit inside `chunk / fps`. It must stay well
  under the replan interval or it becomes the `--execution-horizon 30` failure
  mode by another door.
- **Starvation holds the last row** rather than returning `None`. The arms do
  the same thing either way (the motor chain holds the last target), but this
  keeps it counted and reported instead of only showing up as the chain's
  `No command for 0.50 s`.
- **A persistent worker failure is re-raised on the control thread** after 5
  consecutive errors. Silence would look exactly like a policy that has decided
  to hold still.
- **A reset cold-starts again**, and a generation counter discards a chunk
  computed before it. `reset()` takes `_compute_lock`, so it waits for an
  in-flight model call rather than resetting the policy underneath one.
- **The chunk that arrives past its own last row keeps the previous plan** and
  says so. That is the 900 ms RTC failure, reported rather than executed.

**This is deliberately not action-identical.** The blocking FIFO was a pure speed
change and was bit-identical to the sync engine; this re-plans four to five times
more often, drops rows and blends. That is the point.

**Wiring.** `policy.use_chunk_fifo(ctx, asynchronous=, replan_at=, blend=, fps=)`
replaces `ctx.policy.inference` between `build_context` and `strategy.setup` —
`BaseStrategy._init_engine` keeps whatever it finds there, so that attribute is
the whole of the seam. It runs **after** `build_context` so `prewarm` has already
built the CUDA graph, and **before** the trace attaches. Pipelines are carried
across by reference, so a gripper inversion already applied still applies.

**It ran on the arms 2026-08-21, and the defaults were sized right.** The
summary over 335 chunks:

```
loop rate     29.9 Hz (target 30)     median tick 33.4 ms, p95 36.0, worst 196.4 (cold start)
  engine      0.01 ms    robot read+send 0.4 ms    idle 33.0 ms
median chunk  212 ms (pre 41 · model 169 · post 0)
  plan age    212 ms on arrival = 7 rows dropped
  queue       8 rows in hand at the splice, 23 after, 4 blended
  starved     0 ticks
gripper       policy commanded +0.033 .. +1.000
```

Three things to keep from that. **In-situ latency is 212 ms, not the ~310 ms the
blocking run implied** — that figure included the pipelines being charged to the
wrong place. **`robot read+send` is 0.4 ms**, so the control loop was never the
constraint and the old `loop 34.0` really was almost all `precise_sleep`, as the
rebuilt trace claimed. And the **policy commands the gripper across almost its
whole range**, which is the first direct evidence on this cell that it uses the
channel at all.

`replan_at = 15` left 8 rows (270 ms) in hand at every splice. There is room to
raise it if fresher plans are ever wanted; nothing yet says they are.

---

## Geometry: the lens, the crop and the model's input

### The camera crop: our lens is 105 degrees, the checkpoint's wrists are 87

Built 2026-08-20 in response to the third rollout's spatial failure. Nothing has
been run on the arms with it yet.

**The two numbers.** Ours is **105° HFOV / 116° diagonal**, from the Innomaker
U30CAM-4K-S1 user manual — a 2.25 mm f/2.0 M12 lens on a 1/2.8" IMX415. The
model string is confirmed off the USB descriptor here (`0bda:5883
Innomaker-U30CAM-4K-S1`), so it is the right datasheet, but the angle is the
manufacturer's figure and not a measurement. Theirs is in
`sim_eval/robots/bimanual_yam.py` on the colleague's `sim-eval-dk1` branch, and
it is not a lens spec at all — the sim *builds its intrinsics from an angle*:
`fx = (w/2)/tan(hfov/2)`, `fy = fx`, top 69.4° (D435i), wrists 87.0° (D405). So
the trained geometry is exactly a pinhole at those angles, which is what makes
the correction well-posed.

**The correction.** On a pinhole, the fraction of the frame width that spans the
target is `tan(target/2) / tan(source/2)` = `tan(43.5°)/tan(52.5°)` = **0.7282**.
Every rounding takes the *larger* box, at Nikolas's stated preference: too much
field of view degrades more gracefully than too little.

**On top of that sit three hand-tuned adjustments**, added after the fourth
rollout: `crop_inset` (extra pixels off the left and right edges, with top and
bottom following so the box keeps the frame's aspect ratio — an anisotropic box
would be stretched on the way back out, which is the exact distortion this is
undoing), and `crop_shift_x` / `crop_shift_y`. Currently **inset 6, shift_y −40**,
i.e. the view is lifted. At 1280×720 that is the box **909×511 at (185, 24)** =
**85.6° H / 55.0° V**, sitting 80 px above centre — and note the `y` of 24, which
means about −64 is as far as this box can be raised before the shift clamps.
Read the box off `dk1 config show` rather than from here; restating it by hand
has gone stale twice.

**All three are quoted in pixels at a 640-wide reference** (`fov.REFERENCE_WIDTH`)
and scaled to whatever frame the camera delivers. They are eyeballed on a
picture, so pixels are the natural unit — but a pixel is a different angle at
every capture resolution and this cell runs two. Scaling from one reference is
what keeps teleop and rollout geometrically identical, which is the only thing
that makes "it looked right in teleop" evidence about what the policy gets. It
also means changing the capture resolution does not silently retune the crop.
A shift that would run off the sensor is **clamped, not raised** — retuning a
number beats refusing to produce a picture mid-rollout — and `fov.describe`
prints `CLAMPED` with the shift it actually achieved, because reporting the
number that was *asked* for would be a lie in exactly the case an operator most
needs to know about.

**The model input is 378×378, not 224×224.** This was wrong here for a day and
it matters. The 224s in the checkpoint's `molmoact2_masked_normalizer` are
declared feature shapes under `VISUAL: IDENTITY` — they normalise nothing and
resize nothing. The real resize is in the HF image processor
(`processor_config.json`: `crop_mode: "resize"`, `size: 378×378`,
`patch_size: 14`, `resample: 2`), and the packed tensor proves it: `pixel_values`
comes out `[3, 729, 588]` = three images of 27×27 patches of 14×14×3 = **378×378**.
It is a *stretch*, not a letterbox — 16:9 becomes 1:1 — which is what training
did too, so it is consistent.

**So the crop only costs no detail if it clears 378 in both axes**, and at
`[capture.policy]` 640×360 it did not: the tuned wrist box was 455×256, and 256
rows were being upsampled 1.5× to fill a 378-row input. That is why the policy
capture is now **1280×720** — see *The capture resolution*, below.

**Where it lives, and why there.** `dk1lab/fov.py` is the arithmetic (no lerobot
import, like every other decision module here); `dk1lab/crop.py` is
`CroppedOpenCVCamera`, an `OpenCVCamera` subclass whose only addition is a crop
and resize at the end of `_postprocess_image`. In the *camera*, not in a LeRobot
processor and not in `dk1lab/policy.py`, because the crop has to be true of every
image this cell produces: a processor step covers rollout and misses teleop, and
a `policy.py` step covers rollout and misses **recording**, which is the one that
would quietly poison a Phase 4 fine-tune. Frame size is unchanged, so no feature
shape, no capture profile and no test downstream had to move.

Two mechanical notes worth keeping. The crop runs *after* the base class's
rotation, so the box is centred on the picture as finally seen — right at 0° and
180°, meaningless at 90°/270°, where the output's width is the sensor's vertical
axis; that combination is rejected in `config.load` and again in the config
dataclass rather than cropped wrongly. And registering
`CroppedOpenCVCameraConfig` under `type: opencv_cropped` is not enough on its
own: `make_cameras_from_configs` has a hardcoded branch per built-in type and
falls through to `make_device_from_device_class`, which looks the *class* up by
name in the package holding the config's module. Same trap as
`SafeBiDK1Follower`, same fix — `dk1lab/__init__.py`'s lazy `__getattr__`.

**Checked on the real cameras** (video devices only; no arms, nothing energised):
all three connect and deliver 1280×720, the wrists report the 909×511 box, and
stills through the cropped path are upright, correctly framed and visibly
tighter. One thing the stills showed that the datasheet does not: the uncropped
105° frame has **black vignette corners** — the lens's image circle does not
quite cover the sensor — and the crop removes them entirely.

Checked end to end as well, on the same read-only path: real frames from all
three cameras pushed through the **real** preprocessor come out as three upright,
correctly ordered 378×378 tensors. That is the check `--display-policy-input`
now does live — see *Watching what the model sees*.

**What this does not fix.** The lens has real barrel distortion (the manual
specifies TV distortion < −6.2%, and it is obvious in the stills: straight edges
bow). The crop is a pinhole correction, so it matches the trained geometry at the
centre of the frame and only approximately at the edges; undistorting properly
needs a calibration this cell does not have. Mounting pose is untouched too —
the sim's wrist cameras sit at a specific place on `link_6` and ours sit where
they sit. So this narrows one known divergence and leaves two.

### The capture resolution: 378 rows in, so more than 378 rows out

`[capture.policy]` was raised from **640×360 to 1280×720** on 2026-08-20, once
the 378×378 model input was established. The arithmetic is the whole argument:

| | wrist crop | rows the model wants | verdict |
| --- | --- | --- | --- |
| 640×360 | 455×256 | 378 | **1.5× upsample** — detail the sensor never caught |
| **1280×720** | **909×511** | 378 | genuine downscale |

The top view goes 720 → 378 likewise, where it was 360 → 378 before.

**It costs nothing on the control loop, and that was measured, not assumed.**
All three cameras sustain **30.3 fps** at 1280×720 MJPG; `OpenCVCamera` decodes
on a **per-camera background thread**, so `async_read` picks up the latest frame
rather than paying for the decode; and our crop-and-resize costs **1.6 ms per
frame** on that thread (0.22 ms at 640×360). Read latency measured from the loop
is frame-rate bound and identical at both sizes — median 32–33 ms against a
33.3 ms budget either way. `dk1 config check --formats` confirms all three
cameras advertise the mode.

One consequence worth noting: cropping the **top** view to its trained 69.4° is
now arithmetically possible — it would keep 683×384, still above 378 — where at
640×360 it kept 341×192 and was not. It is left uncropped by request, but it has
moved from "impossible" to "an experiment we could run".

---

## The speed caps, the home sweep, and the simulator

### What the caps were doing, and the numbers that raised them

Measured 2026-08-20 by recording a 120 s sim episode of this checkpoint and
replaying its commanded joint targets through the **real** `SlewLimiter`.

The policy's motion is **bursty, not fast**: median demand 0.036 rad/s, p95 0.31,
peak 4.56. So the old 0.3 rad/s cap did not slow everything evenly — it truncated
exactly the reach and transit moves and left the slow parts alone. 30% of ticks
had at least one joint demanding more than the cap.

Replayed through the limiter (best case: a perfectly tracking arm, so `max_lag`
never binds — real hardware is worse):

| cap | worst joint ends up behind the policy's intent | ticks >0.1 rad behind |
| --- | --- | --- |
| 0.3 rad/s (old) | **0.98 rad = 56°** | 25.6% |
| **1.0 rad/s (now)** | 0.40 rad = 23° | 3.3% |
| 2.0 rad/s | 0.10 rad | 0% |

**`max_lag` was the worse of the two, and it is not a position clamp — it is a
torque clamp.** Impedance torque is `arm_kp * (q_des − q)` with
`arm_kp = [100, 100, 100, 20, 20, 10]`, so `max_lag = 0.1` held the PD torque to
10 Nm on j1–j3 (motor limit 28), 2 Nm on j4/j5 (limit 10) and **1 Nm on j6**
(limit 10) — a tenth of the wrist's authority, on the joints the policy drives
fastest. And because `limiter.limit` writes the *clamped* value back into
`_prev_cmd`, the command can never build more lead: a joint that cannot break
stiction within 0.1 rad stalls there silently and permanently. That is precisely
the deadlock `dk1lab/limiter.py`'s own docstring argues against when it explains
why the ramp anchors to the previous command rather than to the measurement.
Now **0.4 rad**. `joint_torque_limits` downstream is the thing that should be
bounding torque, and it still is.

Note the earlier "±1.8 rad excursion ÷ 0.3 rad/s = six seconds" argument was
wrong in kind: ±1.8 rad was a *position range*, not a per-tick demand. Total
travel on the busiest joint (17.9 rad over 120 s) averages 0.15 rad/s, under even
the old cap. The cap's damage was to the peaks, not to the average.

### The home sweep speed: 0.3 rad/s, eased at both ends

Reported from the arms 2026-08-21: the sweep follows a smooth trajectory but is
**too fast**. Two separate things were wrong, and both are fixed in
`dk1lab/home.py`. **The fixed sweep has since run on the arms** and is the one
in use.

**It was running at the policy's speed cap, which is not a speed.** `sweep_to_home`
and `policy._home_rate` both read `[limits.policy].max_joint_rate` and used it as
the sweep rate. That cap is an *upper bound* on what a policy nobody trusts may
do; reading an upper bound as an instruction meant the 2026-08-20 raise from 0.3
to 1.0 rad/s tripled the speed of homing as a side effect, without anyone
deciding to. `home.home_rate(cap)` now returns `min(DEFAULT_HOME_RATE, cap)` —
0.3 rad/s, or the cap when the cap is tighter, because commanding faster than
the limiter allows only means the limiter clamps it and then the two ramps
disagree about what was commanded. `false` (no cap at all) gives 0.3, not
"any speed you like".

**It was one constant rate, so it started and stopped with a step change in
velocity.** `home.ease_scale` scales the per-tick step by the smaller of two
smoothsteps: one in *time* since the start (`DEFAULT_EASE_IN_S` = 0.75 s), one in
*distance still to go* (`DEFAULT_EASE_OUT_RAD` = 0.25 rad). Taking the minimum is
what lets a sweep shorter than both simply never reach full speed instead of the
two ramps fighting.

Three details worth keeping:

- **The ease-out reads the distance left in *command* space**, not against the
  measurement. The ramp is driven from the previous command, so that is the
  quantity that actually reaches zero; the measurement lags it and would hold the
  profile at its floor for the whole settling time.
- **The profile never reaches zero** — `DEFAULT_EASE_FLOOR` = 0.2 of full rate.
  A profile that did would never leave the start and would only approach home
  asymptotically, timing out just short of the tolerance.
- **The timeout accounts for the easing** (`home.ease_overhead`, generously
  rounded up), or it would cut off sweeps that were going to arrive.

Net effect, simulated against the same fake arms the tests use: a 2 rad sweep
went 2.0 s at 1.0 rad/s flat out, and now takes 7.5 s — **3.7x slower**, peaking
at 0.3 rad/s, starting and ending at ~0.06 rad/s with no step change anywhere in
the profile. Short sweeps are slower still relative to before (a 0.2 rad move is
7x) because they never leave the eased region.

The gripper is deliberately **not** eased or slowed: it is at its home value
already in the ordinary case, and a slow gripper is not the hazard.

`dk1 policy home --max-joint-rate 0.6` still names a peak explicitly and is
honoured as given — `sweep_to_home` gained a `rate` argument for it. That flag's
help used to say "sweep speed" while setting the limiter, which after this change
would have been a lie in the one direction an operator would notice.

### The sim run: the policy behaves the same way with every hardware excuse removed

`sai-prasanna/molmoact2`'s `sim_eval` is a **pure HTTP client** — it posts three
camera frames, a 14-D state and an instruction to an `/act` endpoint and executes
the chunk that comes back. It never imports a model. So `dk1lab/serve.py` answers
that endpoint with **the exact checkpoint the arms run**, through the same
LeRobot pipelines, and `sim_eval` is used byte-identical to upstream.

Clone at `~/Documents/RobotLearning/molmoact2` (scratch, not tracked here). Two
forced deviations, both recorded there: `torch` repinned from `2.5.1+cu121` to
`>=2.7+cu128`, because the 5090 is sm_120 and the pinned wheels have no kernel
image for it; and ManiSkill's YCB asset pack downloaded separately
(`python -m mani_skill.utils.download_asset ycb`), which the sim_eval README does
not mention.

Two structural differences from a rollout, both deliberate and both *in our
favour* for this experiment:

- **No RTC.** `sim_eval` blocks on each response and executes the whole 30-step
  chunk, so there is no deadline, no latency to compensate, no queue to starve
  and no seam to blend. Read as a control for the arms' timing faults at first;
  it turned out to be the *arrangement that works*, and `dk1 policy run` now
  defaults to the same thing (`--sync`). See *The stall*.
- **No gripper inversion.** The wire protocol *is* the YAM convention —
  `sim_eval`'s own `yam_state_adapter` sends `grip in [0,1] (1=open)` and
  `yam_action_adapter` maps `1.0` back to ManiSkill's `-1.0` = open. That is a
  **third** independent statement of the convention, and it means the sim tests
  the policy with the DK1's gripper sign out of the picture entirely.

`BimanualYAMPutEverythingInBox-v1`, "put everything into the box".

**The first day's answer was 0/10, and it was wrong — the episodes were too
short.** 800 steps is 27 s at 30 Hz. The policy barely acts for the first ~30 s,
so every one of those episodes was scored before the policy had started. Also
0/3 at `--n-action-steps` 10 and 5, which ruled out the open-loop chunk length
and made the too-short-episode explanation look less likely than it was.

**With 3600 steps (120 s), 2026-08-20:**

| server | checkpoint | result |
| --- | --- | --- |
| `dk1 policy serve` | our LeRobot bf16 copy | **3/3** |
| `examples/yam/host_server_yam.py` | `allenai/MolmoAct2-BimanualYAM`, HF | ~50% |

Successful episodes averaged **1627 steps = 54 s**. So the policy does the task
zero-shot on its own embodiment, and our packaging is not subtly wrong — it beat
the reference path on the same task, budget and seed. That closes the one thing
the previous round left open. Episode outcome is stochastic (flow-matching
sampler, and the server holds its own RNG), so treat 3/3 vs ~50% as "ours is at
least as good", not as a measured margin: a later single 3600-step episode at
seed 42 recorded here did *not* succeed.

Supporting numbers from a recorded 120 s episode (state and action logged every
tick, correct 16-D `qpos` mapping — the sim interleaves left/right, `qpos[2k]`
is left joint k+1 and `qpos[2k+1]` is right):

| | |
| --- | --- |
| tracking, median \|commanded − measured\| | **0.0018 rad** (p95 0.025) |
| travel per arm joint, total | 5.2 – 17.9 rad |
| demanded joint rate | median **0.036** rad/s, p95 **0.31**, max **4.56** |
| gripper commanded | left −1.00 → +0.94, right −1.00 → +0.70 |

The simulator executes what the policy asks to within a couple of milliradians,
and the policy opens and closes both grippers. The camera views were checked by
eye and are well-formed.

**The gripper convention now has a fourth, behavioural confirmation.** The sim
succeeds while mapping the policy's `1.0` to *open*. So the checkpoint really
does speak YAM (1=open) and the DK1 really is 0=open — which means
`--invert-gripper` is very likely **required** on hardware, and the current
default of off is very likely wrong. Still not watched on the arms; still a flag.

The colleague's `sim-eval-dk1` branch turns out to carry a **full DK1 in sim** —
`bimanual_dk1.urdf`, the meshes, `robots/bimanual_dk1.py`, a DK1 task and DK1
adapters — so running our own embodiment in sim is a checkout away, not a build.
That is the agreed next step after YAM.

---

## The instruments

### Watching a rollout: `--trace` and `--display-policy-input`

Both added this round, both read-only.

`--trace` (default **on** for `dk1 policy run`) prints one line per chunk — not
per tick — with the cost breakdown, the consumed count, the queue depth after
the merge, and the starved ticks; then the policy's **own** action row next to
the same row after the postprocessor. Those two vectors are kept separate on
purpose: a question about the policy cannot be answered with a vector LeRobot
rewrote. At the end it prints a summary that states, in words, whether the queue
ran dry, whether inference is too slow for RTC to leave anything, where the
unaccounted milliseconds are, and whether the policy ever moved a gripper.
`RolloutTrace` works under `--sync` too, and on its own terms: a record is the
window between two real inference calls, its total is measured wall clock, its
costs are **per tick**, and the queue readings are *absent* rather than zero.
Cutting a record at the postprocessor — which is what it used to do — cut one
per tick and reported nonsense; see *The 27.7 Hz loop*.

`--display-policy-input` opens Rerun and logs, under `policy_input/`, the images
**as the model receives them** — the preprocessor's output, not the robot-side
view `--display` shows. Teleop already confirmed the robot-side view is upright
and correctly named; that was never the open question. What was unverified is
everything the policy pipeline does after that. Available on `dryrun` as well,
which is where to check it: arms energised, nothing sent.

Both attach by wrapping, never by replacing: `_TimedPipeline` proxies the
pipeline objects (forwarding `reset()`, `.steps` and everything else), and the
queue's `merge` is wrapped on the instance. `attach` runs after `build_context`
so `prewarm`'s cold call is not counted as a chunk; `attach_queue` runs after
`strategy.setup`, because the RTC queue does not exist until `engine.start()`.

### Watching what the model sees: `--display-policy-input`, corrected

The flag existed before this round and was showing the wrong thing. It logged
the `observation.images.*` entries of the **preprocessor's output** — but
`molmoact2_pack_inputs` leaves those **untouched**, at the camera's own size and
dtype, and puts what the model consumes in `pixel_values`. So the panel showed a
picture that looked like the model's input, was captioned as the model's input,
and was in fact the robot-side view at a different resolution and aspect ratio —
the one thing `--display` already showed. It could not have caught a resize or
aspect bug, which is most of what it was for.

`dk1lab.trace.model_input_images` now reconstructs the real thing: un-patchify
`pixel_values` (27×27 patches of 14×14×3 → 378×378), undo the `mean 0.5 /
std 0.5` normalisation, scale to bytes. Camera names come from
`layout.IMAGE_KEYS`, which is the order the checkpoint's preprocessor pins, so
row *i* really is that camera. It returns `{}` rather than raising on anything
unexpected — a display must never take a rollout down.

`dk1 policy dryrun --display-policy-input` is the way to check camera
orientation: arms energised, nothing sent, and Rerun carries both the robot-side
view under `--display` and the model's own 378×378 under `policy_input/`.
Verified here against the live cameras: upright, correctly ordered, correctly
stretched.

**`dk1 teleop --display-policy-input` does the same thing while you drive**,
which is the better place for it: you can move a wrist by hand and watch the
model's view track. `dk1lab/modelview.py`:

- It builds the checkpoint's **preprocessor only** — `make_pre_post_processors`
  reads `policy_preprocessor.json` and the HF image processor and never touches
  the 10 GiB of weights. 0.6 s on the CPU, no GPU.
- It **attaches by wrapping**, like `dk1lab/trace.py`: `ModelInputProbe` proxies
  the observation processor `teleop_loop` already calls, returns its result
  untouched, and logs as a side effect. The loop stays upstream's.
- The work runs on a **background thread**. One pass costs ~11 ms and the 60 Hz
  budget is 16.7, so inline sampling would drop ticks; measured with the thread,
  the worst tick is 5.0 ms and the median 1.6 ms. One tick in 12 is sampled and
  anything arriving while the worker is busy is **dropped, not queued**.
- It **pins the Rerun layout**, and that is not optional. `log_rerun_data` builds
  a blueprint from the first observation it sees and caches it on itself; that
  blueprint is an explicit grid, so `policy_input/*` — which never passes through
  it — would get no view and be invisible until the operator built one by hand.
  Filling the cache before the loop starts is what stops that, because
  `_ensure_blueprint` returns early when it is already set. A coupling to an
  upstream implementation detail, written down as one.

### Watching the chain: `--display` now draws three lines per joint

Built 2026-08-21 after the async FIFO ran and the motion was still rough. The
console's two rows are **in different units**, and reading them as a pair is the
obvious mistake — `policy +0.977` and `robot +2.208` are one number written
twice. `policy` is the flow head's output in the checkpoint's normalised space,
nominally [-1, +1]; `robot` is the same row after `clamp_action`, the quantile
unnormaliser and the gripper transform. Both rows are now labelled with their
units. And after a splice the `robot` row is the **cross-faded** one, so it is
only a matched pair up to the blend.

`dk1lab/actionview.py` is the clean comparison. `dk1 policy run --display` logs,
per tick and on one axis per joint:

| | |
| --- | --- |
| `policy/<key>` | the model's own row for this tick, in robot units, **before** the cross-fade and the limiter |
| `command/<key>` | what `send_action` returned — after both. `SafeBiDK1Follower` returns the *limited* action, which is why that is the number logged |
| `observation.<key>` | where the joint really is, from LeRobot as always |

Which makes rough motion attributable without another rollout: `policy` rough
and `command` following it is the plans; `policy` smooth and `command` stepping
or lagging is ours (blend or limiter); `command` smooth and `observation` rough
is the arm. `AsyncChunkFIFOInferenceEngine` keeps a second, **unblended** queue
(`_plan`) purely so the first of those three exists.

**It pins the Rerun layout**, for the same reason `dk1lab/modelview.py` does and
with the same coupling written down: `log_rerun_data` caches a blueprint off its
first observation, and anything not in it is invisible. The layout is one
`TimeSeriesView` per joint, seven across, so the top row is the left arm and the
bottom the right — LeRobot's default puts all fourteen observations in one panel
and all fourteen actions in another, at different scales, which is why the
comparison has never been made. `--display-policy-input` composes: its 378x378
panels are added to the same blueprint rather than replacing it.

Costs **0.26 ms per tick** (median, real Rerun, 28 `rr.log` calls, measured
here), against a 33.3 ms budget.

### The trace, rebuilt for the FIFO

Three changes, all from the two numbers that were wrong above:

- **The FIFO reports its own chunks.** `dk1lab.fifo.ChunkReport` is measured
  inside the engine — on the thread that pays for it, which under the async
  engine is not the control loop — and carries what only the engine knows: plan
  age on arrival, rows dropped, queue depth before and after the splice, rows
  blended, ticks starved. `RolloutTrace.attach` hooks `engine.on_chunk` (detected
  by `hasattr`, not `isinstance`) and pairs each report with the **ticks** that
  ran while it was computing.
- **Robot work is timed separately from sleep.** `attach` wraps
  `robot_wrapper.get_observation` and `send_action`, so a tick reads
  `robot 4.2 · wait 29.2` instead of one `loop 33.4` that means nothing. A loop
  with 29 ms of headroom and one 29 ms over budget used to print the same line.
- **The worst tick is reported, and counted.** A blocking model call is one tick
  in thirty: it moves neither the median nor the p95, so quoting only those
  reports a stop-and-go loop as a smooth one. `blocking_ticks` counts ticks whose
  *engine call* exceeded the budget — one is the async cold start and is fine,
  more than one is the freeze — so the verdict fires on the right thing.

The plain-sync path (`--no-fifo`) also had the stale-timer bug and now clears
`_pre_ms` / `_post_ms` per tick alongside `_model_ms`.

---

## Recording: the crash that ate seven episodes

**2026-08-26.** The machine froze during A0's eighth attempt, at about 17:58 on
2026-08-25, with seven attempts recorded and scored. The journal for that boot
simply stops — no OOM kill, no panic, no shutdown — so the freeze itself has no
software cause on record. What it exposed is entirely ours to fix.

### What survived, and what did not

| | |
| --- | --- |
| `study/scores/A0.csv` | **intact.** Seven rows, written as each attempt ended |
| the three `.mp4` streams | **intact and readable.** 24 878 frames, exactly the frame count in `info.json` |
| `data/chunk-000/file-000.parquet` — the 14-D state and action of every frame | **unreadable.** `Parquet magic bytes not found in footer` |
| `meta/episodes/` — the per-episode index | **absent** |
| `images/observation.images.*/episode-000007/` | 4.6 GB of PNGs of the attempt that was in progress |

Without `meta/episodes/` the dataset cannot even be opened: `LeRobotDataset`
falls back to downloading the metadata from the Hub, and there is no such repo.

### The mechanism, and it is two mechanisms

**The data file.** `DatasetWriter` opens **one** `pq.ParquetWriter` and keeps it
across episodes, appending a row group per episode. A parquet file's schema and
row-group offsets live in the **footer**, and the footer is written by
`close()`, which happens in `finalize()`. A process that never reaches
`finalize` leaves a file whose bytes are all there and whose index is not.

**The episode metadata.** `LeRobotDatasetMetadata` buffers episode records and
flushes them every `metadata_buffer_size` — **ten** by default. Seven episodes
never reached the buffer's threshold, so `meta/episodes/` was never created.

### The fix, and the trap inside it

`DatasetSession` now does three things, and the third is not optional:

1. `update_chunk_settings(data_files_size_in_mb=0.001)` — a data file per
   episode, because a **rotation** is what closes the previous writer;
2. `_metadata_buffer_size = 1` — episode metadata written with its episode;
3. `_seal()` after every commit — both parquet writers closed, so the footer is
   on disk before the next attempt begins.

The trap: **(3) without (1) destroys data.** `_save_episode_data` reopens a
`ParquetWriter` at the *same path* when the size limit has not been reached,
which truncates the file it was appending to. The first version of this fix did
exactly that and lost the earlier episodes;
`test_a_second_episode_does_not_overwrite_the_first` is that bug, kept.

`update_chunk_settings` also writes the number into `info.json`, so a resumed
directory keeps rotating per episode without being told again.

Two smaller things fixed alongside: an episode that was never written leaves its
PNG cache behind, so `open()` deletes frame directories for episodes the
metadata does not know about; and `close()` was already committing a pending
episode, which is what keeps a *clean* Ctrl-C from losing one.

### What it costs to recover, and why we did not

The videos are readable and the scores are intact, so the attempts are not
unwitnessed. Rebuilding the parquet footer means parsing page headers by hand
and reconstructing Thrift metadata for 13 columns across 7 row groups — hours,
with no guarantee — to recover a per-frame stream that A0 does not need: it is a
scored zero-shot row, not training data. The row was re-run instead, against a
recorder that cannot lose it the same way.

---

## Recording: the episode that took minutes to save

**2026-08-26.** Keeping an episode printed SVT-AV1's configuration banner and
then sat there, with the arms energised and the operator waiting.

LeRobot v3.0 encodes with **SVT-AV1 on the CPU** by default (`crf 30`,
`preset 12`, GOP 2). Three 1280x720 streams at 3 550 frames is a lot of AV1.

Measured here, paired, 300 frames of all three cameras, extrapolated to a real
3 550-frame episode:

| | 300 frames | per episode |
| --- | ---: | ---: |
| SVT-AV1, noisy frames (worst case) | 11.1 s | ~131 s |
| `h264_nvenc`, noisy frames | 8.3 s | ~98 s |
| SVT-AV1, compressible frames | 4.1 s | ~49 s |
| `h264_nvenc`, compressible frames | 4.0 s | ~47 s |
| `h264_nvenc` + `--stream-video` | 0.5 s | **~6 s** |

Two readings, and the second is the one that matters:

- **The codec is worth taking, and it is free.** `--vcodec auto` resolves to
  `h264_nvenc` on this machine. One trap: **NVENC refuses a GOP below 4** —
  `avcodec_open2(h264_nvenc)` fails outright with LeRobot's default of 2, and
  the error surfaces as "Video encoding failed" per camera, i.e. as *no video*.
  `dk1lab.dataset` raises the GOP to 4 for hardware encoders.
- **The codec is not the bottleneck.** Most of the wait is LeRobot writing every
  frame to PNG during the rollout and reading it back to encode afterwards.
  `--stream-video` (LeRobot's `streaming_encoding`) skips the PNG entirely and
  encodes as the arms move: keeping an episode drops to seconds, and a crash
  leaves no loose frames.

`--stream-video` is **off by default** because it is not free where it counts.
Paced at 30 Hz, 300 ticks:

| | mean tick | p95 | max |
| --- | ---: | ---: | ---: |
| batch (default) | 1.4 ms | 1.8 ms | 2.9 ms |
| `--stream-video` | 4.5 ms | 4.3 ms | **215 ms** |

The mean is affordable in a 33.3 ms period; the one 215 ms stall at encoder
start-up is the kind of thing that starves the chunk queue. Turn it on
deliberately, and read the trace afterwards.

---

## Recording: the encode that could not fork

**2026-08-27.** The first attempt of A0 — the first scored row this cell has ever
run — recorded no video. Three lines, once per camera:

```
ERROR:lerobot.datasets.dataset_writer:Video encoding failed for observation.images.left:
  [Errno 1313558101] Unknown error occurred: 'avcodec_open2(h264_nvenc)'
ERROR:dk1lab.dataset:could not write episode 0
```

`1313558101` is `UNKN` read as a FOURCC — libav's way of saying the encoder
refused to open and declined to say why.

### It is not the codec, and not the GOP

The obvious suspect was the one already in this file: **NVENC refuses a GOP below
4**, and that failure looks exactly like this. It was not that — `dk1lab.dataset`
has raised the GOP to 4 since 2026-08-26 and `_encoder()` logs `gop 4`. Nor is it
a session limit: three concurrent NVENC encodes open fine.

The encoder is not broken at all. **The same encoder, in the same process, at the
same moment, works.** What it cannot survive is a `fork`.

LeRobot encodes the cameras concurrently:

```python
with concurrent.futures.ProcessPoolExecutor(max_workers=num_cameras) as executor:
```

`ProcessPoolExecutor` takes the default start method, which on Linux and Python
3.12 is **fork**. NVENC needs CUDA, and **a CUDA context cannot survive a fork** —
the child inherits driver state it is not allowed to use, `cuInit` fails inside
libav, and `avcodec_open2` returns a bare `UNKNOWN`. The rollout process holds a
CUDA context from the moment the policy's weights reach the GPU, so by the time
the operator is asked to keep an episode, every forked child is already doomed.

Two triggers, independently sufficient, and both are the same root cause —
`libcuda` initialised in the parent before the fork:

| parent has | encode runs in | result |
| --- | --- | --- |
| nothing | forked child | **OK** |
| a CUDA context (`torch.zeros(8, device="cuda")`) | forked child | **fails** |
| an NVENC encoder opened once in-process | forked child | **fails** |
| a CUDA context | spawned child | OK |
| a CUDA context | the process itself | OK |
| nothing | three forked children | OK — not a session limit |

The second row is the rollout. The first row is why this was never seen on the
bench: the 2026-08-26 codec measurements in § *The episode that took minutes to
save* were taken by a script with no policy loaded, so the fork was legal and
NVENC was fast. **A benchmark that does not hold the GPU the way the real run
does is measuring a different program** — the same warning § *The 27.7 Hz loop*
gives for a different reason.

It also explains the loose end left in `CLAUDE.md`: the single episode whose
encode raised on 2026-08-26, which is what made a failed write loud. Same bug.

### The fix: do not fork a GPU encode

`DatasetSession._parallel_encoding()` returns `False` when the resolved codec is
NVENC, and `commit()` passes it to `save_episode`. The three streams are then
encoded one after another **in the rollout process**, where CUDA is valid and
NVENC works. A CPU codec forks perfectly well and gains most of a 3x from doing
so, so it keeps the parallel path.

`auto` is a request, not a codec, and it is the default. Judging `self.vcodec`
would see the word `auto`, conclude it is not a hardware encoder and fork
anyway — losing every episode on this machine, where `auto` **is** NVENC. So
`_encoder()` now records what `auto` resolved to and the decision reads that.
`tests/test_dataset.py` pins all four cases.

### What it costs

Paired, same process, 600 frames of three 1280x720 cameras through the batch
path, extrapolated to a 3 600-frame (120 s) episode:

| | 600 frames | per episode |
| --- | ---: | ---: |
| parallel fork, no CUDA in the parent (the old bench) | 18.7 s | ~112 s |
| **serial in-process** | 25.1 s | **~150 s** |
| serial in-process, CUDA live (what now runs) | 25.0 s | ~150 s |
| parallel fork, CUDA live (the rollout) | **fails** | — |

About a third more wall clock, and **the parallelism was never worth much
anyway**: the difference is ~6 s of 25, because the wait is dominated by staging
every frame through PNG and reading it back, not by the encode. NVENC does its
part in about 3 s per stream. `--stream-video` is the lever that removes the
staging — it encodes in-process as the arms move, so it is immune to this bug by
construction, and it took the same measurement to **0.9 s**. It stays off by
default for the reason already recorded: one 215 ms stall at encoder start-up,
and the loop is the experiment.

Verified end-to-end with a live CUDA context: two episodes committed, no
failures, the dataset reads back 240 frames with all three camera streams
decoding to real pixels.

### One trap found on the way

`DatasetSession.__init__` defaults `streaming=True`, while the CLI passes
`streaming=stream_video`, which defaults **False**. The behaviour of `dk1` is
correct — every caller is explicit — but a `DatasetSession` constructed directly,
as a benchmark or a test does, silently takes the streaming path and never forks,
which is why this bug hid from the first attempt to reproduce it. Left alone
rather than changed: it is a default, and the defaults table in `CLAUDE.md` is
not edited in passing.

---

## The session console: a silenced prompt and a shouted decoder

**2026-08-27.** Two faults on the same screen, during A0's first attempt. The
prompt that names the scene and the attempt —
`[A0 scene 1/3, attempt 1/3 | episode 0 | 120s | dataset] task>` — had stopped
appearing, so the operator was typing blind into a session that walks three
scenes. And while the episode saved, thousands of lines of
`DEBUG:PIL.PngImagePlugin:STREAM b'IHDR' 16 13` ran up the terminal.

Unrelated causes. Both arrived with the 2026-08-26 work and neither is in the
log file, which was clean throughout — that is the tell for both.

### The prompt went to file descriptor 2

`input()` writes its prompt to **fd 2**, not stdout. Verified in a pty: redirect
fd 2 to a file and the prompt is *in the file*, not on the terminal.

`_quiet_stderr()` points fd 2 at `/dev/null` for exactly the length of the read,
because the cameras' MJPG stream makes libjpeg print `Corrupt JPEG data` from C,
past any Python redirect, into the middle of the line being typed (§ *The
cameras talking over the prompt*). That fix was right. Handing the prompt to
`input()` inside it was not: the prompt went to `/dev/null` with the chatter.

It took all three prompts — `task>`, `score>`, and `keep this episode?`, the
last because click hands the whole prompt to `input()` on this platform too.

`_ask` now writes the prompt to **stdout** itself and calls `input()` bare; the
chatter is still silenced while the line is read. That is what click does on
Windows, for the same reason. `typer.confirm` is replaced by `_confirm`, built
on `_ask`, keeping click's rules — empty takes the default, `y`/`n` decide,
anything else asks again, and **EOF takes the default rather than aborting**,
since a non-interactive keep must not become a lost attempt.

### The decoder shouted because we lowered the root logger

Opening the session log calls `root.setLevel(DEBUG)`, and it has to: the root
level gates every child *before* any handler sees a record, so our file handler
cannot get DEBUG any other way. Our handler filters (`Interesting`), which is
why the **file** stayed clean.

But root already had a handler nobody here attached. Importing `lerobot` reaches
`lerobot/utils/import_utils.py`, which calls the module-level `logging.debug()`
while probing for optional packages — and the module-level convenience functions
call `logging.basicConfig()` when root has no handlers yet. So a bare
`StreamHandler` on stderr, at no level and with no filter, exists before any of
our code runs. Lowering root to DEBUG turned it into a firehose: PIL on every
PNG read, `httpcore` on every connection.

`logs.start` now applies the same policy to every handler already on root, at
`CONSOLE_LEVEL` (INFO): ours and LeRobot's at INFO, everybody else at WARNING.
That is what the terminal showed before, minus the chatter — measured on a sim
run, console DEBUG lines 42 -> 0, INFO lines kept.

**The general rule, and it is the one to remember:** lowering the root level is
never local. It changes what every handler in the process emits, including the
ones a dependency installed at import time. Whoever lowers it owns the handlers
they did not attach.

---

## Recording: four minutes to keep one episode

**2026-08-27.** A0's first attempt was a 120 s rollout, and the operator then
waited **4 minutes 25 seconds** before the score prompt — arms energised, the
chain warning that nothing was commanding it. The log has both ends:
`15:04:38 encoding episode 0 (3539 frames)` and the next line at `15:09:03`.

Where it goes, measured paired at 900 frames of three 1280x720 cameras with a
live CUDA context, split by instrumenting the writer:

| phase | 900 frames | share |
| --- | ---: | ---: |
| waiting for the PNG cache to finish being written | 16.4 s | 45% |
| reading those PNGs back and encoding | 17.0 s | 47% |
| statistics, parquet, metadata | 3.0 s | 8% |

**The encode is not the problem; the PNG round-trip is.** NVENC itself is about
a second of that 17. Tripling the image-writer threads (4 -> 12 per camera) took
the wait from 16.4 s to 10.2 s — real, and not the answer.

`--stream-video` (LeRobot's `streaming_encoding`) skips the cache entirely and
encodes as the arms move. Same measurement: **save 36.5 s -> 0.9 s**, which is
~145 s -> ~3 s on a full episode, and two streamed episodes verified readable
with a live CUDA context.

**It was made the default, tried on the arms, and reverted the same day.** The
bench had said ~2 ms a tick against ~30 ms of idle. The cell said otherwise:

| | batch | `--stream-video`, on the arms |
| --- | ---: | ---: |
| worst tick | — | **983.7 ms** |
| starved ticks | 0 | **6** |
| loop rate | 29.9 Hz | 29.2 Hz |
| keeping an episode | ~4 min | seconds |

Paired on the bench, the tick cost of each mode, 600 ticks paced at 30 Hz with a
GPU load standing in for the policy:

| | mean | p95 | worst |
| --- | ---: | ---: | ---: |
| batch | 0.12 ms | 0.16 ms | **0.7 ms** |
| `--stream-video` | 1.7 ms | 2.8 ms | **117 ms** |

Batch is nearly free to the loop because the PNG writing is already on the
image-writer threads; only the *waiting* is expensive, and it is all after the
arms stop.

The streaming stall is **ticks 1 and 2, and only those** — LeRobot's
`_CameraEncoderThread` opens its container lazily, on the first frame, "to get
width/height", so three NVENC encoders initialise inside the control loop.
Pre-warming an encoder before the run halves it on an idle GPU (224 -> 106 ms)
and does nothing under load. Opening them early would need the width and height
pushed in ahead of the first frame, which is upstream's structure, not a flag.

None of that was the deciding argument. **The queue running dry means the arms
held their last commanded target instead of executing a new one**, six times, in
an attempt that was being scored. A worse attempt costs more than a wait, so
the wait is what we buy. Nikolas's call, and the right one: the loop is the
experiment.

**The mode is in `dk1_notes.jsonl` per episode**, because it is the one recording
setting that changes the control loop, and a row whose episodes were not all
recorded the same way is worth being able to find out about.

> Before reaching for this again: it is not a tuning knob, it is a trade of
> attempt quality for operator time. Read the starved-tick line of the trace
> after any run that uses it.

---

## The cameras talking over the prompt

**2026-08-26.** `Corrupt JPEG data: 11 extraneous bytes before marker 0xd7`
appeared in the middle of the line the operator was typing at the session
prompt.

It is libjpeg, decoding the cameras' MJPG stream: the UVC firmware pads a few
bytes before a restart marker, the decoder skips them, the frame is fine. The
message is written to **file descriptor 2 from C** — under no Python logger, and
past `contextlib.redirect_stderr`.

Fixed by silencing fd 2 for exactly as long as the session is waiting for a
line: `_quiet_stderr()` in `dk1lab/cli/policy_cmds.py`, around the task prompt,
the score prompt and the keep prompt. Nothing is commanded while the operator
types, so nothing is missed; everything printed **during** a rollout still
reaches the terminal.

What would not have been a fix: treating the message as a fault. Two lines at
start-up is normal. What matters is the recorder reporting **dropped frames**,
or the message repeating every tick with visibly blocky images — that is USB
bandwidth, and a different problem.

## Recording demonstrations: the Enter that stopped the episode it started

**2026-08-28.** Recording the fine-tune dataset, some episodes ended the instant
they began: **one frame written, and the session over.** Not every time, and no
error anywhere — the log for such a session ends with an ordinary disconnect.

### What the logs said

Three sessions on 2026-08-27 (`logs/20260827-1920`, `-1923`, `-1924`) end the
same way, and it is the same shape in all three:

```
19:21:45.663  encoding episode 1 (850 frames) ...      <- the operator's Enter
19:21:46.766  recording this episode into study/demos  <- 1.1 s later, ep2 starts
19:21:47.076  encoding episode 2 (1 frames) ...        <- 0.31 s later, it is over
19:21:48.307  DK1Follower disconnected.
```

Two numbers carry it. The **1.1 s** between the Enter and the episode actually
starting is `save_episode` on the *previous* episode: `start_episode` commits the
held one before it opens the next, and the loop does not tick while it does.
The **0.31 s** after is one tick and then the loop exiting — the follower's
`No command for 0.64 s` warning fires at the same moment, so the loop stopped
ticking immediately, and `dk1_notes.jsonl` records the episode as 1 frame.

### The cause: what was typed while nothing was listening

The loop is the only thing reading the keyboard, and it stops reading for as long
as it is writing an episode. The terminal goes quiet mid-command, which is
indistinguishable from a hung process, so the operator presses Enter again — and
those keystrokes wait in the terminal's input queue. The loop resumes *after* the
next episode has started and reads them back one per tick: the first is an empty
line, which means stop, so the new episode ends after a single frame; the second
is `done`, or another Enter followed by `done`, and the session is over. Every
keystroke did exactly what it says. It just arrived at an episode that did not
exist when it was typed.

Reproduced off the hardware, in a pty, with a fake dataset whose `save_episode`
sleeps 1.2 s: one Enter and one `done` written during that sleep produce a
1-frame episode and an ended session, exactly as on the cell.

The same queue explains the first session's version of it, where the 1-frame
episode is the *first* one: connecting takes fifteen seconds and asks two
questions, and anything typed over that is still queued when the loop starts.

### The fix

`TerminalConsole.drain()` — every complete line already typed is discarded, and
the operator is told how many. It is called where the loop resumes after not
listening: at the top of `DemoSession.loop`, and in `start_episode` immediately
after `commit_held`. Only whole lines go; a half-typed word survives, because the
terminal has already echoed it and deleting it silently is its own confusion.

Second guard, because the first one cannot cover a stray keypress that is
genuinely mistimed: an episode shorter than `demos.MIN_EPISODE_S` (0.5 s) is
**dropped rather than written**, with a line saying so. Half a second of
teleoperation cannot show a task being done, and `dk1 dataset check` already
reports a sub-two-frame episode as a fault — writing one and then flagging it is
the worse half of both options.

Third, unrelated to the input and found beside it: `DatasetEpisodeRecorder._close_tick`
caught only `KeyError` around `_frame`. Anything else — a bad dtype, a `TypeError`
— left through `send_action`, i.e. through the teleoperation loop, with the arms
live. It catches every exception now, which is what the module docstring already
claimed.

### What this does not explain, and what it costs

Nothing here touches the freeze itself: writing an episode still blocks the loop
(about a second with `--stream-video`, minutes without — § *Recording: four
minutes to keep one episode*), and the arms are uncommanded for that long. The
drain makes the freeze harmless to the *next* episode; it does not remove it.

The price is that a `done` typed during a write is thrown away with everything
else and has to be typed again. Against a ruined episode and a terminated
session, that is not a close call.

`study/demos` as recorded on 2026-08-27 carries two of these: episodes 0 and 5,
one frame each. v3.0 cannot take an episode back out, so they stay until the
directory is re-recorded.

## Recording: the demonstration set, and what the training-time crop costs

2026-08-28. `STUDY.md` Phase 3 is done and Phase 4's input exists.

**The set.** 26 episodes, 18 484 frames, 10.3 minutes, one continuous session
10:47–11:08 under `--profile common` / `[capture.policy]` 1280x720 / 30 Hz /
uncapped, NVENC, `--stream-video` on. One task string on every frame. Episode
length 431–1040 frames, mean 711 (23.7 s), and it falls through the session —
1040 frames for the first, ~500 for the twenties — which is the operator getting
faster, not the task changing. Every video file decodes to exactly the frame
count its episode metadata claims: 6098 + 12386 = 18 484 per camera, three
cameras, no loss.

**Not 45, and not three labels.** Time ran out at 26; the hold-out drops from 10
to 4 accordingly (`STUDY.md` amendment of 2026-08-28) because 10 of 26 is 38% of
the set. And every episode was written as `scene 1`: the layouts *were* varied but
`scene <n>` was never typed, so the labels were all the session's opening value.
Relabelled from the operator's own ranges — 0–7 scene 1, 8–14 scene 2, 15–25
scene 3 — with `scene_source` on each corrected record saying so, and the file as
written kept beside it as `dk1_notes.recorded.jsonl`. The hold-out then lands on
episodes 4, 11, 15, 25, one from each of the first two scenes and two from the
third, which is the proportion of 8/7/11.

Worth noticing for the next session: **the scene label is sticky and silent.**
Nothing prompts for it and nothing warns when a whole set carries one value, so
the failure mode is not an error, it is a dataset that quietly says something
untrue. `dk1 dataset check` prints the per-scene counts, which is what caught it.

**What the training-time crop costs — 40 dB, and 203 s.**
`dk1 dataset crop study/demos study/demos-optimized` rewrote the two wrist
streams through the box `dk1 config show` reports (909x511 at (185,24), 85.6 deg
H) and copied the top view byte-for-byte. Four files, 36 968 frames, **203 s**.
It reused the source's own encoder settings off `meta/info.json` — h264, crf 30,
GOP 4 — so the copy differs from the original in pixels and nothing else.

The generation loss the module warns about was measured rather than assumed:
against the *ideal* crop-and-resize of each recorded frame, ten sampled frames
give **min 39.2 dB, mean 40.4 dB** PSNR. That is visually lossless. The copy is
also **41% smaller** (289 MB against 487 MB), and that is not quality thrown
away — an upscaled crop carries less high-frequency detail than the frame it came
from, so it compresses harder at the same CRF. Do not read the size drop as a
warning.

What this does **not** measure: the crop against what the `optimized` *camera*
delivers live, which is never encoded at all. The training frames are bounded by
the recording's own NVENC pass, and that bound is the same one A1's frames carry.

## The gripper command that was never executed

2026-08-28. The first real fine-tune probe trained ten steps at 1.13 step/s and
then died at its first evaluation:

```
ValueError: MolmoAct2 action gripper values are not under [-1, 1].
Please set normalize_gripper=True.
```

**The mechanism, from the top.** MolmoAct2 does not normalise the gripper
channel. `_add_gripper_masks_to_stats` builds a mask over the 14 channels —
`"gripper" not in name.lower()` — and `_MolmoAct2MaskedNormalizationMixin`
normalises the masked channels and **passes the gripper through untouched**,
because the checkpoint expects it already in [-1, 1]. That is the same fact the
gripper inversion rests on. `_validate_masked_passthrough_range` then **raises**
on any passthrough value outside that range.

**The data.** Of 18 484 recorded frames, 1 316 (7.1%) carry a gripper *command*
above 1.0 — max 1.0230 left, 1.0344 right — across 16 of the 26 episodes. The
measured `observation.state` never exceeds 0.9842. So the robot never went there.
`DK1Robot.command_gripper` (upstream, `robot.py:138`) does
`np.clip(normalized_pos, 0.0, 1.0)` **inside** the robot and returns nothing, so
a leader trigger squeezed past the follower's closed stop was *executed* as 1.0
and *recorded* as 1.03.

`SafeBiDK1Follower.send_action` promised to "return the action that was actually
sent, not the one requested — so a recorded dataset stores what the arms were
told to do". For the gripper that was not true, and 7% of a dataset was enough
to stop a training run dead.

Two things were wrong with how it surfaced, and both are worth keeping in mind:

* **it is not deterministic.** The first ten training steps drew batches with no
  offending frame and passed. The hold-out (episodes 4, 11, 15, 25) contains
  three of the sixteen, so the *evaluation* was certain to fail — which is why it
  looked like an eval-specific fault and is not one;
* **`normalize_gripper=True`, which the error suggests, is the wrong fix here.**
  It normalises the gripper with the YAM quantile statistics, so the fine-tuned
  checkpoint's gripper would mean something different from A0's and R0's. A1
  against A0 is supposed to differ by the LoRA and nothing else.

**The fix, in two halves.** `SafeBiDK1Follower.send_action` now clips the gripper
channels of what it returns, so the recorded action is the executed one — the
docstring's promise, made true. `dk1 dataset clamp` repairs a dataset recorded
before that: it rewrites `data/` and the gripper entries of `meta/stats.json`
in place, leaves the videos and `observation.state` alone — clipping a
*measurement* would be inventing data — and writes what was recorded to
`dk1_clamp.json` first. `dk1 dataset check` now reports the fault, so the next
one is found in seconds rather than in a training run.

### And the log that was not there

Found in the same session, and worse in its way. `dk1 policy curve` reported
*no held-out loss was logged* on a run whose console had plainly printed
`step 20: eval_loss=0.0685`. The run's `train.log` held **one line**.

Two independent causes, both silent:

* `lerobot.utils.utils.init_logging` — `lerobot_train`'s third statement — does
  `logging.getLogger().handlers.clear()`. Our fsynced file handler, attached
  moments earlier, was simply gone. `finetune.patched` now wraps `init_logging`
  and `dk1lab.logs.restore` puts the handlers back;
* `lerobot_train` logs with bare `logging.info(...)`, and a bare call goes to the
  **root** logger — so its records arrive named `root`, not
  `lerobot.scripts.lerobot_train`. `dk1lab.logs.Interesting` held anything not
  named `dk1lab*` or `lerobot*` to WARNING, so every one of those lines was
  dropped from the file *and*, once the console handler was being tamed too,
  from the screen. `APPLICATION_LOGGERS` now treats `root` and `__main__` as
  application code at INFO. A library that spams uses
  `logging.getLogger(__name__)` and is still held at WARNING.

This is not a logging nicety. There is no early stop, so **the log is the
checkpoint-selection mechanism**: a run whose log is empty cannot be selected
from, and its GPU hours buy a directory of checkpoints nobody can rank.

### What a step and an evaluation cost

Measured on the arms' own machine, R1's configuration — MolmoAct2 5.44 B,
`--adapt vlm+expert` (106 M trainable), batch 2, gradient checkpointing on,
22 training episodes / 15 605 frames:

| | |
| --- | --- |
| training | **1.13 step/s** steady (the first ~5 steps are slower: 5.1 s, then 2.6, then settling) |
| one evaluation over the whole hold-out | **5 min 27 s** — 4 episodes, 2 879 frames, 1 440 forward passes at ~4.4 batch/s |
| load, wrap and set up | ~75 s |
| one checkpoint on disk | **1.2 GB** (adapter 406 MB + optimiser state 801 MB) |

So the budget is `steps / 1.13` seconds of training plus `327 s x evaluations`.
At 20 000 steps that is 4.9 h of training, and **evaluating every 1 000 steps
adds 1.8 h of it** — over a quarter of the night spent measuring.

`--max-eval-samples` caps it, but LeRobot takes the **first** N frames of the
hold-out, which is one episode's opening rather than a spread across the three
scenes. Halving the evaluation *count* is the better economy: it keeps the
validation set complete.
