# Loop8 Audio-Guard v4 Candidate Policies

Loop8 searches for a stronger Audio-Guard v4 policy family for the
OmniVideoBench audio-salient subset. The goal is not to prune audio tokens.
Audio, text, image, and special/control tokens remain protected at full top-8.
The policy family uses audio and query regime as a controller for the video
expert budget.

## Starting Evidence

The measured anchors are:

| Policy | Correct | Accuracy | Mean active experts | Top-4 | Top-6 | Top-8 | Role |
|---|---:|---:|---:|---:|---:|---:|---|
| Vanilla top-8 | 84/200 | 42.00% | n/a | n/a | n/a | n/a | quality anchor |
| Global SCEB | 85/200 | 42.50% | 7.04 | 13.27% | 22.08% | 64.65% | matched mainline anchor |
| Boundary-only SCEB | 83/200 | 41.50% | 5.84 | 40.38% | 31.15% | 28.47% | aggressive boundary anchor |
| Audio-Guard v2 | 83/200 | 41.50% | 7.38 | 5.09% | 25.64% | 69.27% | conservative AV guard |
| Loop7 speed v3 | 79/200 | 39.50% | 6.90 | 18.84% | 25.15% | 56.01% | speed frontier, quality loss |

Loop6 explains why v4 should be audio-conditioned but video-targeted:
audio/video top-expert overlap is real but sparse, and audio tokens are only a
small fraction of the sequence. Audio is therefore a protected controller, not
the compute target. Loop7 shows that pushing many non-core rows to top-4
recovers effective budget but loses four samples against v2. V4 should recover
those losses while preserving a clearer budget story than simply returning to
v2.

## Candidate A: Guard-Restored v3.5

This candidate keeps the speed-oriented v3 thresholds only for clearly
speed-safe regimes, then restores top-6 or full guards for fragile regimes.

Rules:

- `background&music understanding`: full top-8 for all video dynamic layers.
- `temporal understanding`, `reference reasoning`, `spatial understanding`:
  minimum video top-6, with layers 24-31 forced full top-8.
- `ego reasoning`: minimum video top-6, with layers 28-31 forced full top-8.
- `counting`, `causal reasoning`, `relationship reasoning`,
  `sentiment analysis`, `summarization`, `hypothetical reasoning`,
  `fine-grained perception`, and `attribute comparison`: v3 thresholds, but
  no lower than top-4.

Expected behavior:

- Recovers the temporal/reference/spatial/ego losses seen in v3.
- Keeps more budget reduction than v2 on counting, causal, relationship,
  sentiment, summarization, hypothetical, perception, and attribute rows.
- Likely mean active experts: between v2 and v3, around 7.05-7.20.

Risk:

- The restored late-middle full guards may erase much of the v3 budget gain.
- Ego has only 10 rows, so extra protection may not move the 200-row score.

## Candidate B: Audio-Core Plus Late-Middle Full Guard

This candidate assumes most v3 quality loss comes from late integration rather
than the top-4 floor itself. It keeps the broader v3 speed thresholds, but adds
a late-middle full guard for AV-sensitive task groups.

Rules:

- `background&music understanding`: full top-8 for all video dynamic layers.
- `temporal understanding`, `reference reasoning`, `spatial understanding`,
  `ego reasoning`: minimum video top-6, layers 24-31 full top-8.
- `counting`: minimum video top-6, layers 28-31 full top-8.
- All other rows: v3 speed thresholds and top-4 floor.

Expected behavior:

- Stronger quality recovery than Candidate A for counting if counting flips
  are late-grounding failures.
- Still allows top-4 on lower-risk reasoning and summarization rows.
- Likely mean active experts: around 7.10-7.25.

Risk:

- Counting was not a measured v3 failure bucket by aggregate accuracy, so the
  extra guard may be unnecessary budget spend.
- The policy may become harder to explain than Candidate A because counting is
  partly speed-safe in Loop6.

## Candidate C: v2.5 Quality-First Hybrid

This candidate is a conservative bridge between v2 and v3. It starts from
Audio-Guard v2 and applies v3 thresholds only to lower-risk non-core rows.

Rules:

- Keep all v2 audio-core, audio-stress, audio-reasoning, and late-middle full
  protections.
- Apply v3 thresholds only to `sentiment analysis`, `summarization`,
  `relationship reasoning`, `hypothetical reasoning`, `fine-grained
  perception`, and `attribute comparison`.
- Keep v2 thresholds for `counting`, `causal reasoning`, `temporal
  understanding`, `reference reasoning`, `spatial understanding`, and `ego
  reasoning`.

Expected behavior:

- Best chance to stay at 82-83 correct.
- Smaller but still measurable top-4 increase against v2 on lower-risk rows.
- Clean long-paper story: quality-first AV-SCEB extension with a restrained
  speed subfamily.

Risk:

- Budget reduction may be too small to distinguish from v2.
- If the verifier only rewards full Loop8 completion, this candidate may be
  scientifically useful but not numerically differentiated enough.

## Runnable Choice For Loop8

Use Candidate A as the first runnable v4 policy:

```text
policy_version = sceb_loop8_audio_guard_v4
family = guard-restored v3.5
target = video tokens only
protected = audio/text/image/special tokens
dynamic layers = middle layers
audio-core = full top-8
fragile AV rows = top-6 floor plus late-middle full guard
speed-safe rows = v3 thresholds
```

Candidate A directly responds to the Loop7 failure analysis: temporal,
reference, spatial, and ego rows lost too much quality under the speed policy,
so v4 restores their floor while keeping the v3 speed frontier where Loop6
marked risk lower. It is also simpler than Candidate B and less likely than
Candidate C to collapse into a near-v2 no-op.

## Measurement Criteria

Strong success:

- limit-200 completes at least 180 rows;
- v4 reaches at least 82/200 correct;
- v4 preserves a clearer audio-guard story than v2/v3;
- mean active experts remains below v2 or top-4 share remains above v2.

Useful success:

- v4 loses only one or two samples against v2;
- or v4 recovers most v3 losses while preserving some effective expert-budget
  reduction.

Negative result:

- If v4 stays near 79/200, the restored fragile-row guard did not address the
  causal failure; report that the v3 losses likely come from broader
  lower-risk-row threshold pressure.
- If v4 returns to 83/200 but budget distribution matches v2, report it as a
  quality recovery with little extra budget value.
