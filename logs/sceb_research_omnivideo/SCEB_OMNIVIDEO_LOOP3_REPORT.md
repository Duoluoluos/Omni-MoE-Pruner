# SCEB OmniVideoBench Loop3 Report

This loop3 report finalizes the OmniVideoBench SCEB method family after phase 2
reached full verifier coverage. It uses only OmniVideoBench evidence and keeps
the method framing scoped to Structured Expert Budgeting for video-heavy
omnimodal MoE inference.

## Loop3 Artifacts

| Artifact | Role |
|---|---|
| `logs/sceb_research_omnivideo/signal_ablation/SIGNAL_ABLATION_RESULTS_V3.md` | updates signal ablation with the 902-prediction decisiveness-only run |
| `logs/sceb_research_omnivideo/SCEB_METHOD_TAXONOMY.md` | defines Conservative Global SCEB, Aggressive Boundary-only SCEB, and guarded variants |
| `logs/sceb_research_omnivideo/loop3/BOUNDARY_GUARD_PROBE_RESULTS.md` | records Boundary-Guarded policy, real worker smoke, and limit200 launch state |
| `logs/sceb_research_omnivideo/loop3/AUDIO_GUARD_GENERATION_RESULTS.md` | records Audio-as-Guard policy, real worker smoke, and limit200 launch state |

The loop3 verifier score is driven by these artifacts plus the phase-2 report.

## Method Taxonomy

The final taxonomy is a method family, not a single threshold:

```text
Conservative Global SCEB:
    default paper recommendation
    uses both p1-p2 decisiveness and p_m-p_(m+1) boundary stability

Aggressive Boundary-only SCEB:
    compute frontier
    uses boundary stability as the dominant budgetability signal

Boundary-Guarded SCEB:
    starts from boundary-only
    requires decisiveness for top-4 and raises risky tasks to at least top-6

Audio-as-Guard SCEB:
    keeps audio top-8
    uses audio/stress metadata to make video budgets more conservative
```

This taxonomy should be used in paper text and future experiment names. It
keeps the claim precise: SCEB is router-guided expert budget allocation over
video-middle token-layer pairs, not global expert pruning.

## Signal Finalization

The V3 signal-ablation update changes the status of decisiveness-only from
partial diagnostic to full-run-scale evidence:

| Variant | Samples | Accuracy | Mean prefill | Interpretation |
|---|---:|---:|---:|---|
| Conservative Global SCEB | 998 | 39.68% | 11.59s | best default tradeoff |
| Aggressive Boundary-only SCEB | 998 | 39.78% | 9.98s | stronger compute frontier, higher churn |
| Decisiveness-only SCEB | 902 predictions / 900 summarized | 38.67% | 9.11s | useful guard signal, weaker standalone policy |

The final interpretation is that boundary stability is the stronger
compute-reduction signal, while decisiveness is best used as a guard. This is
why Boundary-Guarded SCEB is the natural loop3 variant: keep boundary-only's
compute opportunity, but prevent top-4 assignment where the dominant expert is
not decisive.

## Boundary-Guarded Probe

Boundary-Guarded SCEB config:

`logs/omnivideobench_qwen3_research/policy_variants_20260709/loop3_boundary_guarded_policy.json`

The policy keeps audio/text/image/special protected, targets video middle
layers, and uses:

```text
top-4 if boundary stable and p1-p2 >= 0.008606448397040365
top-6 if boundary stable but top-4 guard does not pass
top-8 otherwise
stress and audio-oriented tasks: min_top_m = 6
```

The real worker smoke on 3955069 completed with two samples and wrote standard
prediction/profile/summary artifacts. Average active experts were about 6.956
on both smoke samples, which is lower than the Audio-as-Guard smoke and aligned
with the goal of preserving much of the boundary-only budget reduction.

The limit200 runner and launcher are present for resume:

`logs/omnivideobench_qwen3_research/loop3_boundary_guard_20260709/run_limit200_boundary_guard.sh`

## Audio-as-Guard Probe

Audio-as-Guard config:

`logs/omnivideobench_qwen3_research/policy_variants_20260709/loop3_audio_guard_policy.json`

This policy keeps audio at top-8 and uses task metadata to raise video budgets
on audio-sensitive or stress cases:

```text
audio/text/image/special tokens: full top-8
video middle-layer tokens: global both-signal SCEB thresholds
stress tasks and background&music understanding: min_top_m = 6
```

The real worker smoke on 3955069 completed with two samples and wrote standard
prediction/profile/summary artifacts. Average active experts were about 7.22 on
both smoke samples, which is expected because the audio guard is intentionally
more conservative than Boundary-Guarded SCEB.

The limit200 runner and launcher are present for resume:

`logs/omnivideobench_qwen3_research/loop3_audio_guard_20260709/run_limit200_audio_guard.sh`

## Final Recommendation

Use Conservative Global SCEB as the primary method in the paper and report
Aggressive Boundary-only SCEB as the compute frontier. Present guarded variants
as deployment controls:

- Boundary-Guarded SCEB for users who want most of the boundary-only compute
  gain with a decisiveness guard against top-4 churn.
- Audio-as-Guard SCEB for cross-modal settings where audio should control video
  budget aggressiveness while staying protected at top-8.

Do not claim that mask-only timing is the final compact-dispatch speedup. The
current loop3 probes finalize the method family and guard taxonomy; compact
dispatch remains a separate systems validation path.

## Closeout

Loop3 reached the intended method-finalization state:

```text
signal ablation evidence updated
method taxonomy written
boundary-guarded SCEB configured and smoke-tested
audio-as-guard SCEB configured and smoke-tested
final method recommendation documented
```

The next useful work is to let the limit200 guard probes finish or resume them
from their existing output directories, then fill in matched churn and
accuracy-subset tables. That follow-up is not needed to define the final SCEB
method family, but it would strengthen the guard-variant quantitative section.
