# Project memory: Omni-MoE-Pruner → ICLR 2027

## North star

- The goal remains an empirical ICLR 2027 paper on efficient omnimodal MoE computation.
- `research_paper/` is the canonical paper workspace. The live scientific entry point is now `research_paper/MY_BUDGETING_RESEARCH_STATUS.md`.
- OmniZip is prior CVPR 2026 work. Any new submission must establish a distinct contribution with leakage-free evidence.
- Planning notes may be Chinese; manuscript prose and final assets should be publication-quality English unless the author requests otherwise.

## Scientific reset — 2026-08-30

- **ARC-SCEB is withdrawn.** The ARC path accidentally used the ground-truth `audio_type` label. Its apparent acceleration was caused by this implementation leakage; ARC has no valid acceleration evidence.
- Do not cite, restore, summarize, reinterpret, or reuse ARC mechanisms, configurations, figures, results, logs, or derived claims as paper evidence.
- The former audio-free SCEB comparator is temporarily named **My Budgeting**. This is a placeholder name, not a frozen method, novelty claim, or permission to transfer old ARC claims.
- The former SCEB result line is retained under the temporary name **My Budgeting**. Its audited index is `research_paper/iclr2027/evidence/MY_BUDGETING_RESULT_ARCHIVE.md`.
- The active quantitative set contains (i) the recorded aggregate comparison between `No acceleration` and `My Budgeting` in `research_paper/iclr2027/paper/results/inference_results.csv`, and (ii) the fully archived, verified P-M1 workload decomposition and P-M2 equal-layer uniform frontier. The aggregate method definition and end-to-end provenance still must be re-frozen before supporting a submission-level contribution.
- P-M4 is a separate `verified` modality--layer-depth characterization from
  worker 4207965. Its full 1,000-example run uses exactly matched
  audio/video × early/middle/late layer-depth prefill interventions and no
  evaluation labels; its identity, budget-match, and raw-data gates passed.
  It is active only as a terminal, model/benchmark-specific stability detail:
  it does not validate a selector, establish modality utility, or support a
  speed claim.
- Historical router-policy ablations remain archived, but they are not uniformly valid paper evidence. In particular, the surviving boundary-only configuration consumes benchmark `task_type` to change the minimum budget; variants sharing that path are `placeholder` until rerun without evaluation metadata. Report-only variants whose raw run directories are absent are also `placeholder`.
- ARC training results, audio-conditioned results, token-compression composition results, and E7/E8/E16/E17 are outside the active evidence set and must not re-enter the manuscript.
- Another agent is exploring a replacement idea. Do not treat its files or running experiments as accepted paper evidence until the author explicitly integrates them.
- The manuscript is in scientific-reset state, not submission-ready.

## Second-backbone replication — 2026-09-02 (Nemotron DROPPED)

- The Nemotron-3-Nano-Omni second-backbone attempt is **abandoned on this cluster**: its custom HF code needs an unreleased `transformers` `main` build (symbols `initialization`, `lazy_load_kernel`, `resolve_internal_import`, `output_capturing`, `merge_with_config_defaults`, `ParakeetFeatureExtractor` are absent from 4.55/4.56/4.57), and the isolated-env attempt fought the shared CUDA/torch stack. All Nemotron scripts, protocol, aborted run dir, plans, logs, and the 62 GB model were dropped by author decision. Do not restore Nemotron artifacts as evidence.
- Replacement second-backbone candidate: **UniMoE 2.0 Omni**, to be tried on the existing shared `.venv` (torch 2.6.0+cu124, transformers 4.57.3, flash_attn 2.7.4). Status: evaluating compatibility / download.

## Structure and token--expert budgeting framing — 2026-09-01

- The manuscript is now intro-centric. There is no separate Motivating Study section. Figure `fig:intro-observations` (`paper/figures/motivation_observations.pdf`) has three panels: (a) a definition-only token--expert budgeting grid, where the row modality supplies a router-derived signal and the column modality's tokens receive the resulting expert budget, (b) a verified P-M1 video/audio/text workload donut with true-share separated audio/text wedges, and (c) verified P-M2 equal-layer uniform frontier. No structural-marker or image category enters panel (b); the exhaustive input-token ledger is in the appendix. Panel (a) is not a signal-utility result.
- Panel (c) directly labels its three operating points as **Native top-$k$**, **Uniform top-$3k/4$**, and **Uniform top-$k/2$**. Here $k$ is the native number of selected experts; uniform means that the indicated count applies to every token at every MoE layer. The plotted 95\% CI whiskers are deliberately subdued, axes stop just outside their verified extrema, and points are enlarged; no duplicate uniform-top-$k$ point is shown because it is execution-equivalent to native routing.
- Audio guidance is OUT of the main text by author decision. Do not re-add an audio bridge, P-M3, or any audio-benefit claim. The audio row in Figure~1(a) is solely a token--expert budgeting definition; it supports no audio claim. P-M4 may show its paired audio/video token conditions only as a non-utility matched characterization in Experiments.
- P-M4 (matched modality×layer-depth expert-budget sensitivity, worker 4207965) is `verified`. Its protocol is leakage-free (selection uses only token positions and modality sentinel IDs; never `task_type`, `audio_type`, or answer labels) and exactly budget-matched (every non-native arm removes `q*16*4` token-expert pairs). The full 1,000-example summary reports passed identity/budget/data gates. It may appear only as a terminal Figure 2 / Experiments characterization: all primary paired-score CIs include zero, while the secondary flip endpoint is higher early than late for both token conditions. It stays out of Figure~1 and the Introduction's core chain, and cannot establish modality utility, a selector, or speed.
- My Budgeting is framed as a label-independent policy that separates structural scope (P-M1) from runtime selectivity (still unresolved). The next signal-selectivity question is an explicit TODO in `research_paper/iclr2027/paper_story.md` and MB-NEW-4 in the experiment inventory.

## Evidence and writing contract

- Start every paper task from the live status file. If code, logs, or older notes conflict with it, the status file wins unless the author explicitly changes it.
- Use only `planned`, `placeholder`, and `verified` for active results. Withdrawn ARC artifacts are not active results and receive no reusable result status.
- Only `verified` evidence may support the abstract, Introduction contributions, Conclusion, or submission tables.
- Do not invent or interpolate metrics, baselines, ablations, error bars, citations, formulas, or implementation details.
- Matched comparisons must align model, split, seed, optimizer, batch sizes, hardware, software stack, warm-up/measurement window, and evaluation protocol; disclose deviations.
- Do not claim general speedup from a single device, short run, logical expert-count reduction, or an invalidated implementation path.
- Verify citations from authoritative metadata or the paper itself. Do not generate final BibTeX from memory.
- Preserve double-blind anonymity and maintain a concise AI-use ledger.
- Preserve files actively produced by the replacement-idea agent and unrelated working-tree changes.

## Current terminology

- **OmniDial** is the author-approved public method name as of 2026-09-28, replacing CG-CC in the active paper, figures, tables and planning prose. Descriptor: confidence-guided expert budgeting for omnimodal models. This is a naming change, not new evidence or a revised policy.
- Frozen experiment paths, arm keys (`cgcc`, `cg_*`) and source snapshots retain their original identifiers for reproducibility. The running two-backbone scope studies must not be edited or restarted for the rename.

- **My Budgeting** is the temporary name for the surviving audio-free expert-budgeting line formerly called SCEB.
- Do not use `SCEB`, `Global SCEB`, `ARC`, `ARC-SCEB`, `audio guard`, or `audio-conditioned budget floor` in the active manuscript.
- Do not infer a finalized method definition from the temporary name.
- **Equal-layer uniform reduction** means applying the same reduced expert budget to every token at every MoE layer (P-M2); do not abbreviate this as merely ``uniform reduction'' in active manuscript text.
- **Layer depth** is the noun for P-M4's early/middle/late layer locations; use **layer-depth band** only as the compound modifier in figure labels or prose.
- In the **token--expert budgeting design space**, video/text/audio label token modalities, not separate expert pools. A row is the token modality supplying a router-derived signal; a column is the token modality receiving an expert budget (the number of selected experts). The video-token column is a P-M1 workload opportunity ceiling; no router-derived signal has established utility.

## Working mode

- Page-count work and reviewer/editor agents remain disabled unless the author explicitly re-enables them.
- Main repo-to-paper workflow: `ai-research-writing-skill`.
- Literature workflow: `paper-collector` → `paper-reader` → `paper-wiki` → `topic-literature`.
- Experiment planning/results integration: `empirical-paper-writer` and `results-backfill`.
- Use `research-paper-writer` only if the author explicitly requests real multi-agent orchestration.

## ICLR 2027 venue snapshot

- Re-check official pages before deadline-sensitive, formatting, anonymity, policy, or submission work.
- Abstract deadline: September 18, 2026 AOE; paper deadline: September 25, 2026 AOE.
- Initial submission is double blind with at most nine main-text pages; references are excluded. The author has temporarily disabled page-count tracking during research iteration.
- Required AI-use disclosure and current official style must be handled before submission.

## Next integration gate

1. Freeze the replacement method with the author.
2. Prove that its inputs contain no evaluation labels or outcome-derived features.
3. Rebuild the claim–evidence map from leakage-free runs only.
4. Re-run matched quality and wall-clock experiments before restoring any acceleration claim.
5. Resolve MB-NEW-4 with a matched, label-independent signal-selectivity test before making any local, cross-modal, partial, global, audio, text, or video signal-utility claim.
6. Rewrite the paper around the replacement method only after those gates pass.
