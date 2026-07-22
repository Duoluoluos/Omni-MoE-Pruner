# SCEB Audio+Query Guard v2: SOTA-Selective Label-Free Policy

## Motivation

The first label-free Audio+Query Guard was fair, but too broad. Its
`fine_visual_relation_reference` cluster fired on 1070/3172 WorldSense samples
and had low accuracy. Its `audio_reference` cluster fired on 1163/1734 AVUT
samples, so it behaved more like a broad audio-task oracle than a selective
query guard.

Current omni-modal benchmark designs consistently make audio useful when the
query asks for cross-modal grounding rather than when it merely contains an
audio word. Daily-Omni emphasizes cross-modal temporal reasoning and text-only
leakage filtering; AV-Odyssey and JointAVBench separate speech, sound events,
music, temporal/spatial/emotional dimensions; LongVALE highlights
audio-visual correlation in long videos; AVUT focuses on audio content
understanding and audio-visual alignment such as speaker grounding, event
counting, event localization, and screen/audio matching.

## Policy

`audio_query_guard_v2_sota_selective_label_free` uses only query text, answer
options, and audio-token presence. It never uses benchmark task labels.

Protected clusters set a video-token dynamic-budget floor of `top_m=6`:

- `audio_speech_grounding`: speech or speaker cues tied to who/which/where or
  visual person/screen cues.
- `audio_event_temporal_count`: audio events tied to temporal localization or
  counting.
- `audio_music_event_alignment`: music/sound cues tied to visual action,
  matching, dance, scene, or event alignment.
- `screen_text_ocr_reference`: explicit screen text, subtitles, captions,
  diagrams, charts, signs, logos, written text, or OCR.

Everything else falls back to the Global SCEB dynamic floor `top_m=4`.

## Why This Is Less Heuristic

The clusters are not benchmark task labels. They are observable query-intent
patterns that correspond to recurring audio-query families in recent
omni-modal evaluation work:

- speech grounding;
- sound-event temporal reasoning;
- audio-event counting;
- music/event alignment;
- screen/OCR reference.

The claim is not that these rules are a final language parser. The claim is
that query-visible multimodal risk can calibrate SCEB's budget floor without
using private task labels.

## Experiment

Config:

- `logs/audio_query_guard_label_free_20260717/audio_query_guard_v2_sota_selective_policy.json`

Offline rule simulation:

- `logs/audio_query_guard_label_free_20260717/V2_RULE_SIMULATION.md`

Full-run outputs:

- `logs/audio_query_guard_label_free_20260717/omnivideobench_audio_query_guard_v2_full`
- `logs/audio_query_guard_label_free_20260717/avut_audio_query_guard_v2_full`
- `logs/audio_query_guard_label_free_20260717/worldsense_audio_query_guard_v2_full`

Launched 2026-07-17 12:16 CST on worker `4008126`.

Session:

- `audio_query_guard_v2_three_bench`

Launcher log:

- `logs/audio_query_guard_label_free_20260717/launcher/audio_query_guard_v2_three_bench_worker_4008126_20260717_121605.log`

Initial verification:

- OmniVideoBench produced 3 prediction/profile rows.
- First three profile rows use
  `policy_version=audio_query_guard_v2_sota_selective_label_free`.
- `task_layer_full_layers=[]`, so no task-label layer protection is active.

Protocol:

- run OmniVideoBench full;
- then AVUT full;
- then WorldSense full;
- no `run_keepalive.sh`;
- primary latency metric is model-side E2E `generate_latency_ms`.
