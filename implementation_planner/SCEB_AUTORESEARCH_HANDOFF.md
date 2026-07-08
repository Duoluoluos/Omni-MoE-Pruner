# SCEB Autoresearch Handoff

本文档给服务器端 autoresearch codex 使用，目标是把当前 **video-only SCEB** 进展、论文故事线、实验优先级和风险分支整理成可执行任务单。


当前最稳妥策略是保留两条故事线：

- **主线 A：Pure Expert Budgeting / Dynamic Expert Skipping**  
  不依赖 audio 实验成功，先把 video-only expert budgeting 做扎实。
- **主线 B：Audio-Aware Expert Budgeting / AV-SCEB 扩展**  
  只作为增强线。audio 不一定要被剪，也不一定要证明 audio layer-level expert risk 可预算；优先把 audio 当成 video expert budget 的 guard / controller。

---

## 0. 当前结论快照

### 0.1 当前方法状态

当前实现方向是 **video-only SCEB**：

```text
target modality: video
target layers: middle transformer layers
protected modalities: text / audio / image / special
budget options: top-4 / top-6 / top-8
risk proxy: router probability geometry
implementation: confidence_dynamic_topk_mask
```

核心规则：

```text
m = 4 if p1 - p2 is high and p4 - p5 boundary is stable
m = 6 if p1 - p2 is medium and p6 - p7 boundary is stable
m = 8 otherwise
```

当前实现仍是 **mask-only**：

```text
zero dropped expert weights
-> renormalize retained weights
-> run original MoE expert loop
```

这说明当前结果更适合证明 **quality / tradeoff 可行性**，还不是最终 compact dispatch 形态。

### 0.2 已有 full-run anchor

以下是当前 research story 里已有的 anchor：

| Method | Accuracy | Prefill Mean | E2E Mean | Note |
|---|---:|---:|---:|---|
| Vanilla | 39.38% | 12540 ms | 15632 ms | original top-8 |
| Global SCEB | 39.68% | 11586 ms | 14516 ms | main current setting |
| Uniform top-6 | 37.8% | 12680 ms | 14450 ms | - |
| Uniform top-4 | 34.9% | 11650ms | 13900ms | - |


---

## 1. 当前 Video-Only SCEB 的核心故事

### 1.1 研究问题

不要把问题说成：

```text
Can we prune experts to reduce latency?
```

推荐表述：

```text
Can we reduce MoE expert compute in omnimodal video inference by assigning
different expert budgets to different token-layer pairs under router-derived
risk constraints?
```

### 1.2 核心 claim

```text
Expert redundancy in video-heavy omnimodal MoE inference is not uniform.
It is structured by modality, layer stage, and router probability geometry.
```

### 1.3 最强 framing

推荐名称：

```text
Structured Expert Budgeting for Efficient Omnimodal MoE Inference
```

或者更直接：

```text
Router-Guided Dynamic Expert Budgeting for Omnimodal MoE Inference
```

方法不要被讲成 threshold pruning。要讲成：

```text
token-layer expert budget allocation
```

### 1.4 当前 eligibility

```text
budgetable region = video tokens × middle layers × confident router states
```

保护规则：

```text
text: full top-8
audio: full top-8
image: full top-8
special/control: full top-8
early layers: full top-8
late layers: full top-8
low-confidence routing states: full top-8
```

---

## 2. 主线 A：Pure Expert Budgeting / Dynamic Expert Skipping

主线 A 是当前最稳的论文主线。它不依赖 audio 成功，也不依赖 token pruning。

### 2.1 主线 A 的目标

证明：

```text
Uniform top-k reduction is brittle.
Fine-grained token-layer expert budgeting gives a better latency-accuracy tradeoff.
```

### 2.2 主线 A 必跑实验

#### A0. Baseline sanity

| Run | Purpose |
|---|---|
| Vanilla top-8 | 原始模型 anchor |
| Global SCEB current config | 复现当前 best setting |

必须输出：

```text
accuracy
prefill mean / median / p90
E2E mean / median / p90
MoE block latency
avg active experts
top-4 / top-6 / top-8 share
answer flip rate vs vanilla
base-correct -> wrong
base-wrong -> correct
```


#### A1. Fixed top-m vs dynamic top-m

必须跑：

| Variant | Rule |
|---|---|
| Fixed video middle top-4 | video middle layers fixed top-4 |
| Fixed video middle top-6 | video middle layers fixed top-6 |
| Dynamic SCEB | top-4/6/8 by router geometry |

目的：

```text
证明 router confidence 不是装饰，而是比 fixed budget 更稳。
```

#### A2. Signal complementarity ablation

必须跑：

| Variant | Rule |
|---|---|
| Decisiveness only | use p1 - p2 only |
| Boundary only | use p_m - p_(m+1) only |
| Both signals | full SCEB |

目的：

```text
p1 - p2 controls dominant-expert confidence.
p_m - p_(m+1) controls truncation-boundary stability.
Two signals cover different failure modes.
```

必须额外输出：

```text
top-4 share
top-6 share
top-8 share
base-correct -> wrong
answer flip rate
```

#### A3. Modality-stage budgetability probe

第一版可以做 matched subset，不一定 full-run。

Probe rule：

```text
Apply fixed top-4 to one target region at a time.
Keep all other token-layer pairs at original top-8.
```

Target regions：

| Region | Why |
|---|---|
| video early | test early visual sensitivity |
| video middle | expected best budgetable region |
| video late | test late semantic integration risk |
| text/special middle | protected modality sanity |
| audio middle | diagnostic only, do not make main claim yet |
| all middle | tests why modality guard is needed |

目的：

```text
证明 budgetability is structured by modality and layer stage.
```

注意：

- audio middle probe 只是 diagnostic，不要预设 audio 可预算。
- 如果 audio middle 掉点明显，反而支持当前 protected audio policy。

#### A5. Confidence-risk monotonicity

Bucket token-layer pairs by:

```text
p1 - p2
p_m - p_(m+1)
combined confidence
```

Measure:

```text
hidden cosine distance
logit KL
answer flip rate
accuracy delta
```

目的：

```text
证明 router probability geometry 可以作为 label-free pruning-risk proxy。
```

### 2.3 主线 A 成功标准

主线 A 可以成稿的最低标准：

```text
Global SCEB accuracy >= Vanilla accuracy - 0.3 pp
Global SCEB E2E latency improves >= 5%
Global SCEB beats Uniform top-6 or top-4 on accuracy-latency tradeoff
Both-signal SCEB has lower flip rate than either single-signal variant
Video-middle probe is safer than video-early/video-late or all-middle
```

如果 latency 改善主要来自 mask-only artifact，需要补充：

```text
MoE block assignment reduction
estimated expert compute reduction
compact dispatch next-step plan
```

---

## 3. 主线 B：Audio-Aware Expert Budgeting / AV-SCEB Extension

主线 B 是增强线，不要作为当前唯一押注。

关键原则：

```text
Audio does not need to be budgeted directly for the story to work.
Audio can first be used as a guard or controller for video expert budgeting.
```

### 3.1 为什么不直接押 audio expert budgeting

当前 profile 里已有判断：

```text
text/audio token 数量少
text/audio expert preference 更尖锐
audio 可能携带 question intent / cross-modal control signal
```

因此：

- 直接剪 audio 或降低 audio expert budget，速度收益可能小。
- 质量风险可能高。
- audio layer-level expert risk 是否可预算，目前没实验证据。

所以主线 B 第一阶段不要证明：

```text
audio tokens can safely use fewer experts
```

而是证明：

```text
audio can improve video expert budget decisions
```

### 3.2 B1：Audio Salience Guard

不动 audio expert budget，只让 audio salience 调节 video budget。

Policy：

```text
if aligned audio salience is high:
    video min_top_m = 6 or 8
else:
    video can use top-4 if router confidence and boundary stability pass
```

需要构造的 audio salience 可先用简单 proxy：

```text
VAD / speech activity
audio energy
audio token norm
audio encoder entropy
audio-text/query relevance
```

目的：

```text
salient audio protects aligned video compute.
low-salience audio allows more aggressive video expert budgeting.
```

这条线不需要 audio 自己被剪。

### 3.3 B2：Audio-Video Consistency Guard

不做 layer-aware audio risk，只做 window-level guard。

按时间窗口定义：

```text
window g = aligned audio segment + video frames/tokens
```

计算：

```text
audio_salience(g)
video_salience(g)
query_relevance(g)
audio_video_consistency(g)
```

Policy：

| Window type | Video expert budget |
|---|---|
| high audio salience + high AV consistency | protect, min top-6/top-8 |
| high audio salience + AV conflict | protect, fallback top-8 |
| low audio salience + low video salience | allow more top-4 |
| low audio salience + high video salience | normal SCEB |

目的：

```text
用 audio-video relation 调节 video expert budget，而不是单纯剪 audio。
```

### 3.4 B3：Query-Level Regime

借鉴 OmniSelect 类思路，但用于 expert budget，而不是 token pruning。

Query regimes：

| Regime | Policy |
|---|---|
| audio-centric | protect audio and aligned video; video top-4 more conservative |
| video-centric | audio protected; video SCEB normal |
| AV reasoning | protect cross-modal aligned windows |
| general visual | video SCEB normal |
| silence/background-heavy | video SCEB more aggressive |

第一版 query regime 可以人工从 benchmark task metadata 或 prompt keyword 分类，不一定训练 classifier。

### 3.5 B4：Audio Expert Budgeting Diagnostic

这是最后再试的增强，不要放在 P0。

第一版只允许：

```text
audio top_m in {6, 8}
```

只有以下 audio token-layer 允许 top-6：

```text
silence
background
low query relevance
high router confidence
stable top-6 boundary
middle layers only
```

禁止第一版：

```text
audio top-4
speech token aggressive budgeting
salient event audio budgeting
audio late-layer budgeting
```

如果 audio diagnostic 失败：

```text
keep audio protected
use audio only as guard/controller
```

这不影响主线 A。

### 3.6 主线 B 成功标准

B1/B2/B3 成功任一即可作为 AV-SCEB extension：

```text
Audio-aware guard reduces answer flip rate vs video-only SCEB
or improves audio-heavy / AV reasoning tasks
while keeping most of the latency gain.
```

如果 audio-aware guard 降低了速度收益但明显降低 flip：

```text
frame as conservative deployment mode
```

如果没有收益：

```text
drop AV-SCEB from main paper
keep audio as protected modality
```

---

## 4. 推荐执行顺序

### Phase 1：把主线 A 跑扎实

Priority:

```text
P0. Vanilla / Global SCEB reproducibility
P1. Fixed top-4/top-6 vs dynamic SCEB
P1. Signal complementarity ablation
P1. Modality-stage budgetability probe
P2. Confidence-risk monotonicity probe
```

只有 Phase 1 支撑住，才继续主线 B。

### Phase 2：做 Audio-as-Guard，不碰 audio expert budget

Priority:

```text
P2. Build audio salience / VAD / energy window stats
P2. Audio salience guard for video min_top_m
P2. Query-level regime for video expert budget
```

### Phase 3：可选 Audio Expert Budgeting

Priority:

```text
P3. Audio top-6-only diagnostic on low-salience audio
P3. Audio/video coupled expert budget
P4. Joint token budget + expert budget
```

Phase 3 是增强线，不要阻塞主线 A。

---

## 5. 输出文件要求

服务器端每轮实验建议产出：

```text
logs/sceb_runs/<run_name>/
  config.json
  metrics_summary.json
  per_task_metrics.csv
  latency_summary.json
  budget_histogram.json
  answer_diff.jsonl
  flip_analysis.md
  report.md
```

必须记录：

```text
model checkpoint
dataset split
sample order / seed
number of evaluated samples
hardware
batch size
generation config
router threshold source
layer range
protected modality policy
```

### 5.1 metrics_summary.json 最小字段

```json
{
  "method": "global_sceb",
  "accuracy": 0.3968,
  "prefill_mean_ms": 11586,
  "e2e_mean_ms": 14516,
  "moe_block_mean_ms": null,
  "avg_active_experts": 7.06,
  "top_m_share": {
    "4": 0.106,
    "6": 0.256,
    "8": 0.638
  },
  "answer_flip_rate_vs_vanilla": null,
  "base_correct_to_wrong": null,
  "base_wrong_to_correct": null
}
```

### 5.2 report.md 必须回答

每个 run 的 `report.md` 至少回答：

```text
1. Compared to vanilla, what changed in accuracy / latency / MoE compute?
2. Compared to uniform top-k, is SCEB a better tradeoff?
3. Which tasks improved or regressed?
4. How many baseline-correct samples flipped wrong?
5. What top-m distribution did the method choose?
6. Did protected modalities remain full top-k?
7. Is the result compatible with the current story?
```

---

## 6. Benchmark
All experiments can be implemented on a single benchmark : `eval_data/OmniVideoBench`


## 7. Files To Read Before Running


```text
implementation/00_profile_signals_to_design.md
implementation/01_quality_floor_fixed_topk_mask.md
implementation/02_confidence_dynamic_topk_mask.md
implementation/03_task_layer_modality_policy.md
implementation/SCEB_RESEARCH_STORY_REVISED.md
implementation/SCEB_EXPECTED_EVIDENCE_DATA.md
implementation/GLOBAL_SCEB_FLOW_CHART.md
```

---

## 8. One-Line Instruction For Server Autoresearch

```text
First validate the pure video-only SCEB / Dynamic Expert Budgeting story with
measured uniform top-k, fixed-vs-dynamic, signal-ablation, and modality-stage
probe experiments. Treat audio-aware AV-SCEB only as a non-blocking extension:
use audio as a guard/controller for video expert budgets before attempting any
direct audio expert budgeting.
```
