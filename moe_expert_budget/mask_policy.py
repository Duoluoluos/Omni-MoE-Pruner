"""Mask policy: configuration + per-token target/protect resolution.

Modality codes mirror profile_omnivideobench_qwen3/run_omnivideobench_profile.py:
    text=0, video=1, audio=2, image=3, decode=4
"text" here covers both regular text and special/control tokens.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


MODALITY_CODES = {"text": 0, "video": 1, "audio": 2, "image": 3, "decode": 4, "special": 0}
MODALITY_NAMES = {0: "text", 1: "video", 2: "audio", 3: "image", 4: "decode"}


def _default_layer_policy() -> dict[str, dict[str, Any]]:
    return {
        "early": {"range_ratio": [0.0, 0.25], "mode": "full"},
        "middle": {"range_ratio": [0.25, 0.67], "mode": "dynamic"},
        "late": {"range_ratio": [0.67, 1.0], "mode": "full"},
    }


def _default_task_policy() -> dict[str, dict[str, Any]]:
    return {
        "stress": {
            "tasks": [
                "temporal understanding",
                "spatial understanding",
                "reference reasoning",
                "counting",
            ],
            "min_top_m": 6,
        },
        "medium": {
            "tasks": [
                "causal reasoning",
                "hypothetical reasoning",
                "relationship reasoning",
                "ego reasoning",
            ],
            "min_top_m": 4,
        },
        "perception": {
            "tasks": [
                "fine-grained perception",
                "attribute comparison",
                "sentiment analysis",
            ],
            "min_top_m": 4,
        },
        "audio": {
            "tasks": ["background&music understanding"],
            "min_top_m": 4,
        },
        "default": {"min_top_m": 4},
    }


@dataclass
class ExpertBudgetConfig:
    enabled: bool = True
    mode: str = "fixed_topm_mask"
    target_modality: str = "video"
    protected_modalities: list[str] = field(default_factory=lambda: ["text", "audio", "image", "special"])
    original_top_k: int = 8
    top_m: int = 4
    layer_start: int | None = 12
    layer_end: int | None = 32
    layer_start_ratio: float | None = None
    layer_end_ratio: float | None = None
    renormalize_weights: bool = True
    apply_during_decode: bool = False
    note: str = ""

    # --- step 02: confidence_dynamic_topk_mask ---
    # Candidate budgets, sorted descending in priority (high conf -> small top_m).
    top_m_options: list[int] = field(default_factory=lambda: [4, 6, 8])
    # Where thresholds come from: "profile_percentile" => calibrate on N samples;
    # "explicit" => use the *_threshold values below directly.
    threshold_source: str = "profile_percentile"
    high_margin_percentile: float = 70.0
    medium_margin_percentile: float = 40.0
    boundary_margin_percentile: float = 50.0
    # Explicit thresholds (used when threshold_source == "explicit", or filled in
    # after calibration so the run is reproducible).
    high_margin_threshold: float | None = None
    medium_margin_threshold: float | None = None
    boundary_threshold: float | None = None
    # Number of dataset samples to scan during calibration of the percentiles.
    calibrate_samples: int = 40
    # "global" uses one calibrated threshold triplet for all dynamic layers.
    # "per_layer" calibrates a separate triplet for each dynamic layer.
    threshold_granularity: str = "global"
    per_layer_thresholds: dict[str, dict[str, float]] = field(default_factory=dict)

    # --- step 03: task / layer / modality policy ---
    policy_version: str = "omnivideobench_expert_budget_v1"
    layer_policy: dict[str, dict[str, Any]] = field(default_factory=_default_layer_policy)
    task_policy: dict[str, dict[str, Any]] = field(default_factory=_default_task_policy)
    # Optional task x layer overrides. Entries are matched by task name and can
    # force selected layers back to full top-k even if layer_policy marks them
    # dynamic. This supports targeted protection such as keeping late-middle
    # layers full only for hypothetical/relationship reasoning.
    task_layer_policy: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Optional label-free query guard. When enabled, top-m floors are selected
    # from input text and modality presence instead of benchmark task labels.
    #
    # Shape:
    # {
    #   "enabled": true,
    #   "default_min_top_m": 4,
    #   "clusters": {
    #     "audio_reference": {
    #       "min_top_m": 6,
    #       "requires_audio": true,
    #       "any_keywords": ["hear", "sound"],
    #       "any_regex": ["\\bsaid\\b"]
    #     }
    #   }
    # }
    query_policy: dict[str, Any] = field(default_factory=dict)
    # Optional audio-router conditioning for video-target SCEB. This keeps audio
    # tokens protected, but uses their same-layer router geometry as a guard for
    # video-token expert budgets.
    audio_router_conditioning: dict[str, Any] = field(default_factory=dict)

    # --- related-work baseline: local router-probability threshold ---
    # MoDES-style local-probability baseline. For target tokens, choose the
    # smallest configured top-m bucket that covers the selected experts whose
    # router probabilities exceed this threshold. Modalities/layers/tasks are
    # still governed by the same SCEB protection policy.
    local_prob_threshold: float | None = None

    @classmethod
    def from_json(cls, path: str | Path) -> "ExpertBudgetConfig":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        known = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    @classmethod
    def disabled(cls) -> "ExpertBudgetConfig":
        return cls(enabled=False)

    def resolved_layer_range(self, num_layers: int) -> tuple[int, int]:
        if self.layer_start is not None and self.layer_end is not None:
            ls = max(0, min(int(self.layer_start), num_layers))
            le = max(ls, min(int(self.layer_end), num_layers))
            return ls, le
        ls_r = self.layer_start_ratio if self.layer_start_ratio is not None else 0.25
        le_r = self.layer_end_ratio if self.layer_end_ratio is not None else 0.67
        ls = int(math.floor(ls_r * num_layers))
        le = int(math.floor(le_r * num_layers))
        ls = max(0, min(ls, num_layers))
        le = max(ls, min(le, num_layers))
        return ls, le

    def target_layer_set(self, num_layers: int) -> set[int]:
        ls, le = self.resolved_layer_range(num_layers)
        return set(range(ls, le))

    def layer_modes(self, num_layers: int) -> dict[int, str]:
        """Resolve layer policy ratios to a per-layer mode map.

        Backward compatible default: if no explicit ``layer_policy`` is present,
        use the legacy ``layer_start/end`` middle-layer range as dynamic and keep
        all other layers full top-k.
        """
        modes = {idx: "full" for idx in range(num_layers)}
        if self.layer_start is not None and self.layer_end is not None:
            for idx in self.target_layer_set(num_layers):
                modes[idx] = "dynamic"
            return modes
        if not self.layer_policy:
            for idx in self.target_layer_set(num_layers):
                modes[idx] = "dynamic"
            return modes
        for name, spec in self.layer_policy.items():
            mode = str(spec.get("mode", "full")).lower()
            if mode not in {"full", "dynamic", "fixed"}:
                raise ValueError(f"Unknown layer_policy mode {mode!r} for {name!r}")
            if "layers" in spec:
                layers = [int(x) for x in spec["layers"]]
            else:
                ratios = spec.get("range_ratio")
                if ratios is None:
                    continue
                if len(ratios) != 2:
                    raise ValueError(f"layer_policy {name!r} range_ratio must have 2 values")
                start = max(0.0, min(float(ratios[0]), 1.0))
                end = max(start, min(float(ratios[1]), 1.0))
                ls = max(0, min(int(math.floor(start * num_layers)), num_layers))
                le = max(ls, min(int(math.floor(end * num_layers)), num_layers))
                layers = list(range(ls, le))
            for idx in layers:
                if 0 <= idx < num_layers:
                    modes[idx] = mode
        return modes

    def target_layer_set_from_policy(self, num_layers: int, mode: str = "dynamic") -> set[int]:
        want = mode.lower()
        return {idx for idx, layer_mode in self.layer_modes(num_layers).items() if layer_mode == want}

    def task_min_top_m(self, task_type: str | None) -> tuple[int, str]:
        """Return the minimum top-m for an OmniVideoBench task and the matched cluster."""
        default_min = int((self.task_policy.get("default") or {}).get("min_top_m", self.top_m))
        if not task_type:
            return default_min, "default"
        normalized = str(task_type).strip().lower()
        for cluster, spec in self.task_policy.items():
            if cluster == "default":
                continue
            tasks = {str(t).strip().lower() for t in spec.get("tasks", [])}
            if normalized in tasks:
                return int(spec.get("min_top_m", default_min)), cluster
        return default_min, "default"

    def query_policy_enabled(self) -> bool:
        return bool((self.query_policy or {}).get("enabled", False))

    def query_min_top_m(
        self,
        query_text: str | None,
        audio_token_count: int = 0,
    ) -> tuple[int, str, list[str]]:
        """Return a label-free min top-m from query text and audio presence.

        This deliberately does not inspect benchmark task labels. It is meant
        for fair inference-time guards where only input text and modality
        structure are available.
        """
        policy = self.query_policy or {}
        default_min = int(policy.get("default_min_top_m", self.top_m))
        if not policy.get("enabled", False):
            return default_min, "query_default", []

        text = str(query_text or "").strip().lower()
        clusters = policy.get("clusters") or {}
        ordered = sorted(
            clusters.items(),
            key=lambda kv: int((kv[1] or {}).get("priority", 1000)),
        )
        for cluster, raw_spec in ordered:
            spec = raw_spec or {}
            if bool(spec.get("requires_audio", False)) and audio_token_count <= 0:
                continue
            min_audio_tokens = spec.get("min_audio_tokens")
            if min_audio_tokens is not None and audio_token_count < int(min_audio_tokens):
                continue

            exclude = [str(x).lower() for x in spec.get("exclude_keywords", [])]
            if any(term and term in text for term in exclude):
                continue

            all_terms = [str(x).lower() for x in spec.get("all_keywords", [])]
            if all_terms and not all(term in text for term in all_terms):
                continue

            matches: list[str] = []
            any_terms = [str(x).lower() for x in spec.get("any_keywords", [])]
            matches.extend([term for term in any_terms if term and term in text])
            for pattern in spec.get("any_regex", []):
                pattern_s = str(pattern)
                try:
                    if re.search(pattern_s, text):
                        matches.append(f"re:{pattern_s}")
                except re.error:
                    continue

            if any_terms or spec.get("any_regex"):
                if not matches:
                    continue
            elif not all_terms:
                continue

            return int(spec.get("min_top_m", default_min)), str(cluster), matches[:16]

        return default_min, "query_default", []

    def task_layer_full_layers(self, task_type: str | None, num_layers: int) -> set[int]:
        """Resolve task-specific layer overrides that force full top-k.

        Config shape:

        ``{"name": {"tasks": [...], "mode": "full", "layers": [24, ...]}}``

        ``range_ratio`` can be used instead of explicit ``layers`` and follows
        the same floor-based layer indexing as ``layer_policy``.
        """
        if not task_type or not self.task_layer_policy:
            return set()
        normalized = str(task_type).strip().lower()
        full_layers: set[int] = set()
        for _name, spec in self.task_layer_policy.items():
            mode = str(spec.get("mode", "full")).lower()
            if mode != "full":
                continue
            tasks = {str(t).strip().lower() for t in spec.get("tasks", [])}
            if normalized not in tasks:
                continue
            if "layers" in spec:
                layers = [int(x) for x in spec["layers"]]
            else:
                ratios = spec.get("range_ratio")
                if ratios is None:
                    continue
                if len(ratios) != 2:
                    raise ValueError(f"task_layer_policy {_name!r} range_ratio must have 2 values")
                start = max(0.0, min(float(ratios[0]), 1.0))
                end = max(start, min(float(ratios[1]), 1.0))
                ls = max(0, min(int(math.floor(start * num_layers)), num_layers))
                le = max(ls, min(int(math.floor(end * num_layers)), num_layers))
                layers = list(range(ls, le))
            full_layers.update(idx for idx in layers if 0 <= idx < num_layers)
        return full_layers

    def target_modality_code(self) -> int:
        if str(self.target_modality).lower() == "all":
            return -1
        if self.target_modality not in MODALITY_CODES:
            raise ValueError(f"Unknown target_modality {self.target_modality}")
        return MODALITY_CODES[self.target_modality]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_modality_vector(input_ids_1d: np.ndarray, token_ids: dict[str, int]) -> np.ndarray:
    """Mirror Qwen3ProfileCollector.begin_sample modality detection.

    Anything not in {video, audio, image} stays as text (covers special/control tokens).
    """
    modalities = np.full(input_ids_1d.shape[0], MODALITY_CODES["text"], dtype=np.int8)
    if "video" in token_ids:
        modalities[input_ids_1d == token_ids["video"]] = MODALITY_CODES["video"]
    if "audio" in token_ids:
        modalities[input_ids_1d == token_ids["audio"]] = MODALITY_CODES["audio"]
    if "image" in token_ids:
        modalities[input_ids_1d == token_ids["image"]] = MODALITY_CODES["image"]
    return modalities


def resolve_target_token_mask(
    modalities: np.ndarray,
    target_code: int,
    protected_codes: Iterable[int],
) -> np.ndarray:
    """True where the token should have its top-k reduced; False where it must keep full top-k."""
    protected = set(int(c) for c in protected_codes)
    if int(target_code) == -1:
        if not protected:
            return np.ones_like(modalities, dtype=bool)
        return ~np.isin(modalities, list(protected))
    if int(target_code) in protected:
        return np.zeros_like(modalities, dtype=bool)
    return modalities == int(target_code)
