"""Mask policy: configuration + per-token target/protect resolution.

Modality codes mirror profile_omnivideobench_qwen3/run_omnivideobench_profile.py:
    text=0, video=1, audio=2, image=3, decode=4
"text" here covers both regular text and special/control tokens.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


MODALITY_CODES = {"text": 0, "video": 1, "audio": 2, "image": 3, "decode": 4, "special": 0}
MODALITY_NAMES = {0: "text", 1: "video", 2: "audio", 3: "image", 4: "decode"}


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

    def target_modality_code(self) -> int:
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
    if int(target_code) in protected:
        return np.zeros_like(modalities, dtype=bool)
    return modalities == int(target_code)
