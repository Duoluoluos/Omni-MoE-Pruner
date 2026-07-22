"""Synthetic sanity test for ConfidenceDynamicMaskInstaller."""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moe_expert_budget import ConfidenceDynamicMaskInstaller, ExpertBudgetConfig  # noqa: E402

VIDEO_ID, AUDIO_ID, IMAGE_ID = 151656, 151675, 151655
HIDDEN, NE, TOPK = 16, 8, 8


class Expert(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.w = nn.Linear(h, h, bias=False)

    def forward(self, x):
        return self.w(x)


class SparseMoeBlock(nn.Module):
    def __init__(self, h, ne, topk):
        super().__init__()
        self.num_experts = ne
        self.top_k = topk
        self.norm_topk_prob = True
        self.gate = nn.Linear(h, ne, bias=False)
        self.experts = nn.ModuleList([Expert(h) for _ in range(ne)])

    def forward(self, hidden_states):
        bsz, seq, h = hidden_states.shape
        flat = hidden_states.view(-1, h)
        router_logits = self.gate(flat)
        rw = F.softmax(router_logits, dim=1, dtype=torch.float)
        rw, sel = torch.topk(rw, self.top_k, dim=-1)
        rw = rw / rw.sum(dim=-1, keepdim=True)
        rw = rw.to(flat.dtype)
        out = torch.zeros((bsz * seq, h), dtype=flat.dtype, device=flat.device)
        mask = F.one_hot(sel, num_classes=self.num_experts).permute(2, 1, 0)
        for e in torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero():
            el = self.experts[e]
            idx, top_x = torch.where(mask[e].squeeze(0))
            cs = flat[None, top_x].reshape(-1, h)
            ch = el(cs) * rw[top_x, idx, None]
            out.index_add_(0, top_x, ch.to(flat.dtype))
        return out.reshape(bsz, seq, h), router_logits


def build_fake_model():
    block = SparseMoeBlock(HIDDEN, NE, TOPK)

    class Layer(nn.Module):
        def __init__(self, blk):
            super().__init__()
            self.mlp = blk

    class Thinker(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Layer(block)])
            text_config = SimpleNamespace(
                num_hidden_layers=1,
                num_experts=NE,
                num_experts_per_tok=TOPK,
                norm_topk_prob=True,
            )
            self.config = SimpleNamespace(
                text_config=text_config,
                image_token_id=IMAGE_ID,
                video_token_id=VIDEO_ID,
                audio_token_id=AUDIO_ID,
            )

    return SimpleNamespace(thinker=Thinker()), block


def main():
    torch.manual_seed(0)
    model, block = build_fake_model()
    seq = [10, 11] + [VIDEO_ID] * 8 + [AUDIO_ID]
    input_ids = torch.tensor([seq], dtype=torch.long)
    hidden = torch.randn(1, len(seq), HIDDEN)

    cfg = ExpertBudgetConfig(
        mode="confidence_dynamic_topk_mask",
        target_modality="video",
        protected_modalities=["text", "audio", "image", "special"],
        top_m_options=[4, 6, 8],
        layer_start=0,
        layer_end=1,
        threshold_source="explicit",
        high_margin_threshold=0.10,
        medium_margin_threshold=0.03,
        boundary_threshold=0.0,
        renormalize_weights=True,
    )
    installer = ConfidenceDynamicMaskInstaller(model, cfg)
    assert installer.enabled and installer.thresholds_ready
    assert installer.install() == 1

    installer.begin_sample(0, input_ids)
    with torch.inference_mode():
        block(hidden)
    stats = installer.end_sample()
    hist = stats["top_m_histogram"]
    assert sum(hist.values()) == 8, hist
    print(f"[ok] dynamic histogram totals target tokens: {hist}")

    installer.begin_sample(1, input_ids, task_type="temporal understanding")
    with torch.inference_mode():
        block(hidden)
    stress_stats = installer.end_sample()
    assert stress_stats["task_policy_cluster"] == "stress", stress_stats
    assert stress_stats["task_min_top_m"] == 6, stress_stats
    assert stress_stats["top_m_histogram"]["4"] == 0, stress_stats["top_m_histogram"]
    print(f"[ok] stress clamp: {stress_stats['top_m_histogram']}")

    cfg2 = ExpertBudgetConfig(
        mode="confidence_dynamic_topk_mask",
        target_modality="video",
        top_m_options=[4, 6, 8],
        layer_start=0,
        layer_end=1,
        threshold_source="profile_percentile",
        renormalize_weights=True,
    )
    inst2 = ConfidenceDynamicMaskInstaller(model, cfg2)
    inst2.install()
    inst2.start_calibration()
    inst2.begin_sample(2, input_ids)
    with torch.inference_mode():
        block(hidden)
    inst2.end_sample()
    rep = inst2.finish_calibration()
    assert inst2.thresholds_ready, rep
    print(f"[ok] calibration thresholds: {rep['thresholds']}")

    cfg3 = ExpertBudgetConfig(
        mode="local_prob_threshold_mask",
        target_modality="video",
        protected_modalities=["text", "audio", "image", "special"],
        top_m_options=[4, 6, 8],
        layer_start=0,
        layer_end=1,
        local_prob_threshold=0.12,
        renormalize_weights=True,
    )
    model3, block3 = build_fake_model()
    inst3 = ConfidenceDynamicMaskInstaller(model3, cfg3)
    assert inst3.enabled and inst3.thresholds_ready
    assert inst3.install() == 1
    inst3.begin_sample(3, input_ids)
    with torch.inference_mode():
        block3(hidden)
    local_stats = inst3.end_sample()
    local_hist = local_stats["top_m_histogram"]
    assert sum(local_hist.values()) == 8, local_hist
    assert local_stats["margin_thresholds"]["local_prob"] == 0.12, local_stats["margin_thresholds"]
    print(f"[ok] local-prob histogram totals target tokens: {local_hist}")

    cfg4 = ExpertBudgetConfig(
        mode="confidence_dynamic_topk_mask",
        target_modality="video",
        protected_modalities=["text", "audio", "image", "special"],
        top_m_options=[4, 6, 8],
        layer_start=0,
        layer_end=1,
        threshold_source="explicit",
        high_margin_threshold=0.0,
        medium_margin_threshold=0.0,
        boundary_threshold=0.0,
        renormalize_weights=True,
        task_policy={"default": {"min_top_m": 4}},
        audio_router_conditioning={
            "enabled": True,
            "min_audio_tokens": 1,
            "fragile_margin_threshold": 1.0,
            "fragile_min_top_m": 6,
            "boundary_top_m": 4,
        },
    )
    model4, block4 = build_fake_model()
    inst4 = ConfidenceDynamicMaskInstaller(model4, cfg4)
    assert inst4.enabled and inst4.thresholds_ready
    assert inst4.install() == 1
    inst4.begin_sample(4, input_ids)
    with torch.inference_mode():
        block4(hidden)
    guarded_stats = inst4.end_sample()
    guarded_hist = guarded_stats["top_m_histogram"]
    assert guarded_hist["4"] == 0, guarded_hist
    layer0 = guarded_stats["layer_summaries"][0]
    assert layer0["audio_condition_state_counts"]["fragile"] == 1, layer0
    assert layer0["audio_condition_floor_mean"] == 6.0, layer0
    print(f"[ok] audio-router guard clamps video top-m: {guarded_hist}")

    cfg5 = ExpertBudgetConfig(
        mode="confidence_dynamic_topk_mask",
        target_modality="video",
        protected_modalities=["text", "audio", "image", "special"],
        top_m_options=[4, 6, 8],
        layer_start=0,
        layer_end=1,
        threshold_source="explicit",
        high_margin_threshold=0.0,
        medium_margin_threshold=0.0,
        boundary_threshold=0.0,
        renormalize_weights=True,
        task_policy={"default": {"min_top_m": 4}},
    )
    model5, block5 = build_fake_model()
    inst5 = ConfidenceDynamicMaskInstaller(model5, cfg5)
    assert inst5.enabled and inst5.thresholds_ready
    assert inst5.install() == 1
    keep = torch.tensor(
        [True, True, True, False, True, False, True, False, True, False, True],
        dtype=torch.bool,
    )
    assert len(keep) == len(seq)
    model5.thinker._sceb_omnizip_global_mask = keep
    inst5.begin_sample(5, input_ids)
    with torch.inference_mode():
        block5(hidden[:, keep, :])
    zipped_stats = inst5.end_sample()
    zipped_hist = zipped_stats["top_m_histogram"]
    assert sum(zipped_hist.values()) == 4, zipped_hist
    assert zipped_stats["target_token_count"] == 4, zipped_stats
    assert zipped_stats["protected_token_count"] == int(keep.sum().item()) - 4, zipped_stats
    print(f"[ok] OmniZip-compressed modality mask aligns target tokens: {zipped_hist}")
    print("ALL DYNAMIC TESTS PASSED")


if __name__ == "__main__":
    main()
