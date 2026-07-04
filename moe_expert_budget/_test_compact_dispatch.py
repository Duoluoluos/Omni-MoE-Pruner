"""Synthetic tests for real compact expert dispatch."""
import sys
from pathlib import Path
from types import SimpleNamespace
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moe_expert_budget import ConfidenceDynamicMaskInstaller, ExpertBudgetConfig  # noqa: E402
from moe_expert_budget.compact_dispatch import compact_moe_forward  # noqa: E402

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
        rw = rw / rw.sum(dim=-1, keepdim=True).clamp_min(1e-20)
        out = reference_loop(self.experts, flat, sel, rw, self.num_experts)
        return out.reshape(bsz, seq, h), router_logits


def reference_loop(experts, flat, selected, weights, num_experts):
    out = torch.zeros_like(flat)
    expert_mask = F.one_hot(selected, num_classes=num_experts).permute(2, 1, 0)
    for expert_idx in torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero():
        expert_layer = experts[expert_idx]
        idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
        current_state = flat[None, top_x].reshape(-1, flat.shape[-1])
        current_hidden_states = expert_layer(current_state) * weights[top_x, idx, None]
        out.index_add_(0, top_x, current_hidden_states.to(flat.dtype))
    return out


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


def router_topk(block, hidden):
    flat = hidden.reshape(-1, hidden.shape[-1])
    rw = F.softmax(block.gate(flat), dim=1, dtype=torch.float)
    rw, sel = torch.topk(rw, TOPK, dim=-1)
    rw = rw / rw.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    return flat, rw, sel


def assert_close(a, b, tol=1e-5):
    diff = (a - b).abs().max().item()
    assert diff <= tol, f"max diff {diff} > {tol}"


def main():
    torch.manual_seed(0)
    model, block = build_fake_model()
    hidden = torch.randn(1, 11, HIDDEN)
    flat, rw, sel = router_topk(block, hidden)

    ref = reference_loop(block.experts, flat, sel, rw, NE)
    compact_full, metrics = compact_moe_forward(
        experts=block.experts,
        flat_hidden_states=flat,
        selected_experts=sel,
        routing_weights=rw,
        num_experts=NE,
    )
    assert_close(ref, compact_full)
    assert metrics.assignment_count_before == flat.shape[0] * TOPK
    assert metrics.assignment_count_after == metrics.assignment_count_before
    print("[ok] compact full top-k matches reference")

    for top_m in (1, 4, 6):
        keep = torch.zeros_like(sel, dtype=torch.bool)
        keep[:, :top_m] = True
        masked = rw.clone()
        masked[~keep] = 0.0
        masked = masked / masked.sum(dim=-1, keepdim=True).clamp_min(1e-20)
        mask_ref = reference_loop(block.experts, flat, sel, masked, NE)
        compact, metrics = compact_moe_forward(
            experts=block.experts,
            flat_hidden_states=flat,
            selected_experts=sel,
            routing_weights=masked,
            num_experts=NE,
            keep_mask=keep,
        )
        assert compact.shape == flat.shape
        assert metrics.assignment_count_after == flat.shape[0] * top_m
        assert_close(mask_ref, compact)
        print(f"[ok] compact top_m={top_m} matches mask-only reference")

    empty_keep = torch.zeros_like(sel, dtype=torch.bool)
    empty, metrics = compact_moe_forward(
        experts=block.experts,
        flat_hidden_states=flat,
        selected_experts=sel,
        routing_weights=rw,
        num_experts=NE,
        keep_mask=empty_keep,
    )
    assert empty.shape == flat.shape
    assert empty.abs().max().item() == 0.0
    assert metrics.assignment_count_after == 0
    print("[ok] empty compact assignment list is safe")

    seq = [10, 11] + [VIDEO_ID] * 8 + [AUDIO_ID]
    input_ids = torch.tensor([seq], dtype=torch.long)
    cfg = ExpertBudgetConfig(
        mode="confidence_dynamic_compact_dispatch",
        compact_dispatch=True,
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
    assert installer.compact_dispatch
    installer.install()
    installer.begin_sample(0, input_ids, task_type="temporal understanding")
    with torch.inference_mode():
        out, _ = block(hidden)
    stats = installer.end_sample()
    assert out.shape == hidden.shape
    assert stats["compact_dispatch"] is True
    assert stats["top_m_histogram"]["4"] == 0
    assert stats["layer_summaries"][0]["compact_assignment_count_after"] < stats["layer_summaries"][0]["compact_assignment_count_before"]
    print("[ok] compact installer records reduced assignments and stress clamp")

    # Direct wrapper-level equivalence: mask-only and compact should produce the
    # same hidden states under identical weights, thresholds, input, and task.
    model_mask, block_mask = build_fake_model()
    model_compact, block_compact = build_fake_model()
    block_compact.load_state_dict(copy.deepcopy(block_mask.state_dict()))
    cfg_mask = ExpertBudgetConfig(
        mode="confidence_dynamic_topk_mask",
        compact_dispatch=False,
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
    cfg_compact = copy.deepcopy(cfg_mask)
    cfg_compact.mode = "confidence_dynamic_compact_dispatch"
    cfg_compact.compact_dispatch = True
    inst_mask = ConfidenceDynamicMaskInstaller(model_mask, cfg_mask)
    inst_compact = ConfidenceDynamicMaskInstaller(model_compact, cfg_compact)
    inst_mask.install()
    inst_compact.install()
    inst_mask.begin_sample(0, input_ids, task_type="temporal understanding")
    inst_compact.begin_sample(0, input_ids, task_type="temporal understanding")
    with torch.inference_mode():
        out_mask, logits_mask = block_mask(hidden)
        out_compact, logits_compact = block_compact(hidden)
    stats_mask = inst_mask.end_sample()
    stats_compact = inst_compact.end_sample()
    assert_close(logits_mask, logits_compact)
    assert_close(out_mask, out_compact)
    assert stats_mask["top_m_histogram"] == stats_compact["top_m_histogram"]
    assert stats_mask["assignment_count_effective"] == stats_compact["assignment_count_effective"]
    print("[ok] compact wrapper output matches mask-only wrapper under identical policy")

    if torch.cuda.is_available():
        for dtype in (torch.float16, torch.bfloat16):
            model2, block2 = build_fake_model()
            block2.cuda().to(dtype=dtype)
            hidden2 = torch.randn(1, 7, HIDDEN, device="cuda", dtype=dtype)
            flat2, rw2, sel2 = router_topk(block2, hidden2)
            ref2 = reference_loop(block2.experts, flat2, sel2, rw2, NE)
            compact2, _ = compact_moe_forward(
                experts=block2.experts,
                flat_hidden_states=flat2,
                selected_experts=sel2,
                routing_weights=rw2,
                num_experts=NE,
            )
            assert_close(ref2.float(), compact2.float(), tol=5e-3)
        print("[ok] cuda fp16/bf16 compact full top-k within tolerance")

    print("ALL COMPACT DISPATCH TESTS PASSED")


if __name__ == "__main__":
    main()
