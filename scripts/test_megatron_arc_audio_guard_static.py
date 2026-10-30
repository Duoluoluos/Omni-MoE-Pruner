#!/usr/bin/env python3
"""Regression tests for the calibrated ARC audio-guard static top-k plugin."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PLUGIN = Path(__file__).with_name("megatron_arc_audio_guard_static.py")
REAL_POLICY = (
    PLUGIN.parents[1] / "moe_expert_budget/configs/arc_audio_guard_static_topk_v1.json"
)


def _load_plugin(policy_mutator=None):
    class FakeMoELayer:
        def __init__(self, config, layer_number=None):
            self.config = config
            self.layer_number = layer_number
            self.router = SimpleNamespace(config=config, topk=config.moe_router_topk)
            self.token_dispatcher = SimpleNamespace(
                config=config, router_topk=config.moe_router_topk
            )

        def set_layer_number(self, layer_number):
            self.layer_number = layer_number
            self.router.layer_number = layer_number

    policy = json.loads(REAL_POLICY.read_text())
    if policy_mutator:
        policy_mutator(policy)
    temporary = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    try:
        json.dump(policy, temporary)
        temporary.close()
        module_names = (
            "megatron",
            "megatron.core",
            "megatron.core.transformer",
            "megatron.core.transformer.moe",
            "megatron.core.transformer.moe.moe_layer",
        )
        fake_modules = {name: types.ModuleType(name) for name in module_names}
        fake_modules[module_names[-1]].MoELayer = FakeMoELayer
        env = {"ARC_AUDIO_GUARD_POLICY": temporary.name, "RANK": "1"}
        with patch.dict(sys.modules, fake_modules), patch.dict(os.environ, env, clear=False):
            spec = importlib.util.spec_from_file_location("_arc_audio_guard_test", PLUGIN)
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
        return FakeMoELayer
    finally:
        Path(temporary.name).unlink(missing_ok=True)


class ArcAudioGuardStaticPatchTest(unittest.TestCase):
    def test_delayed_layer_number_applies_only_original_arc_middle(self):
        moe_layer = _load_plugin()
        shared = SimpleNamespace(moe_router_topk=8)
        early = moe_layer(shared)
        ordinary_middle = moe_layer(shared)
        protected_middle = moe_layer(shared)
        late = moe_layer(shared)

        early.set_layer_number(12)             # zero-based 11: outside
        ordinary_middle.set_layer_number(13)   # zero-based 12: top-4
        protected_middle.set_layer_number(23)  # zero-based 22: protected top-6
        late.set_layer_number(33)              # zero-based 32: outside

        self.assertIs(early.config, shared)
        self.assertIs(late.config, shared)
        self.assertEqual(early.router.topk, 8)
        self.assertEqual(late.router.topk, 8)
        self.assertEqual(ordinary_middle.router.topk, 4)
        self.assertEqual(protected_middle.router.topk, 6)
        self.assertEqual(shared.moe_router_topk, 8)
        for layer in (ordinary_middle, protected_middle):
            self.assertIs(layer.router.config, layer.config)
            self.assertIs(layer.token_dispatcher.config, layer.config)
            self.assertEqual(layer.token_dispatcher.router_topk, layer.router.topk)

    def test_exact_schedule_has_three_audio_protected_layers(self):
        policy = json.loads(REAL_POLICY.read_text())
        schedule = {int(k): v for k, v in policy["layer_top_k"].items()}
        self.assertEqual(set(schedule), set(range(12, 32)))
        self.assertEqual({layer for layer, k in schedule.items() if k == 6}, {22, 24, 25})
        self.assertEqual(sum(k == 4 for k in schedule.values()), 17)
        self.assertGreaterEqual(policy["speed_model"]["predicted_speedup_percent"], 8.2)

    def test_direct_constructor_layer_number_is_supported(self):
        moe_layer = _load_plugin()
        shared = SimpleNamespace(moe_router_topk=8)
        layer = moe_layer(shared, layer_number=25)  # zero-based 24: protected
        self.assertEqual(layer.router.topk, 6)
        self.assertEqual(shared.moe_router_topk, 8)

    def test_rejects_any_schedule_entry_outside_middle(self):
        def mutate(policy):
            policy["layer_top_k"]["32"] = 4

        with self.assertRaisesRegex(RuntimeError, "outside=\\[32\\]"):
            _load_plugin(mutate)

    def test_refuses_composition_with_non_native_topk(self):
        moe_layer = _load_plugin()
        layer = moe_layer(SimpleNamespace(moe_router_topk=6))
        with self.assertRaisesRegex(RuntimeError, "do not compose"):
            layer.set_layer_number(13)


if __name__ == "__main__":
    unittest.main()
