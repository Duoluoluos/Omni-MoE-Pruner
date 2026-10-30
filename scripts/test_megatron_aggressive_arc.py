#!/usr/bin/env python3
"""Regression tests for the aggressive ARC per-layer top-k patch.

The important lifecycle case mirrors Megatron-Core 0.18: ``MoELayer`` is
constructed with ``layer_number=None`` and ``TransformerLayer`` supplies the
real number later via ``set_layer_number``.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PLUGIN = Path(__file__).with_name("megatron_aggressive_arc.py")


def _load_plugin(mid_topk: int = 4, start: int = 10, end: int = 34):
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

    module_names = (
        "megatron",
        "megatron.core",
        "megatron.core.transformer",
        "megatron.core.transformer.moe",
        "megatron.core.transformer.moe.moe_layer",
    )
    fake_modules = {name: types.ModuleType(name) for name in module_names}
    fake_modules[module_names[-1]].MoELayer = FakeMoELayer
    env = {
        "AGGR_ARC_MID_TOPK": str(mid_topk),
        "AGGR_ARC_LAYER_START": str(start),
        "AGGR_ARC_LAYER_END": str(end),
        "RANK": "1",  # Keep the unit-test output quiet.
    }
    with patch.dict(sys.modules, fake_modules), patch.dict(os.environ, env, clear=False):
        spec = importlib.util.spec_from_file_location("_aggr_arc_test_plugin", PLUGIN)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
    return FakeMoELayer


class AggressiveArcPatchTest(unittest.TestCase):
    def test_post_construction_layer_number_reduces_only_target_band(self):
        moe_layer = _load_plugin()
        shared = SimpleNamespace(moe_router_topk=8)

        early = moe_layer(shared)
        middle = moe_layer(shared)
        late = moe_layer(shared)

        early.set_layer_number(10)   # zero-based layer 9
        middle.set_layer_number(11)  # zero-based layer 10: first reduced layer
        late.set_layer_number(35)    # zero-based layer 34

        self.assertIs(early.config, shared)
        self.assertIs(late.config, shared)
        self.assertEqual(early.router.topk, 8)
        self.assertEqual(late.router.topk, 8)

        self.assertIsNot(middle.config, shared)
        self.assertEqual(shared.moe_router_topk, 8)
        self.assertEqual(middle.config.moe_router_topk, 4)
        self.assertIs(middle.router.config, middle.config)
        self.assertEqual(middle.router.topk, 4)
        self.assertIs(middle.token_dispatcher.config, middle.config)
        self.assertEqual(middle.token_dispatcher.router_topk, 4)

    def test_direct_layer_number_is_also_supported(self):
        moe_layer = _load_plugin()
        shared = SimpleNamespace(moe_router_topk=8)
        middle = moe_layer(shared, layer_number=34)  # zero-based layer 33
        self.assertEqual(middle.router.topk, 4)
        self.assertEqual(shared.moe_router_topk, 8)

    def test_refuses_to_increase_native_topk(self):
        moe_layer = _load_plugin(mid_topk=9)
        layer = moe_layer(SimpleNamespace(moe_router_topk=8))
        with self.assertRaisesRegex(RuntimeError, "refusing to add experts"):
            layer.set_layer_number(11)


if __name__ == "__main__":
    unittest.main()
