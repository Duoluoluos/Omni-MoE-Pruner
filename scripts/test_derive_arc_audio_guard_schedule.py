#!/usr/bin/env python3
"""Fast synthetic tests for ARC audio-guard calibration and speed budgeting."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("derive_arc_audio_guard_schedule.py")
SPEC = importlib.util.spec_from_file_location("_derive_arc_audio_guard", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ArcAudioGuardScheduleTest(unittest.TestCase):
    def test_maximizes_protection_with_speed_headroom(self):
        rates = [0.18, 0.09, 0.05, 0.09, 0.01, 0.01, 0.00, 0.02, 0.00, 0.03,
                 0.22, 0.05, 0.45, 0.46, 0.06, 0.07, 0.00, 0.00, 0.00, 0.00]
        stats = {
            layer: {"audio_guard_activation_rate": rate}
            for layer, rate in zip(range(12, 32), rates)
        }
        schedule, speed = MODULE.select_static_schedule(
            stats,
            native_top_k=8,
            reduced_top_k=4,
            protected_top_k=6,
            total_layers=48,
            target_speedup_percent=8.0,
            prediction_headroom_percent=0.2,
            anchor_cut_units=96,
            anchor_speedup_percent=10.7,
        )
        self.assertEqual(speed["protected_layers_ranked"], [25, 24, 22])
        self.assertEqual({layer for layer, k in schedule.items() if k == 6}, {22, 24, 25})
        self.assertEqual(speed["cut_topk_layer_units"], 74)
        self.assertAlmostEqual(speed["predicted_speedup_percent"], 8.24791667)
        self.assertAlmostEqual(speed["effective_average_top_k"], 6.45833333)

    def test_fourth_protected_layer_would_break_headroom(self):
        predicted_four = 10.7 * 72 / 96
        self.assertLess(predicted_four, 8.2)
        self.assertGreaterEqual(predicted_four, 8.0)


if __name__ == "__main__":
    unittest.main()
