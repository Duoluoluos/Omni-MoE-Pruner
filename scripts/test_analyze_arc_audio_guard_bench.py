#!/usr/bin/env python3
"""Regression tests for paired ARC audio-guard benchmark timing."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ANALYZER = Path(__file__).with_name("analyze_arc_audio_guard_bench.py")
SPEC = importlib.util.spec_from_file_location("_arc_audio_guard_analyzer", ANALYZER)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ArcAudioGuardTimingTest(unittest.TestCase):
    def test_partial_pair_uses_identical_window(self):
        native = {1: (137, 0.0), 5: (521, 0.0), 10: (997, 0.0), 15: (1447, 0.0)}
        guard = {1: (124, 0.0), 5: (468, 0.0)}
        common = set(native).intersection(guard)
        self.assertEqual(MODULE.steady_sit(native, common), (96.0, 1, 5))
        self.assertEqual(MODULE.steady_sit(guard, common), (86.0, 1, 5))

    def test_mature_run_uses_later_half(self):
        points = {
            1: (137, 0.0),
            5: (521, 0.0),
            10: (997, 0.0),
            15: (1447, 0.0),
            20: (1882, 0.0),
            25: (2317, 0.0),
        }
        self.assertEqual(MODULE.steady_sit(points), (87.0, 15, 25))


if __name__ == "__main__":
    unittest.main()
