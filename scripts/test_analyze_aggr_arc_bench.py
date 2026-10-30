#!/usr/bin/env python3
"""Unit tests for paired-window aggressive ARC benchmark timing."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ANALYZER = Path(__file__).with_name("analyze_aggr_arc_bench.py")
SPEC = importlib.util.spec_from_file_location("_aggr_arc_analyzer", ANALYZER)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class SteadySecondsPerIterationTest(unittest.TestCase):
    def test_partial_pair_uses_the_same_iteration_window(self):
        native = {
            1: (137, 0.0),
            5: (521, 0.0),
            10: (997, 0.0),
            15: (1447, 0.0),
            20: (1882, 0.0),
            25: (2317, 0.0),
        }
        aggressive = {1: (124, 0.0), 5: (450, 0.0)}
        common = set(native).intersection(aggressive)

        self.assertEqual(MODULE.steady_sit(native, common), (96.0, 1, 5))
        self.assertEqual(MODULE.steady_sit(aggressive, common), (81.5, 1, 5))

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
