"""MoE expert budget wrappers for Qwen3-Omni.

Implements the implementation plan:
- step 01: Quality Floor / Fixed Top-M Mask (fixed_topm.py)
- step 02: Confidence Dynamic Top-K Mask (confidence_dynamic.py)
- step 03: Task / Layer / Modality Policy (mask_policy.py + confidence_dynamic.py)
See implementation_planner/01_*.md through 03_*.md.

Note: the "step 05: real compact dispatch" experiment was removed after it was
measured to yield no benefit in either training or inference (variable-length
dispatch is dominated by GPU->CPU sync and sub-linear grouped-GEMM speedups).
The project uses the mask-only path exclusively.
"""
from .mask_policy import ExpertBudgetConfig, MODALITY_CODES
from .fixed_topm import FixedTopMMaskInstaller
from .confidence_dynamic import ConfidenceDynamicMaskInstaller

__all__ = [
    "ExpertBudgetConfig",
    "FixedTopMMaskInstaller",
    "ConfidenceDynamicMaskInstaller",
    "MODALITY_CODES",
]
