"""MoE expert budget wrappers for Qwen3-Omni.

Implements the implementation plan:
- step 01: Quality Floor / Fixed Top-M Mask (fixed_topm.py)
- step 02: Confidence Dynamic Top-K Mask (confidence_dynamic.py)
- step 03: Task / Layer / Modality Policy (mask_policy.py + confidence_dynamic.py)
- step 05: Real Compact Dispatch (compact_dispatch.py)
See implementation_planner/01_*.md through 05_*.md.
"""
from .compact_dispatch import compact_moe_forward
from .mask_policy import ExpertBudgetConfig, MODALITY_CODES
from .fixed_topm import FixedTopMMaskInstaller
from .confidence_dynamic import ConfidenceDynamicMaskInstaller

__all__ = [
    "compact_moe_forward",
    "ExpertBudgetConfig",
    "FixedTopMMaskInstaller",
    "ConfidenceDynamicMaskInstaller",
    "MODALITY_CODES",
]
