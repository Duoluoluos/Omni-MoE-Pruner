"""MoE expert budget wrappers for Qwen3-Omni.

Implements step 01 of the implementation plan: Quality Floor / Fixed Top-M Mask.
See implementation_planner/01_quality_floor_fixed_topk_mask.md.
"""
from .mask_policy import ExpertBudgetConfig, MODALITY_CODES
from .fixed_topm import FixedTopMMaskInstaller

__all__ = [
    "ExpertBudgetConfig",
    "FixedTopMMaskInstaller",
    "MODALITY_CODES",
]
