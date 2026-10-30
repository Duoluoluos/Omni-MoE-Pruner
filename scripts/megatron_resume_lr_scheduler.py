"""Restore the Megatron LR scheduler position for adapter-only resumes.

The original LoRA checkpoints deliberately omit optimizer, RNG, and scheduler
state.  Megatron-SWIFT still restores the completed iteration and consumed
samples from ``common.pt`` when ``--finetune false --adapters <checkpoint>``
is used, but it otherwise initializes a fresh scheduler at step zero.  This
external plugin advances that fresh scheduler to the restored sample count.
"""

from swift.megatron.trainers.base import BaseMegatronTrainer
from swift.utils import get_logger


_original_load_checkpoint = BaseMegatronTrainer._load_checkpoint
_logger = get_logger()


def _load_checkpoint_with_scheduler_resume(self):
    _original_load_checkpoint(self)
    args = self.args
    if not args.finetune and args.adapters and args.mcore_adapter is None:
        consumed_samples = self.state.consumed_train_samples
        if consumed_samples:
            self.opt_param_scheduler.step(increment=consumed_samples)
            _logger.info(
                'Advanced fresh optimizer LR scheduler to %s consumed samples '
                'for adapter-only resume.', consumed_samples)


BaseMegatronTrainer._load_checkpoint = _load_checkpoint_with_scheduler_resume
