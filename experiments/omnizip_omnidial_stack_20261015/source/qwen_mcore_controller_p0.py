"""P0 controller: identical policies, MCore GroupedMLP without row telemetry."""
from qwen_runtime_backend import CleanQwenController, prefix_forward
from qwen_mcore_experts_p0 import MCoreGroupedExpertsP0
import torch


class MCoreQwenControllerP0(CleanQwenController):
    def __init__(self, model, scope=(12,32), tau_reduce=None, progress=None):
        blocks=[m for m in model.thinker.model.modules()
                if m.__class__.__name__=='Qwen3OmniMoeThinkerTextSparseMoeBlock']
        assert len(blocks)==48
        for layer,block in enumerate(blocks):
            block.grouped_experts=MCoreGroupedExpertsP0(block)
            block.experts=None
            torch.cuda.empty_cache()
            if progress:progress(layer)
        super().__init__(model,scope,tau_reduce)

    def wrap(self,layer,block):
        def forward(hidden_states):
            if self.policy in ('native','top8') or not self.scope[0]<=layer<self.scope[1]:
                return prefix_forward(block,hidden_states,budget=8,grouped=block.grouped_experts)
            if self.policy=='cgcc':
                return prefix_forward(block,hidden_states,tau_reduce=self.tau_reduce,grouped=block.grouped_experts)
            return prefix_forward(block,hidden_states,budget=int(self.policy[3:]),grouped=block.grouped_experts)
        return forward

    def remove(self):
        raise RuntimeError('Converted experts replace HF weights; reload the checkpoint to restore HF execution.')
