"""Backbone adapters for the fresh, isolated TODO 3 comparison.

Original evaluation runners and model files are never modified. All arms use
the same model, prepared tensors, precision and generation API within a run.
"""
from __future__ import annotations
import contextlib
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from todo3_baselines import aset_budget, alloc_token_mask

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def tree_to(value, device):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device)
    if isinstance(value, dict):
        return {k: tree_to(v, device) for k,v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(tree_to(v, device) for v in value)
    return value


def input_digest(inputs):
    h = hashlib.sha256()
    for k,v in sorted(inputs.items()):
        if isinstance(v, torch.Tensor):
            h.update(k.encode())
            h.update(str((tuple(v.shape), v.dtype)).encode())
            h.update(v.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


class Controller:
    def __init__(self, backend):
        self.backend = backend
        self.blocks = backend.blocks
        self.originals = [b.forward for b in self.blocks]
        self.policy = {"kind":"native"}
        self.thresholds = {}
        self.pool = None
        self.accounting = None
        if backend.name == "unimoe":
            import uni_moe.model.modeling_qwen_grin_moe as gm
            self.gm = gm
            self.original_select = gm.dynamic_expert_selection
        for layer,block in enumerate(self.blocks):
            block.forward = self._wrap(layer,block)

    @contextlib.contextmanager
    def arm(self, policy):
        old = self.policy
        self.policy = policy
        try:
            yield
        finally:
            self.policy = old

    def decide(self, probs, layer, native_budget):
        p = self.policy
        kind = p["kind"]
        n = probs.shape[0]
        scope = p.get("scope", [0,len(self.blocks)])
        if not scope[0] <= layer < scope[1] or kind in ("native", "identity"):
            return native_budget
        if kind == "fixed":
            return torch.full((n,),p["budget"],device=probs.device,dtype=torch.long)
        if kind == "schedule":
            return torch.full((n,),p["schedule"][layer],device=probs.device,dtype=torch.long)
        if kind == "alloc":
            ceiling = 8 if self.backend.name == "qwen" else 4
            scores = probs.topk(ceiling,dim=-1).values
            return alloc_token_mask(scores,p["schedule"][layer]).sum(-1)
        if kind == "aset":
            return aset_budget(probs,native_budget,p["theta_min"],p["theta_max"],p["gamma"])
        if kind == "cgcc":
            if self.backend.name == "qwen":
                s = probs.topk(4,dim=-1).values.sum(-1)
                return torch.where(s >= self.thresholds["tau_reduce"],4,8)
            real = probs[:,:4].max(-1).values
            skip = (real >= self.thresholds["tau_skip"]) | (probs[:,4] >= self.thresholds["tau_null"])
            return torch.where(skip,0,torch.where(real <= self.thresholds["tau_hard"],3,2))
        raise ValueError(kind)

    def _wrap(self,layer,block):
        original = self.originals[layer]
        if self.backend.name == "qwen":
            def forward(hidden_states):
                p = self.policy
                scope = p.get("scope",[0,len(self.blocks)])
                if p["kind"] == "native" or not scope[0] <= layer < scope[1]:
                    return original(hidden_states)
                shape = hidden_states.shape
                flat = hidden_states.reshape(-1,shape[-1])
                logits = block.gate(flat)
                probs = F.softmax(logits,dim=-1,dtype=torch.float32)
                weights, experts = probs.topk(8,dim=-1)
                native = torch.full((len(flat),),8,device=flat.device,dtype=torch.long)
                budgets = self.decide(probs,layer,native)
                mask = torch.arange(8,device=flat.device)[None,:] < budgets[:,None]
                weights = weights * mask
                weights = (weights / weights.sum(-1,keepdim=True).clamp_min(1e-30)).to(flat.dtype)
                result = torch.zeros_like(flat)
                # The same compact expert dispatch is used for every modified arm.
                # Native is separately reported to expose wrapper/kernel overhead.
                expert_mask = F.one_hot(experts,num_classes=block.num_experts).permute(2,1,0)
                for e in range(block.num_experts):
                    rank,token = torch.where(expert_mask[e])
                    keep = mask[token,rank]
                    token,rank = token[keep],rank[keep]
                    if token.numel():
                        output = block.experts[e](flat[token]) * weights[token,rank,None]
                        result.index_add_(0,token,output.to(flat.dtype))
                return result.reshape(shape),logits
            return forward
        def forward(hidden_states,attention_mask,aux_balance_weight):
            p = self.policy
            scope = p.get("scope",[0,len(self.blocks)])
            if p["kind"] == "native" or not scope[0] <= layer < scope[1]:
                return original(hidden_states,attention_mask,aux_balance_weight)
            def select(logits,top_p):
                native = self.original_select(logits,top_p)
                return self.decide(torch.softmax(logits.float(),-1),layer,native)
            previous = self.gm.dynamic_expert_selection
            self.gm.dynamic_expert_selection = select
            try:
                return original(hidden_states,attention_mask,aux_balance_weight)
            finally:
                self.gm.dynamic_expert_selection = previous
        return forward

    @contextlib.contextmanager
    def capture_pool(self):
        pool = defaultdict(list)
        handles = []
        for layer,block in enumerate(self.blocks):
            def hook(module,args,output,layer=layer):
                if output.shape[0] < 2:
                    return
                logits = output[:,:5] if self.backend.name == "unimoe" else output
                n = len(logits)
                idx = torch.linspace(0,n-1,min(n,1024),device=logits.device).long()
                sample = logits[idx].detach()
                native = (self.original_select(sample,0.7) if self.backend.name == "unimoe"
                          else torch.full((len(idx),),8,device=sample.device,dtype=torch.long))
                pool[layer].append({"probs":sample.float().softmax(-1).cpu(),
                                    "native_budget":native.cpu(),"weight":n/len(idx)})
            handles.append(block.gate.register_forward_hook(hook))
        try:
            yield pool
        finally:
            for h in handles:
                h.remove()

    @contextlib.contextmanager
    def profile(self):
        """Counts real MLP input rows, padding and selected slots outside timing."""
        ledger = defaultdict(lambda:defaultdict(int))
        handles = []
        phase = ["prefill"]
        for layer,block in enumerate(self.blocks):
            def pre(module,args,layer=layer):
                phase[0] = "prefill" if args[0].shape[-2] > 1 else "decode"
            handles.append(block.register_forward_pre_hook(pre))
            if self.backend.name == "qwen":
                experts = [("routed_rows",e) for e in block.experts]
            else:
                experts = [("routed_rows",e) for e in block.dynamic_real_moe.deepspeed_moe.experts.deepspeed_experts]
                experts += [("shared_rows",e) for e in block.fixed_real_moe]
                def selected(module,args,out,layer=layer):
                    key = f"{phase[0]}:{layer}"
                    ledger[key]["selected_slots"] += int(out[2].sum())
                    ledger[key]["selected_real"] += int(out[3][:,:4].sum())
                    ledger[key]["selected_null"] += int(out[3][:,4].sum())
                    ledger[key]["tokens"] += len(out[2])
                handles.append(block.register_forward_hook(selected))
            for category,expert in experts:
                def count(module,args,category=category,layer=layer):
                    ledger[f"{phase[0]}:{layer}"][category] += args[0].numel() // args[0].shape[-1]
                handles.append(expert.register_forward_pre_hook(count))
        try:
            yield ledger
        finally:
            for h in handles:
                h.remove()


class Backend:
    def __init__(self,name):
        self.name = name
        if name == "qwen":
            import run_mb_new2_cgskip_qwen3omni as old
            from transformers import Qwen3OmniMoeForConditionalGeneration,Qwen3OmniMoeProcessor
            from qwen_omni_utils import process_mm_info
            self.old,self.process_mm = old,process_mm_info
            self.model_path = old.MODEL_PATH
            self.processor = Qwen3OmniMoeProcessor.from_pretrained(str(self.model_path),use_fast=False)
            self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                str(self.model_path),torch_dtype=torch.bfloat16,device_map="cuda",attn_implementation="flash_attention_2").eval()
            self.model.disable_talker()
            self.language = self.model.thinker.model
            self.head = self.model.thinker.lm_head
            self.blocks = [m for m in self.language.modules() if m.__class__.__name__ == "Qwen3OmniMoeThinkerTextSparseMoeBlock"]
            assert len(self.blocks) == 48
            assert all(b.top_k == 8 and b.norm_topk_prob for b in self.blocks)
        else:
            import run_mb_new2_cgskip_unimoe2 as old
            old._ensure_matplotlib_stub()
            from uni_moe.model.processing_qwen2_vl import Qwen2VLProcessor
            from uni_moe.model.modeling_qwen_grin_moe import GrinQwen2VLForConditionalGeneration
            from uni_moe.qwen_vl_utils import process_mm_info
            from uni_moe.model import deepspeed_moe_inference_utils
            self.old,self.process_mm = old,process_mm_info
            self.model_path = ROOT / "models/Uni-MoE-2.0-Omni"
            self.processor = Qwen2VLProcessor.from_pretrained(str(self.model_path))
            self.model = GrinQwen2VLForConditionalGeneration.from_pretrained(
                str(self.model_path),torch_dtype=torch.bfloat16,attn_implementation="flash_attention_2").cuda().eval()
            self.processor.data_args = self.model.config
            self.language = self.model.model
            self.head = self.model.lm_head
            self.blocks = [m for m in self.language.modules() if m.__class__.__name__ == "GRINMoESparseMoeBlock"]
            assert len(self.blocks) == 28
        self.controller = Controller(self)

    @contextlib.contextmanager
    def audit_embeddings(self, inputs):
        """Compare encoder features with actual language inputs outside timing."""
        features, report, handles, originals = {}, {}, [], {}
        if self.name == "qwen":
            thinker = self.model.thinker
            for modality, name in [("audio","get_audio_features"),("video","get_video_features")]:
                originals[name] = getattr(thinker,name)
                def capture(*args,modality=modality,name=name,**kwargs):
                    value = originals[name](*args,**kwargs)
                    features[modality] = value if modality == "audio" else value[0]
                    return value
                setattr(thinker,name,capture)
        else:
            def audio(module,args,value):
                features["audio"] = value.reshape(-1,value.shape[-1])
            def video(module,args,value):
                side = self.model.vision_tower.num_patches_per_side
                n,_,dim = value.shape
                value = value.view(n,side,side,dim).permute(0,3,1,2).contiguous()
                stride = self.model.config.vision_spatial_pool_stride
                value = F.adaptive_avg_pool2d(value,(stride,stride)).permute(0,2,3,1)
                features["video"] = value.reshape(-1,dim)
            handles += [self.model.audio_aligner.register_forward_hook(audio),
                        self.model.vision_aligner.register_forward_hook(video)]
        def check(module,args,kwargs):
            embeds = kwargs["inputs_embeds"]
            if embeds.shape[1] == 1:
                return
            assert not report, "multiple prefill calls"
            ids = inputs["input_ids"]
            for modality in ["audio","video"]:
                token = (151675 if modality == "audio" else 151656) if self.name == "qwen" else getattr(self.model.config,modality+"_token_id")
                actual = embeds[ids == token]
                expected = features[modality].to(embeds.device,embeds.dtype)
                assert len(actual)>0 and actual.shape==expected.shape
                assert torch.isfinite(actual).all() and torch.equal(actual,expected),modality
                report[modality] = len(actual)
        handles.append(self.language.register_forward_pre_hook(check,with_kwargs=True))
        try:
            yield report
            assert report.get("audio",0)>0 and report.get("video",0)>0
        finally:
            for h in handles:
                h.remove()
            if self.name == "qwen":
                for name,value in originals.items():
                    setattr(self.model.thinker,name,value)

    def prepare(self,record):
        assert set(record) <= {"video","question","options","sample_index"}
        path = ROOT / "eval_data/OmniVideoBench" / record["video"]
        conv = self.old.build_conversation(path,record)
        if self.name == "qwen":
            text = self.processor.apply_chat_template(conv,add_generation_prompt=True,tokenize=False)
            audios,images,videos = self.process_mm(conv,use_audio_in_video=True)
            # All-flat kwargs reproduce the existing full-benchmark pipeline;
            # do not mix flat and nested kwargs (the withdrawn density-pilot bug).
            inputs = self.processor(text=text,audio=audios,images=images,videos=videos,
                                    return_tensors="pt",padding=True,use_audio_in_video=True)
            from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import _get_feat_extract_output_lengths
            expected = int(_get_feat_extract_output_lengths(inputs["feature_attention_mask"].sum(-1)).sum())
            assert int((inputs["input_ids"]==151675).sum()) == expected > 0
            inputs = inputs.to(self.model.device).to(self.model.dtype)
        else:
            inputs = self.old.move_inputs(self.old.build_inputs(self.processor,self.process_mm,conv),self.model)
        ids = inputs["input_ids"]
        assert ids.ndim == 2 and ids.shape[0] == 1
        audio_id = 151675 if self.name == "qwen" else self.model.config.audio_token_id
        video_id = 151656 if self.name == "qwen" else self.model.config.video_token_id
        counts = {"audio":int((ids==audio_id).sum()),"video":int((ids==video_id).sum()),"total":ids.numel()}
        assert counts["audio"] > 0 and counts["video"] > 0
        return inputs,counts

    def generate(self,inputs,tokens=4):
        res = self.old.generate_once(self.model,self.processor,inputs,tokens)
        if self.name == "qwen":
            return res
        return {"answer":res.answer,"output_text":res.output_text,"first_logits":res.first_logits,
                "end_to_end_latency_ms":res.e2e_seconds*1000}

    def cache_language(self,inputs):
        captured = {}
        class Captured(Exception):
            pass
        def capture(module,args,kwargs):
            captured.update(tree_to(kwargs,"cpu"))
            raise Captured()
        h = self.language.register_forward_pre_hook(capture,with_kwargs=True)
        try:
            self.generate(inputs,1)
        except Captured:
            pass
        finally:
            h.remove()
        assert "inputs_embeds" in captured
        captured.update(past_key_values=None,use_cache=False,return_dict=True,output_hidden_states=False)
        if self.name == "qwen":
            captured["output_router_logits"] = False
        else:
            captured["output_router_logits_and_topk"] = False
        ids = inputs["input_ids"][0].detach().cpu()
        audio_id = 151675 if self.name == "qwen" else self.model.config.audio_token_id
        video_id = 151656 if self.name == "qwen" else self.model.config.video_token_id
        last_mm = int(torch.where((ids==audio_id)|(ids==video_id))[0].max())
        special = set(self.processor.tokenizer.all_special_ids)
        positions = [i for i in range(last_mm+1,len(ids)) if int(ids[i]) not in special]
        assert len(positions) >= 4
        return {"kwargs":captured,"positions":torch.tensor(positions),"target_ids":ids[positions]}

    def input_nll(self,cache):
        kwargs = tree_to(cache["kwargs"],self.model.device)
        with torch.inference_mode():
            hidden = self.language(**kwargs)[0]
            positions = cache["positions"].to(hidden.device)
            logits = self.head(hidden[0,positions-1]).float()
            targets = cache["target_ids"].to(logits.device)
            nll = F.cross_entropy(logits,targets,reduction="sum")
        return float(nll),len(targets)
