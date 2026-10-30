#!/usr/bin/env python3
import json, sys
from pathlib import Path
import torch
ROOT=Path('/mnt/bn/isp-traindata-lf-1/wq/Omni-MoE-Pruner'); HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(ROOT),str(ROOT/'qwen-omni-utils/src'),str(ROOT/'scripts')]
from modeling_qwen3_omni_moe_stack import Qwen3OmniMoeForConditionalGeneration
from transformers import Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info
import run_mb_new2_cgskip_qwen3omni as old
MODEL=ROOT/'models/Qwen3-Omni-30B-A3B-Instruct'; records=json.load(open(ROOT/'experiments/omnidial_layer_followup_20260930/qwen_diagnostic32_v1/inputs.json'))
processor=Qwen3OmniMoeProcessor.from_pretrained(str(MODEL),use_fast=False)
model=Qwen3OmniMoeForConditionalGeneration.from_pretrained(str(MODEL),torch_dtype=torch.bfloat16,device_map='cuda',attn_implementation='flash_attention_2').eval(); model.disable_talker(); model.thinker.enable_omnizip=False
option_ids={s:processor.tokenizer.encode(s,add_special_tokens=False)[0] for s in 'ABCD'}; checks=[]
for idx in (32,41):
 rec=records[idx]; conv=old.build_conversation(ROOT/'eval_data/OmniVideoBench'/rec['video'],rec); text=processor.apply_chat_template(conv,add_generation_prompt=True,tokenize=False); audios,images,videos=process_mm_info(conv,use_audio_in_video=True); inputs=processor(text=text,audio=audios,images=images,videos=videos,return_tensors='pt',padding=True,use_audio_in_video=True).to(model.device).to(model.dtype)
 val=old.generate_once(model,processor,inputs,4); lp=val['first_logits'].reshape(-1).float().log_softmax(-1); now={s:float(lp[i]) for s,i in option_ids.items()}
 hist=json.load(open(ROOT/f'experiments/omnidial_scope_20260929/qwen_full1000_v2/predictions/{idx:04d}.json'))['native']; delta=max(abs(now[s]-hist['option_logp'][s]) for s in now); passed=val['output_text']==hist['output_text'] and delta<=1e-5
 checks.append({'sample_index':idx,'passed':passed,'output_text':val['output_text'],'historical_output_text':hist['output_text'],'max_abs_option_logp_delta':delta})
 del inputs,val; torch.cuda.empty_cache()
out={'passed':all(x['passed'] for x in checks),'checks':checks,'tolerance':1e-5}; (HERE.parent/'identity_gate.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2)); assert out['passed']
