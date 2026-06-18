#!/usr/bin/env python3
"""Download missing Qwen3-Omni shards via the ModelScope SDK.

The SDK uses ModelScope's official CDN with chunked/concurrent transfer, which
often beats a single raw curl stream. Downloads straight into the existing model
dir; already-complete shards are skipped. Resumable + outer retry loop.
"""
import os
import sys
import time

os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("ALL_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("all_proxy", None)
# Default cache path is read-only on this devbox; point it at a writable dir.
os.environ.setdefault("MODELSCOPE_CACHE", "/tmp/ms_cache")
os.makedirs(os.environ["MODELSCOPE_CACHE"], exist_ok=True)

MODEL_DIR = "/mnt/bn/isp-traindata-lq2/wq/Omni-MoE-Pruner/models/Qwen3-Omni-30B-A3B-Instruct"
REPO = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

WANT = {
    "model-00004-of-00015.safetensors": 4997755648,
    "model-00012-of-00015.safetensors": 4997755792,
    "model-00013-of-00015.safetensors": 4999771808,
    "model-00014-of-00015.safetensors": 4996618552,
    "model-00015-of-00015.safetensors": 553698794,
    "configuration.json": 43,
}


def missing():
    out = []
    for n, w in WANT.items():
        p = os.path.join(MODEL_DIR, n)
        cur = os.path.getsize(p) if os.path.exists(p) else 0
        if cur != w:
            out.append(n)
    return out


def main():
    from modelscope.hub.snapshot_download import snapshot_download

    for attempt in range(1, 201):
        todo = missing()
        if not todo:
            print(f"[{time.strftime('%F %T')}] ALL COMPLETE", flush=True)
            return 0
        print(f"[{time.strftime('%F %T')}] attempt {attempt}, missing={todo}", flush=True)
        try:
            snapshot_download(
                REPO,
                local_dir=MODEL_DIR,
                allow_patterns=todo,
                max_workers=4,
            )
        except Exception as exc:
            print(f"[{time.strftime('%F %T')}] error: {exc!r}", flush=True)
        time.sleep(3)

    todo = missing()
    print(f"[{time.strftime('%F %T')}] STOP after retries, still missing={todo}", flush=True)
    return 1 if todo else 0


if __name__ == "__main__":
    sys.exit(main())
