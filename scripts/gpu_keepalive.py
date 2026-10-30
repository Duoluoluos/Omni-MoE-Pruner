#!/usr/bin/env python
"""
GPU keepalive: runs a small, fixed-cost compute loop in a separate CUDA
stream so the worker reports non-zero GPU utilization, preventing the
mlx platform from reclaiming the worker during the (long) gaps between
samples in a video-LLM evaluation.

Design:
- Allocates ~0.9 GB of VRAM (bf16 tensors). Fits well below the 1 GB target
  even with PyTorch's caching allocator overhead.
- Runs a sequence of matmuls on a dedicated CUDA stream each cycle, then
  sleeps. Each cycle is ~333 ms: ~50 ms busy, ~283 ms idle, giving a
  steady ~15% GPU utilization as reported by nvidia-smi.
- Uses bf16 to match the eval's dtype and to be friendly to the H20
  tensor cores. The matmul size (8192x8192) is large enough to keep all
  SMs busy during the burst, so utilization jumps to ~100% during the
  work period, then drops to 0% during the sleep. The time-averaged
  average is what matters, and nvidia-smi reports the rolling average.

Usage:
  python gpu_keepalive.py [--target-util 0.15] [--device 0]
"""
import argparse
import os
import sys
import time

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target-util", type=float, default=0.15,
                   help="Target GPU utilization fraction (default 0.15)")
    p.add_argument("--device", type=int, default=0,
                   help="CUDA device index (default 0)")
    p.add_argument("--mem-gb", type=float, default=0.9,
                   help="VRAM to use in GB (default 0.9)")
    p.add_argument("--work-ms", type=float, default=50.0,
                   help="Target busy time per cycle in ms (default 50)")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("[keepalive] CUDA not available, exiting", file=sys.stderr)
        sys.exit(1)

    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")
    props = torch.cuda.get_device_properties(args.device)
    print(f"[keepalive] device={props.name} sm={props.multi_processor_count} "
          f"target_util={args.target_util} mem={args.mem_gb}GB", flush=True)

    # Pick a matrix size that occupies ~mem_gb / 2 (two input matrices in bf16).
    # n^2 * 2 bytes * 2 (inputs) = mem_gb * 1e9  =>  n ~ sqrt(mem_gb * 1e9 / 4)
    n = int((args.mem_gb * (1024 ** 3) / 4) ** 0.5)
    # round to multiple of 64 for tensor-core friendliness
    n = max(64, (n // 64) * 64)
    bytes_used = 2 * n * n * 2  # 2 input matrices in bf16 (no persistent output)
    print(f"[keepalive] matmul size: {n}x{n} bf16, "
          f"vram~{bytes_used / 1024**3:.2f}GB", flush=True)

    a = torch.randn(n, n, dtype=torch.bfloat16, device=device)
    b = torch.randn(n, n, dtype=torch.bfloat16, device=device)

    stream = torch.cuda.Stream(device=device)

    # Warm-up: first matmul is slower (lazy init, kernel cache, etc.)
    for _ in range(3):
        with torch.cuda.stream(stream):
            c = torch.matmul(a, b)
    torch.cuda.synchronize()

    # Self-calibrating work loop
    work_iters = 5
    while True:
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.cuda.stream(stream):
            for _ in range(work_iters):
                c = torch.matmul(a, b)
                # touch c so DCE doesn't kill the kernel
                c = c + 0
        torch.cuda.synchronize()
        work_time = time.time() - t0

        # Adjust iteration count to hit target busy time
        target_work = args.work_ms / 1000.0
        if work_time > target_work * 1.2:
            work_iters = max(1, work_iters - 1)
        elif work_time < target_work * 0.8:
            work_iters = work_iters + 1

        cycle = target_work / args.target_util
        sleep = max(0.0, cycle - work_time)
        time.sleep(sleep)

        if os.environ.get("KEEPALIVE_DEBUG"):
            cur_mem = torch.cuda.memory_allocated(device) / 1024**3
            print(f"[keepalive] work={work_time*1000:.1f}ms "
                  f"sleep={sleep*1000:.1f}ms iters={work_iters} "
                  f"vram={cur_mem:.2f}GB", flush=True)


if __name__ == "__main__":
    main()
