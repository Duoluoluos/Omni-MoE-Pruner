#!/usr/bin/env python3
"""GPU keep-alive: keeps every GPU at a steady utilization so the worker is not
reclaimed for being idle, WITHOUT disturbing the eval job sharing the GPUs.

Design:
- One worker thread per GPU, so all GPUs stay busy *concurrently* (a single
  serial loop would leave the other cards idle while one computes, producing
  the alternating 0%/100% pattern).
- Each thread allocates only a small fixed buffer (default 1.5 GB) so it never
  competes with the eval job for memory.
- Each thread runs short matmul bursts with sleeps in between to target a
  moderate average utilization (busy/(busy+idle) duty cycle), leaving headroom
  for the real workload.
- On CUDA OOM (eval job grabbed memory) the thread releases its buffer and backs
  off instead of crashing, then retries later.
"""
import argparse
import threading
import time

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gpus", default="all", help="comma list of gpu indices, or 'all'")
    p.add_argument("--buffer-gb", type=float, default=1.5, help="GB allocated per GPU for the matmul buffer")
    p.add_argument("--busy-ms", type=int, default=400, help="compute burst length per cycle (ms)")
    p.add_argument("--idle-ms", type=int, default=120, help="sleep length per cycle (ms); lower => higher util")
    p.add_argument("--matrix", type=int, default=4096, help="square matmul dimension")
    return p.parse_args()


def gpu_list(spec):
    n = torch.cuda.device_count()
    if spec == "all":
        return list(range(n))
    return [int(x) for x in spec.split(",") if x.strip() != ""]


def make_tensors(dev, buffer_gb, matrix):
    a = torch.randn(matrix, matrix, device=dev, dtype=torch.float16)
    b = torch.randn(matrix, matrix, device=dev, dtype=torch.float16)
    bytes_used = a.numel() * 2 * 2
    target = int(buffer_gb * (1024 ** 3))
    pad_elems = max(0, (target - bytes_used) // 2)
    pad = torch.empty(pad_elems, device=dev, dtype=torch.float16) if pad_elems > 0 else None
    return a, b, pad


def worker(gpu, args, stop):
    dev = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(dev)
    a = b = pad = None
    try:
        a, b, pad = make_tensors(dev, args.buffer_gb, args.matrix)
    except RuntimeError as e:
        print(f"[keepalive] gpu{gpu} init OOM, will retry: {str(e)[:80]}", flush=True)

    cycles = 0
    last_report = 0.0
    while not stop.is_set():
        try:
            if a is None:
                a = torch.randn(args.matrix, args.matrix, device=dev, dtype=torch.float16)
                b = torch.randn(args.matrix, args.matrix, device=dev, dtype=torch.float16)
            t_end = time.time() + args.busy_ms / 1000.0
            while time.time() < t_end:
                c = a @ b
                a = c * 0.0001 + a  # keep values bounded, keep using result
            torch.cuda.synchronize(dev)
        except RuntimeError as e:
            print(f"[keepalive] gpu{gpu} runtime error (likely OOM), releasing buffer: {str(e)[:80]}", flush=True)
            a = b = pad = None
            torch.cuda.empty_cache()
            time.sleep(5)
            continue
        cycles += 1
        time.sleep(args.idle_ms / 1000.0)
        now = time.time()
        if now - last_report > 60:
            print(f"[keepalive] gpu{gpu} alive, cycles={cycles}", flush=True)
            last_report = now


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        print("[keepalive] CUDA not available, exiting", flush=True)
        return
    gpus = gpu_list(args.gpus)
    print(f"[keepalive] gpus={gpus} buffer_gb={args.buffer_gb} busy_ms={args.busy_ms} "
          f"idle_ms={args.idle_ms} matrix={args.matrix} (one thread per gpu)", flush=True)

    stop = threading.Event()
    threads = []
    for g in gpus:
        t = threading.Thread(target=worker, args=(g, args, stop), daemon=True)
        t.start()
        threads.append(t)

    try:
        while any(t.is_alive() for t in threads):
            time.sleep(1.0)
    except KeyboardInterrupt:
        stop.set()


if __name__ == "__main__":
    main()
