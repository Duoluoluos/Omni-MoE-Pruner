import json, os, statistics

RUN_DIR = open("/mnt/bn/isp-traindata-lq2/wq/Omni-MoE-Pruner/logs/model_download/.last_qf_full_dir").read().strip()
rows = [json.loads(l) for l in open(os.path.join(RUN_DIR, "predictions.jsonl")) if l.strip()]
ok = [r for r in rows if not r.get("error")]
oom = [r for r in rows if r.get("error")]
print(f"total={len(rows)} OK={len(ok)} OOM={len(oom)}")
print("OK indices (first 25):", sorted(r["sample_index"] for r in ok)[:25])
print("OOM indices (first 25):", sorted(r["sample_index"] for r in oom)[:25])


def durs(rs):
    out = []
    for r in rs:
        d = r.get("duration")
        try:
            out.append(float(d))
        except Exception:
            pass
    return out


okd, oomd = durs(ok), durs(oom)
if okd:
    print(f"OK duration: n={len(okd)} mean={statistics.mean(okd):.0f}s median={statistics.median(okd):.0f}s max={max(okd):.0f}s")
if oomd:
    print(f"OOM duration: n={len(oomd)} mean={statistics.mean(oomd):.0f}s median={statistics.median(oomd):.0f}s max={max(oomd):.0f}s")
