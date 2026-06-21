import json

d = json.load(open("/mnt/bn/isp-traindata-lq2/wq/Omni-MoE-Pruner/eval_data/OmniVideoBench/data.json"))


def secs(dur):
    if not dur:
        return None
    try:
        parts = [float(x) for x in str(dur).split(":")]
    except Exception:
        return None
    s = 0
    for p in parts:
        s = s * 60 + p
    return s


buckets = {"<=1min": 0, "1-3min": 0, "3-5min": 0, "5-10min": 0, ">10min": 0, "unknown": 0}
for s in d:
    t = secs(s.get("duration"))
    if t is None:
        buckets["unknown"] += 1
    elif t <= 60:
        buckets["<=1min"] += 1
    elif t <= 180:
        buckets["1-3min"] += 1
    elif t <= 300:
        buckets["3-5min"] += 1
    elif t <= 600:
        buckets["5-10min"] += 1
    else:
        buckets[">10min"] += 1
print("total:", len(d))
for k, v in buckets.items():
    print(f"  {k:10s}: {v} ({v/len(d)*100:.1f}%)")
