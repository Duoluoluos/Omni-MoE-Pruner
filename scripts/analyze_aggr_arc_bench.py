#!/usr/bin/env python3
"""Analyze the aggressive-ARC benchmark log: steady-state s/it + KPI verdict.

Parses both arms (native, aggr_*) from the run log, derives true steady-state
seconds/iteration from elapsed_time deltas (excluding the one-off model-load
inflation baked into the cumulative train_speed), and reports:
  * speedup% of aggr vs native (KPI: >= 10%)
  * eval_loss of each arm (KPI: aggr eval < global-top4's 1.737, i.e. better)

Usage: python3 analyze_aggr_arc_bench.py [logfile]
(defaults to logs/arc_speed_opt/aggr_arc_bench_latest.log)
"""
import re, sys, json

LOG = sys.argv[1] if len(sys.argv) > 1 else "logs/arc_speed_opt/aggr_arc_bench_latest.log"
GLOBAL_TOP4_EVAL = 1.737   # quality red line: aggr must beat this
NATIVE_EVAL_REF = 1.581    # native top-8 reference from prior runs
KPI_SPEEDUP = 10.0         # percent

def parse_elapsed(s):
    # "8m 41s" / "1h 0m 51s" / "0s"
    h = re.search(r"(\d+)h", s); m = re.search(r"(\d+)m", s); sec = re.search(r"(\d+)s", s)
    return (int(h.group(1))*3600 if h else 0) + (int(m.group(1))*60 if m else 0) + (int(sec.group(1)) if sec else 0)

def load_arms(path):
    """Split the log into arms by the '===== Aggressive-ARC benchmark: NAME' headers."""
    text = open(path, "r", errors="ignore").read()
    # arm headers
    headers = [(m.start(), m.group(1)) for m in re.finditer(r"===== Aggressive-ARC benchmark: ([A-Za-z0-9_\-]+)", text)]
    arms = {}
    for i, (pos, name) in enumerate(headers):
        end = headers[i+1][0] if i+1 < len(headers) else len(text)
        seg = text[pos:end]
        # collect iteration -> (elapsed_sec, loss); dict de-dups repeated flushes
        pts = {}
        for mm in re.finditer(r"\{'loss': ([0-9.]+).*?'iteration': '(\d+)/\d+'.*?'elapsed_time': '([^']+)'", seg):
            loss = float(mm.group(1)); it = int(mm.group(2)); el = parse_elapsed(mm.group(3))
            pts[it] = (el, loss)
        ev = re.findall(r"'eval_loss': ([0-9.]+)", seg)
        arms[name] = {"points": pts, "eval": float(ev[-1]) if ev else None}
    return arms

def steady_sit(pts, iterations=None):
    """s/it over the latest useful logged window, excluding model-load time."""
    its = sorted(set(pts) if iterations is None else set(pts).intersection(iterations))
    if len(its) < 2:
        return None
    # With a mature run, use its later half.  With only 2-3 logged points, use
    # the last adjacent interval so the first (usually warmest) interval is not
    # kept in the average longer than necessary.
    lo = its[len(its)//2] if len(its) >= 4 else its[-2]
    hi = its[-1]
    (el_lo, _), (el_hi, _) = pts[lo], pts[hi]
    return (el_hi - el_lo) / (hi - lo), lo, hi

def main():
    arms = load_arms(LOG)
    if not arms:
        print("no arms parsed yet (run may still be loading)"); return
    native = arms.get("native")
    aggr_name = next((n for n in arms if n.startswith("aggr")), None)
    print(f"log: {LOG}")
    print(f"arms found: {list(arms)}\n")
    results = {}
    for name, a in arms.items():
        ss = steady_sit(a["points"])
        if ss:
            sit, lo, hi = ss
            results[name] = sit
            print(f"[{name}] steady s/it = {sit:.2f} (it{lo}->{hi}); eval_loss = {a['eval']}; steps_logged={sorted(a['points'])}")
        else:
            print(f"[{name}] not enough points yet: {sorted(a['points'])}")
    print()
    if native and aggr_name and native.get("points") and arms[aggr_name].get("points"):
        # Partial runs must be compared over the exact same iteration window.
        # Otherwise native may use its mature it15->25 window while a just-started
        # aggressive arm uses it1->5, producing a spurious KPI verdict.
        common_iterations = set(native["points"]).intersection(arms[aggr_name]["points"])
        base_pair = steady_sit(native["points"], common_iterations)
        test_pair = steady_sit(arms[aggr_name]["points"], common_iterations)
        if base_pair and test_pair:
            b, lo_b, hi_b = base_pair
            t, lo_t, hi_t = test_pair
            assert (lo_b, hi_b) == (lo_t, hi_t)
            sp = (b - t) / b * 100
            ev = arms[aggr_name]["eval"]; ev_n = native["eval"]
            complete = ev is not None and ev_n is not None
            print("=== KPI VERDICT ===" if complete else "=== PROVISIONAL PAIRED SPEED ===")
            speed_label = ('PASS' if sp >= KPI_SPEEDUP else 'FAIL') if complete else 'pending full run'
            print(
                f"speedup (it{lo_b}->it{hi_b}): native {b:.2f} -> aggr {t:.2f} s/it "
                f"= {sp:.1f}% faster (KPI >= {KPI_SPEEDUP}%)  {speed_label}"
            )
            if ev is not None:
                q_pass = ev < GLOBAL_TOP4_EVAL
                print(f"quality: aggr eval={ev:.4f} vs global-top4 {GLOBAL_TOP4_EVAL} (must be < )  {'PASS' if q_pass else 'FAIL'}")
                if ev_n is not None:
                    print(f"         native eval={ev_n:.4f}; aggr degradation vs native = +{ev-ev_n:.4f}")
                print(f"\nOVERALL: {'GOAL MET' if (sp>=KPI_SPEEDUP and q_pass) else 'NOT YET MET'}")
            else:
                print("aggr eval_loss not available yet")

if __name__ == "__main__":
    main()
