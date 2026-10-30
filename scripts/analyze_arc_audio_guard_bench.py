#!/usr/bin/env python3
"""Analyze paired native vs ARC audio-guard training speed and eval quality."""

from __future__ import annotations

import re
import sys


LOG = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "logs/arc_speed_opt/arc_audio_guard_bench_latest.log"
)
KPI_SPEEDUP = 8.0
PRIOR_AGGRESSIVE_EVAL = 1.62124598


def parse_elapsed(value: str) -> int:
    hours = re.search(r"(\d+)h", value)
    minutes = re.search(r"(\d+)m", value)
    seconds = re.search(r"(\d+)s", value)
    return (
        (int(hours.group(1)) * 3600 if hours else 0)
        + (int(minutes.group(1)) * 60 if minutes else 0)
        + (int(seconds.group(1)) if seconds else 0)
    )


def load_arms(path: str) -> dict[str, dict]:
    text = open(path, "r", errors="ignore").read()
    headers = [
        (match.start(), match.group(1))
        for match in re.finditer(
            r"===== ARC-Audio-Guard benchmark: ([A-Za-z0-9_\-]+)", text
        )
    ]
    arms = {}
    for index, (position, name) in enumerate(headers):
        end = headers[index + 1][0] if index + 1 < len(headers) else len(text)
        segment = text[position:end]
        points = {}
        for match in re.finditer(
            r"\{'loss': ([0-9.]+).*?'iteration': '(\d+)/\d+'.*?'elapsed_time': '([^']+)'",
            segment,
        ):
            points[int(match.group(2))] = (
                parse_elapsed(match.group(3)),
                float(match.group(1)),
            )
        eval_losses = re.findall(r"'eval_loss': ([0-9.]+)", segment)
        arms[name] = {
            "points": points,
            "eval": float(eval_losses[-1]) if eval_losses else None,
            "plugin_installed": "[ARC-AUDIO-GUARD] installed calibrated static policy" in segment,
            "top4_applied": "reduced static top-4 applied" in segment,
            "top6_applied": "audio-confidence protected static top-6 applied" in segment,
        }
    return arms


def steady_sit(points: dict[int, tuple[int, float]], iterations=None):
    available = sorted(set(points) if iterations is None else set(points).intersection(iterations))
    if len(available) < 2:
        return None
    lower = available[len(available) // 2] if len(available) >= 4 else available[-2]
    upper = available[-1]
    return (
        (points[upper][0] - points[lower][0]) / (upper - lower),
        lower,
        upper,
    )


def main() -> int:
    arms = load_arms(LOG)
    if not arms:
        print("no arms parsed yet (run may still be loading)")
        return 1
    guard_name = next((name for name in arms if name.startswith("arc_audio_guard")), None)
    print(f"log: {LOG}")
    print(f"arms found: {list(arms)}\n")
    for name, arm in arms.items():
        steady = steady_sit(arm["points"])
        if steady:
            seconds, lower, upper = steady
            print(
                f"[{name}] steady s/it={seconds:.2f} (it{lower}->it{upper}); "
                f"eval_loss={arm['eval']}; steps={sorted(arm['points'])}"
            )
        else:
            print(f"[{name}] not enough points yet: {sorted(arm['points'])}")
    native = arms.get("native")
    if not native or not guard_name:
        return 1
    guard = arms[guard_name]
    common = set(native["points"]).intersection(guard["points"])
    native_pair = steady_sit(native["points"], common)
    guard_pair = steady_sit(guard["points"], common)
    if not native_pair or not guard_pair:
        print("\npaired timing window not available yet")
        return 1
    native_sit, lower, upper = native_pair
    guard_sit, guard_lower, guard_upper = guard_pair
    assert (lower, upper) == (guard_lower, guard_upper)
    speedup = (native_sit - guard_sit) / native_sit * 100
    complete = native["eval"] is not None and guard["eval"] is not None
    runtime_proof = guard["plugin_installed"] and guard["top4_applied"] and guard["top6_applied"]
    print("\n=== FINAL KPI VERDICT ===" if complete else "\n=== PROVISIONAL PAIRED SPEED ===")
    print(
        f"speed (it{lower}->it{upper}): native {native_sit:.2f} -> guard {guard_sit:.2f} s/it "
        f"= {speedup:.2f}% faster (KPI >= {KPI_SPEEDUP:.1f}%) "
        f"{'PASS' if complete and speedup >= KPI_SPEEDUP else 'pending' if not complete else 'FAIL'}"
    )
    print(
        "runtime schedule proof: "
        f"plugin={guard['plugin_installed']} top4={guard['top4_applied']} "
        f"audio-protected-top6={guard['top6_applied']} "
        f"{'PASS' if runtime_proof else 'FAIL'}"
    )
    if not complete:
        return 1
    eval_delta = guard["eval"] - native["eval"]
    quality_pass = guard["eval"] <= PRIOR_AGGRESSIVE_EVAL
    print(
        f"quality: native={native['eval']:.8f}, guard={guard['eval']:.8f}, "
        f"delta={eval_delta:+.8f}"
    )
    print(
        f"quality protection vs prior unguarded aggressive eval {PRIOR_AGGRESSIVE_EVAL:.8f}: "
        f"{'PASS' if quality_pass else 'FAIL'}"
    )
    overall = speedup >= KPI_SPEEDUP and quality_pass and runtime_proof
    print(f"\nOVERALL: {'GOAL MET' if overall else 'NOT YET MET'}")
    return 0 if overall else 2


if __name__ == "__main__":
    raise SystemExit(main())
