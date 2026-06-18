#!/usr/bin/env bash
# Inspect/kill lingering Quality Floor processes on the worker.
for d in /proc/[0-9]*; do
  if grep -qa "run_omnivideobench_budget\|_remote_run_quality_floor" "$d/cmdline" 2>/dev/null; then
    pid=$(basename "$d")
    ppid=$(awk '/^PPid:/{print $2}' "$d/status" 2>/dev/null)
    pcmd=$(tr '\0' ' ' < "/proc/$ppid/cmdline" 2>/dev/null | cut -c1-90)
    cmd=$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null | cut -c1-60)
    echo "P: pid=$pid ppid=$ppid | self=[$cmd] | parent=[$pcmd]"
  fi
done
echo "GPU=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null)"
