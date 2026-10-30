#!/bin/bash
pid=$(pgrep -f "lmms_eval --model" | head -1)
echo "pid=$pid"
[ -z "$pid" ] && exit 0
echo "=== top threads (cpu time) ==="
ps -eL -o pid,tid,pcpu,time,cmd 2>/dev/null | awk -v p=$pid '$1==p' | sort -k4 -r | head -10
echo "=== open .incomplete files (heuristic: pid points at) ==="
ls -la /proc/$pid/fd 2>/dev/null | grep incomplete
echo "=== py-spy if available ==="
which py-spy
