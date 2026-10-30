#!/bin/bash
# Find the lmms_eval child (the actual download worker)
parent=$(pgrep -f "accelerate launch.*lmms_eval" | head -1)
echo "parent_accelerate=$parent"
child=$(pgrep -P $parent 2>/dev/null | head -1)
echo "child_lmms=$child"
[ -z "$child" ] && exit 0
echo "=== child status ==="
cat /proc/$child/status 2>/dev/null | grep -E "State|Threads|VmRSS"
echo "=== child open .incomplete files ==="
ls -la /proc/$child/fd 2>/dev/null | grep -i incomplete
echo "=== child open sockets ==="
ls -la /proc/$child/fd 2>/dev/null | grep -c sock
echo "=== top threads of child by recent CPU ==="
ps -L -o tid,pcpu,time,wchan,stat,cmd -p $child --no-headers 2>/dev/null | sort -k2 -nr | head -8
echo "=== children of child ==="
pgrep -P $child 2>/dev/null
echo "=== /proc/$child/wchan main ==="
cat /proc/$child/wchan 2>/dev/null; echo
