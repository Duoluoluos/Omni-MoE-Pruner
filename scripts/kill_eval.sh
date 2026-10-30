#!/bin/bash
pkill -9 -f lmms_eval 2>/dev/null
pkill -9 -f "accelerate launch" 2>/dev/null
sleep 1
pkill -9 -f "bash eval.sh" 2>/dev/null
sleep 2
echo "=== remaining processes ==="
ps -ef | grep -E "lmms_eval|accelerate|eval.sh" | grep -v grep | head
echo "=== done ==="
