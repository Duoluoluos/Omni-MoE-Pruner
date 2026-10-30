#!/bin/bash
# Sample 6 times, 30s apart, into a log file we can read.
HUB=/mnt/bn/isp-traindata-lf-1/wq/OmniZip/eval_data/huggingface/hub/datasets--lmms-lab--Video-MME
LOG=/mnt/bn/isp-traindata-lf-1/wq/OmniZip/logs/dl_progress.log
> $LOG
for i in 1 2 3 4 5 6; do
  ts=$(date +"%H:%M:%S")
  total=$(du -sb $HUB/blobs 2>/dev/null | awk '{print $1}')
  comp=$(ls $HUB/blobs 2>/dev/null | grep -v incomplete | wc -l)
  inc=$(ls $HUB/blobs 2>/dev/null | grep incomplete | wc -l)
  active=$(find $HUB/blobs -name '*.incomplete' -mmin -1 -printf '%s %f\n' 2>/dev/null | sort -nr | head -2 | tr '\n' '|')
  pid=$(pgrep -f "lmms_eval --model" | head -1)
  echo "[$ts] total=${total} done=${comp} inc=${inc} pid=${pid} active=${active}" >> $LOG
  [ $i -lt 6 ] && sleep 30
done
cat $LOG
