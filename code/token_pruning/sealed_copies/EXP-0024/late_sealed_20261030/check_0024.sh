#!/bin/bash
# 只读检查：EXP-0024 的进度（不读任何结果文件的内容，只看状态与进度条）。
O=/d/6CJNM007/idea_rearch/code/output/EXP-0024
echo "== time: $(date)"
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
echo "== chain log (run lines hidden: they carry the headline)"; cat /d/6CJNM007/tmp/run_vision_0024.log | tr '\r' '\n' | grep -v '^$' | grep -v -E '^[AM] |^EXP-0024-R' | tail -n 6
echo "== runs (status only)"; test -f $O/runs.tsv && cut -f1,2,3,4 $O/runs.tsv
echo "== prior_k1"; ls $O/prior_k1 2>/dev/null | tr '\n' ' '; echo
latest=$(ls -d $O/EXP-0024-R* 2>/dev/null | sort | tail -n 1)
echo "== latest folder: $(basename "$latest")"
tail -c 400 "$latest/stdout.log" 2>/dev/null | tr '\r' '\n' | grep -v '^$' | grep -E 'it/s|s/it|Loading' | tail -n 1
