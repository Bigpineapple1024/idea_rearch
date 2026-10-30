#!/usr/bin/env bash
# SQZ-EXP-0032 的计算（CPU，只读已封存的运行）；输出写到 code/output/EXP-0032/squeeze（已存在就不运行），日志存 tmp；通过后另行封存。
set -uo pipefail
X=/d/6CJNM007/tmp/exp0032/squeeze
L=$X/squeeze_0032.log
test ! -e /d/6CJNM007/idea_rearch/code/output/EXP-0032/squeeze || { echo "squeeze folder exists"; exit 1; }
{
source /d/6CJNM007/idea_rearch/code/env/windows_env.sh tp
export CUDA_VISIBLE_DEVICES=""
echo "repository HEAD $(/d/6CJNM007/git/cmd/git.exe -C /d/6CJNM007/idea_rearch log -1 --format=%h); squeeze_0032.py sha256 $(sha256sum $X/squeeze_0032.py | cut -c1-16)"
cd $X
python -B squeeze_0032.py
echo "exit=$?"
} > $L 2>&1
tr -d '\r' < $L | grep -v -E 'bitsandbytes|cextension|CUDA Setup|Traceback|File "|lib = |cuda_specs|IndexError|list index|get_native|python -m bitsandbytes|^\s*$' | tail -n 80
