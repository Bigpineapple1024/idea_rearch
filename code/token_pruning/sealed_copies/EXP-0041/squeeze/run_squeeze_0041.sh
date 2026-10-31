#!/usr/bin/env bash
# SQZ-EXP-0041 的计算（CPU，只读已封存的运行）：squeeze_0041.py（写于看到其结果之前，sha256 见 notes.md）。
# 输出写到 code/output/EXP-0041/squeeze（已存在就不运行），日志存 tmp；通过后另行封存。
set -uo pipefail
X=/d/6CJNM007/tmp/exp0041/squeeze
O=/d/6CJNM007/idea_rearch/code/output/EXP-0041
H=de283a743fd9eeb2819dfe30717a0432f7e4f14d67bf2711721aebc324f50d62
test ! -e $O/squeeze || { echo "$O/squeeze exists"; exit 1; }
grep -q $H $X/notes.md || { echo "notes.md lacks the hash"; exit 1; }
test "$(sha256sum $X/squeeze_0041.py | cut -d' ' -f1)" = "$H" || { echo "squeeze_0041.py changed"; exit 1; }
source /d/6CJNM007/idea_rearch/code/env/windows_env.sh tp >/dev/null 2>&1
export CUDA_VISIBLE_DEVICES=""
cd $X
{ echo "repository HEAD $(/d/6CJNM007/git/cmd/git.exe -C /d/6CJNM007/idea_rearch log -1 --format=%h); squeeze_0041.py sha256 $(sha256sum $X/squeeze_0041.py | cut -c1-16)"
  python -B squeeze_0041.py; echo "exit=$?"; } > $X/squeeze_0041.log 2>&1
tail -n 2 $X/squeeze_0041.log
grep -c 'STOP' $X/squeeze_0041.log
