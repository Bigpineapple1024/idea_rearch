#!/usr/bin/env bash
# SQZ-EXP-0038 的计算（CPU，只读已封存的运行）：squeeze_0038.py（写于看到其结果之前，sha256 见 notes.md）。
# 输出写到 code/output/EXP-0038/squeeze（已存在就不运行），日志存 tmp；通过后另行封存。
set -uo pipefail
X=/d/6CJNM007/tmp/exp0038/squeeze
O=/d/6CJNM007/idea_rearch/code/output/EXP-0038
H=8c275c8e87a9bb0b6efa276004ac68bc6d7bd77665d8670c12c749d9a994a49d
test ! -e $O/squeeze || { echo "$O/squeeze exists"; exit 1; }
grep -q $H $X/notes.md || { echo "notes.md lacks the hash"; exit 1; }
test "$(sha256sum $X/squeeze_0038.py | cut -d' ' -f1)" = "$H" || { echo "squeeze_0038.py changed"; exit 1; }
source /d/6CJNM007/idea_rearch/code/env/windows_env.sh tp >/dev/null 2>&1
export CUDA_VISIBLE_DEVICES=""
cd $X
{ echo "repository HEAD $(/d/6CJNM007/git/cmd/git.exe -C /d/6CJNM007/idea_rearch log -1 --format=%h); squeeze_0038.py sha256 $(sha256sum $X/squeeze_0038.py | cut -c1-16)"
  python -B squeeze_0038.py; echo "exit=$?"; } > $X/squeeze_0038.log 2>&1
tail -n 2 $X/squeeze_0038.log
grep -c 'STOP' $X/squeeze_0038.log
