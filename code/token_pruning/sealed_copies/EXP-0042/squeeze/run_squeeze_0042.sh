#!/usr/bin/env bash
# SQZ-EXP-0042 的计算（CPU，只读已封存的运行）：squeeze_0042.py（写于看到其结果之前，sha256 见 notes.md）。
# 输出写到 code/output/EXP-0042/squeeze（已存在就不运行），日志存 tmp；通过后另行封存。
set -uo pipefail
X=/d/6CJNM007/tmp/exp0042/squeeze
O=/d/6CJNM007/idea_rearch/code/output/EXP-0042
G=/d/6CJNM007/git/cmd/git.exe
H=08b86fdebd7df8f6088121b9a634ef6ed4d4542597de1a070c3b2a60d204c1e6
test ! -e $O/squeeze || { echo "$O/squeeze exists"; exit 1; }
grep -q $H $X/notes.md || { echo "notes.md lacks the hash"; exit 1; }
test "$(sha256sum $X/squeeze_0042.py | cut -d' ' -f1)" = "$H" || { echo "squeeze_0042.py changed"; exit 1; }
test "$($G -C /d/6CJNM007/idea_rearch log -1 --format=%h)" = cea78da || { echo "code branch is not at cea78da"; exit 1; }
test -z "$($G -C /d/6CJNM007/idea_rearch status --porcelain)" || { echo "working tree not clean"; exit 1; }
source /d/6CJNM007/idea_rearch/code/env/windows_env.sh tp >/dev/null 2>&1
export CUDA_VISIBLE_DEVICES=""
cd $X
{ echo "repository HEAD $($G -C /d/6CJNM007/idea_rearch log -1 --format=%h); squeeze_0042.py sha256 $(sha256sum $X/squeeze_0042.py | cut -c1-16)"
  python -B squeeze_0042.py; echo "exit=$?"; } > $X/squeeze_0042.log 2>&1
tail -n 2 $X/squeeze_0042.log
grep -c 'STOP' $X/squeeze_0042.log
