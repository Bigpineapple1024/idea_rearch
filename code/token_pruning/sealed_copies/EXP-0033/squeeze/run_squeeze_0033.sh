#!/usr/bin/env bash
# SQZ-EXP-0033 的计算（CPU，只读已封存的运行）：squeeze_0033.py、squeeze_0033_types.py（两个都写于看到结果之前，sha256 见 notes.md）。
# 输出写到 code/output/EXP-0033/squeeze 与 squeeze_types（已存在就不运行），日志存 tmp；通过后另行封存。
set -uo pipefail
X=/d/6CJNM007/tmp/exp0033/squeeze
O=/d/6CJNM007/idea_rearch/code/output/EXP-0033
for d in squeeze squeeze_types; do test ! -e $O/$d || { echo "$O/$d exists"; exit 1; }; done
grep -q ddca6819b435b38c48abb61349834d738e442fdfe0e9ac761319492b828f4097 $X/notes.md || { echo "notes.md lacks the hash"; exit 1; }
test "$(sha256sum $X/squeeze_0033.py | cut -d' ' -f1)" = "ddca6819b435b38c48abb61349834d738e442fdfe0e9ac761319492b828f4097" || { echo "squeeze_0033.py changed"; exit 1; }
test "$(sha256sum $X/squeeze_0033_types.py | cut -d' ' -f1)" = "3798d33f8beec2c702dc83deb7507b7963e8a392d75f3466f3d575157b4d753c" || { echo "squeeze_0033_types.py changed"; exit 1; }
source /d/6CJNM007/idea_rearch/code/env/windows_env.sh tp >/dev/null 2>&1
export CUDA_VISIBLE_DEVICES=""
cd $X
for s in squeeze_0033 squeeze_0033_types; do
  { echo "repository HEAD $(/d/6CJNM007/git/cmd/git.exe -C /d/6CJNM007/idea_rearch log -1 --format=%h); $s.py sha256 $(sha256sum $X/$s.py | cut -c1-16)"
    python -B $s.py; echo "exit=$?"; } > $X/$s.log 2>&1
  tail -n 2 $X/$s.log
done
