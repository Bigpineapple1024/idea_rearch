#!/usr/bin/env bash
# EXP-0035 计划 v1.1 的二级分析（CPU，只读已封存的 EXP-0035 运行）：先核脚本的哈希与主分析的封存，再运行（不带 --dev）。
set -uo pipefail
X=/d/6CJNM007/tmp/exp0035/secondary
O=/d/6CJNM007/idea_rearch/code/output/EXP-0035
H=508829fc9be7f5f0a69dc130d9925aa7b1377fa083a590d8e434dfbaffe63b00
test -f $O/analysis/SHA256SUMS || { echo "EXP-0035 analysis not sealed"; exit 1; }
test ! -e $O/secondary || { echo "$O/secondary exists"; exit 1; }
grep -q $H $X/notes.md || { echo "notes.md lacks the hash"; exit 1; }
test "$(sha256sum $X/analyze_0035_secondary.py | cut -d' ' -f1)" = "$H" || { echo "analyze_0035_secondary.py changed"; exit 1; }
source /d/6CJNM007/idea_rearch/code/env/windows_env.sh tp >/dev/null 2>&1
export CUDA_VISIBLE_DEVICES=""
cd $X
{ echo "repository HEAD $(/d/6CJNM007/git/cmd/git.exe -C /d/6CJNM007/idea_rearch log -1 --format=%h); analyze_0035_secondary.py sha256 $(sha256sum $X/analyze_0035_secondary.py | cut -c1-16)"
  python -B analyze_0035_secondary.py; echo "exit=$?"; } > $X/secondary_0035.log 2>&1
tail -n 2 $X/secondary_0035.log
grep -q '^exit=0$' $X/secondary_0035.log || exit 1
cp $X/notes.md $O/secondary/notes.md
bash /d/6CJNM007/tmp/seal_folder.sh EXP-0035 secondary $X/analyze_0035_secondary.py $X/run_secondary_0035.sh $X/secondary_0035.log
