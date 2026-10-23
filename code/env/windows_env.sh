# Lab computer setup for Git Bash on native Windows (DEC-0013, DEC-0014).
# Source it at the start of every command; Claude Code does not keep shell state between commands.
#   source /d/6CJNM007/idea_rearch/code/env/windows_env.sh tp && python ...
# Keeps caches and temporary files inside D:\6CJNM007 and activates the named conda environment.
# Paths handed to Python use the D:/... form: Windows Python does not understand /d/... paths.

LAB_ROOT="D:/6CJNM007"
export HF_HOME="$LAB_ROOT/cache/huggingface"
export TORCH_HOME="$LAB_ROOT/cache/torch"
export PIP_CACHE_DIR="$LAB_ROOT/cache/pip"
export CONDA_PKGS_DIRS="$LAB_ROOT/cache/conda_pkgs"
export MPLCONFIGDIR="$LAB_ROOT/cache/matplotlib"
export TMP="$LAB_ROOT/tmp" TEMP="$LAB_ROOT/tmp" TMPDIR="$LAB_ROOT/tmp"   # Python checks TMPDIR first
export PYTHONUTF8=1                       # open() would otherwise default to the locale code page (cp932/GBK)
export HF_HUB_DISABLE_SYMLINKS_WARNING=1  # without developer mode the Hub cache copies files instead of symlinking
mkdir -p "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$MPLCONFIGDIR" "$TMP" "$LAB_ROOT/backup"

eval "$("/d/6CJNM007/miniforge3/Scripts/conda.exe" shell.bash hook)"
if [ -n "${1:-}" ]; then
  conda activate "$1"
fi
