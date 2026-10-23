"""Run iTransformer's official run.py with another random seed, without modifying the file.

run.py fixes `fix_seed = 2023` at the top of its __main__ block. This wrapper reads the file, replaces that one
assignment in memory and executes the result as __main__ with the given arguments. With seed 2023 the executed
code is byte-identical to run.py, so a seed-2023 run here can be compared with a direct run to check the wrapper.

  python itransformer_seed.py <iTransformer checkout> <seed> -- <run.py arguments>
"""
import os
import sys

root, seed = os.path.abspath(sys.argv[1]), int(sys.argv[2])
run_args = sys.argv[sys.argv.index("--") + 1:]
path = os.path.join(root, "run.py")
with open(path, encoding="utf-8") as f:
    source = f.read()
if source.count("fix_seed = 2023") != 1:
    raise SystemExit(f"expected exactly one 'fix_seed = 2023' in {path}")
source = source.replace("fix_seed = 2023", f"fix_seed = {seed}")
sys.path.insert(0, root)                     # run.py imports exp/, data_provider/, model/ from its own folder
sys.argv = [path] + run_args
exec(compile(source, path, "exec"), {"__name__": "__main__", "__file__": path})
