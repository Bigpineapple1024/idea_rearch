"""Early look at EXP-0014's MME runs (R04-R07) per subtask, with the EXP-0002/EXP-0010 references."""
import sys

import numpy as np

sys.path.insert(0, "D:/6CJNM007/idea_rearch/code/token_pruning")
from bootstrap_ci import PERCEPTION, load_mme   # noqa: E402

O = "D:/6CJNM007/idea_rearch/code/output"
runs = {"A": "EXP-0014/EXP-0014-R04", "A'": "EXP-0014/EXP-0014-R05", "A'+B": "EXP-0014/EXP-0014-R06", "A+B": "EXP-0014/EXP-0014-R07",
        "vanilla": "EXP-0002/EXP-0002-R02", "visionzip64": "EXP-0002/EXP-0002-R35", "sparsevlm64": "EXP-0002/EXP-0002-R40"}
print("config | total | " + " | ".join(PERCEPTION))
for n, r in runs.items():
    it = load_mme(f"{O}/{r}")
    per = {t: 100 * (it[t].mean() + it[t].all(axis=1).mean()) for t in PERCEPTION}
    print(f"{n} | {sum(per.values()):.1f} | " + " | ".join(f"{per[t]:.0f}" for t in PERCEPTION))
