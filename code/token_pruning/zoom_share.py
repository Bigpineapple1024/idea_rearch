"""EXP-0024 运行前、只用题目文字的事实：开发集（及确认用的 TextVQA、V*）上有多少题会被放大（不读答案、不读图像、不调用模型）。

  python -B zoom_share.py [set ...]      # 默认全部：vqav2_dev2 gqa_val400 textvqa_noocr textvqa_ocr vstar

与 run_eval 的口径相同：题目按 model_vqa_loader 的 prompt（vicuna_v1）分词，取图像之后的 id；路由 = routes_to_query；
IDF 在各自的题目文件上计；S = sum(IDF x 可见性) / sum(IDF)，可见性表为 EXP-0015 的 visibility.npy。
放大的题 = 不路由且 S >= 0.2。另按开头词与 VQAv2 的 question_type（只看题目文件里没有的字段时跳过）列出。
"""
import collections
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE if os.path.exists(f"{HERE}/run_eval.py") else "D:/6CJNM007/tmp/exp0024/code")   # tmp copy until committed
sys.path.insert(1, "D:/6CJNM007/idea_rearch/code/token_pruning")
from run_eval import prompt_ids_after_image, question_idf, routes_to_query  # noqa: E402

DATA = "D:/6CJNM007/data/llava_eval"
SETS = {"vqav2_dev2": f"{DATA}/vqav2_val/llava_vqav2_val_dev2.jsonl", "gqa_val400": f"{DATA}/gqa/llava_gqa_val400.jsonl",
        "textvqa_noocr": f"{DATA}/textvqa/llava_textvqa_val_noocr_qid.jsonl", "textvqa_ocr": f"{DATA}/textvqa/llava_textvqa_val_ocr_qid.jsonl",
        "vstar": f"{DATA}/vstar/llava_vstar_test.jsonl"}
VIS = np.load("D:/6CJNM007/idea_rearch/code/output/EXP-0015/visibility/visibility.npy")
S_MIN = 0.2


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("liuhaotian/llava-v1.5-7b", use_fast=False)
    out = {}
    for name, path in SETS.items():
        if sys.argv[1:] and name not in sys.argv[1:]:
            continue
        idf = question_idf(tok, path, "vicuna_v1")
        rows = []
        for line in open(path, encoding="utf-8"):
            q = json.loads(line)
            ids = prompt_ids_after_image(tok, q["text"], "vicuna_v1")
            w = np.array([idf["idf"].get(t, idf["unseen"]) for t in ids])
            s = float((w * VIS[ids]).sum() / w.sum()) if w.sum() > 0 else 0.0
            first = q["text"].strip().split("\n")[0].strip().lower().split()
            rows.append({"routed": routes_to_query(tok.decode(ids)), "S": s, "head": " ".join(first[:2])})
        n = len(rows)
        un = [r for r in rows if not r["routed"]]
        zoom = [r for r in un if r["S"] >= S_MIN]
        heads = collections.Counter(r["head"] for r in un)
        zheads = collections.Counter(r["head"] for r in zoom)
        out[name] = {"n": n, "routed": n - len(un), "unrouted": len(un), "zoomed": len(zoom),
                     "zoomed_share_of_all": round(len(zoom) / n, 4), "zoomed_share_of_unrouted": round(len(zoom) / len(un), 4),
                     "S_unrouted_quantiles": [round(float(x), 3) for x in np.quantile([r["S"] for r in un], [0.1, 0.25, 0.5, 0.75, 0.9])],
                     "top_unrouted_heads": [(h, c, round(zheads.get(h, 0) / c, 3)) for h, c in heads.most_common(12)]}
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
