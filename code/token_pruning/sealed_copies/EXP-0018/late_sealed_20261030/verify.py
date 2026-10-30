"""Independent check (verifier) of the claim that score_vqav2's m4c normaliser deletes no-space commas, unlike the
official VQA evaluation, and of whether that changes any EXP-0018 number. Read-only on project data.

  python -I -X utf8 verify.py
"""
import collections
import importlib.util
import json
import re

M4C = "D:/6CJNM007/idea_rearch/source/LLaVA/llava/eval/m4c_evaluator.py"
ANN = "D:/6CJNM007/data/llava_eval/vqav2_val/llava_vqav2_val_{}_annotations.jsonl"
RUN = "D:/6CJNM007/idea_rearch/code/output/EXP-0018/EXP-0018-R02"
OTHER = {
    "13b vqav2 test-dev": "D:/6CJNM007/data/llava_eval/vqav2/answers/llava_vqav2_mscoco_test-dev2015/llava-v1.5-13b/merge.jsonl",
    "13b textvqa": "D:/6CJNM007/data/llava_eval/textvqa/answers/llava-v1.5-13b.jsonl",
    "13b gqa": "D:/6CJNM007/data/llava_eval/gqa/answers/llava_gqa_testdev_balanced/llava-v1.5-13b.jsonl",
    "13b vizwiz": "D:/6CJNM007/data/llava_eval/vizwiz/answers/llava-v1.5-13b.jsonl",
}

spec = importlib.util.spec_from_file_location("m4c_verify_copy", M4C)
m4c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m4c)
EV = m4c.TextVQAAccuracyEvaluator()
P = EV.answer_processor


def s_m4c(gt, pred):
    """score_pope_mme.py line 85, verbatim"""
    return EV._compute_answer_scores(gt).get(EV.answer_processor(pred), 0.0)


# ---- official vqaEval.py (GT-Vision-Lab/VQA master), written independently ----
PUNCT = [';', r"/", '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_', '-', '>', '<', '@', '`', ',', '?', '!']
COMMA_STRIP = re.compile(r"(\d)(\,)(\d)")
PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
MANUAL = {'none': '0', 'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5', 'six': '6',
          'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10'}
ARTICLES = ['a', 'an', 'the']
CONTR = m4c.EvalAIAnswerProcessor.CONTRACTIONS


def off_punct(t):
    out = t
    for p in PUNCT:
        if (p + ' ' in t or ' ' + p in t) or COMMA_STRIP.search(t) is not None:
            out = out.replace(p, '')
        else:
            out = out.replace(p, ' ')
    return PERIOD_STRIP.sub("", out, re.UNICODE)   # (count=32 quirk kept, as in the official file)


def off_digit_article(t):
    words = []
    for w in t.lower().split():
        w = MANUAL.get(w, w)
        if w not in ARTICLES:
            words.append(w)
    return ' '.join(CONTR.get(w, w) for w in words)


def ws(s):
    return s.replace('\n', ' ').replace('\t', ' ').strip()


def off_norm(s):
    return off_digit_article(off_punct(s))


def loo(gts, r):
    return sum(min(1, float(sum(1 for j, g in enumerate(gts) if j != i and g == r)) / 3) for i in range(len(gts))) / len(gts)


def s_official_raw(gt, pred):
    """vqaEval.py evaluate() on the raw model text"""
    gts, r = [ws(a) for a in gt], ws(pred)
    if len(set(gts)) > 1:
        gts, r = [off_norm(a) for a in gts], off_norm(r)
    return loo(gts, r)


def s_llava_server(gt, pred):
    """LLaVA's VQAv2 pipeline: scripts/convert_vqav2_for_submission.py applies EvalAIAnswerProcessor, the server runs vqaEval"""
    return s_official_raw(gt, P(pred))


def s_lmms(gt, pred):
    """lmms-eval main vqav2_process_results: process_punctuation + process_digit_article (its EvalAIAnswerProcessor) on both
    sides, always"""
    n = lambda s: P.process_digit_article(P.process_punctuation(ws(s)))  # noqa: E731
    return loo([n(a) for a in gt], n(pred))


def s_lmms_offregex(gt, pred):
    """same, with the official commaStrip regex"""
    return loo([off_norm(ws(a)) for a in gt], off_norm(ws(pred)))


DEFS = {"m4c(score_vqav2)": s_m4c, "lmms-eval": s_lmms, "lmms(off regex)": s_lmms_offregex,
        "llava server": s_llava_server, "vqaEval raw text": s_official_raw}

NOSPACE_COMMA = re.compile(r",(?! )")


def cap(s):
    return s[:1].upper() + s[1:]


def comma_variants(a):
    out = {a}
    if ',' in a:
        out |= {re.sub(r",\s*", ",", a), re.sub(r",\s*", ", ", a), re.sub(r"\s*,\s*", " ", a), a.replace(',', '')}
    return out


def analyse_split(split):
    rows = [json.loads(line) for line in open(ANN.format(split), encoding="utf-8")]
    n = len(rows)
    assert all(len(r["answers"]) == 10 for r in rows)
    print(f"\n=== {split}: {n} questions")
    comma_q = [r for r in rows if any(',' in a for a in r["answers"])]
    ns_q = [r for r in rows if any(NOSPACE_COMMA.search(a) for a in r["answers"])]
    print(f"questions with a ',' in some human answer: {len(comma_q)}; with a no-space ',' : {len(ns_q)} "
          f"-> {[(r['question_id'], [a for a in r['answers'] if ',' in a]) for r in ns_q]}")
    # GT partition: do m4c and official normalisation group the ten human answers differently?
    part_diff = []
    for r in rows:
        g = [ws(a) for a in r["answers"]]
        m = [P(a) for a in r["answers"]]
        o = [off_norm(a) for a in g] if len(set(g)) > 1 else g
        pm = sorted(tuple(i for i in range(10) if m[i] == m[k]) for k in range(10))
        po = sorted(tuple(i for i in range(10) if o[i] == o[k]) for k in range(10))
        if pm != po:
            part_diff.append((r["question_id"], r["answers"]))
    print(f"questions whose ten human answers are grouped differently by m4c vs vqaEval normalisation: {len(part_diff)}")
    for q in part_diff[:20]:
        print("   ", q)
    unan_nonfixed = [(r["question_id"], r["answers"][0], P(r["answers"][0])) for r in rows
                     if len(set(ws(a) for a in r["answers"])) == 1 and P(r["answers"][0]) != ws(r["answers"][0])]
    print(f"unanimous questions whose answer is not a fixed point of EvalAIAnswerProcessor: {len(unan_nonfixed)} {unan_nonfixed[:10]}")
    # candidate predictions: the reviewer's set (each distinct human answer, capitalised, + '.', empty) and a wider set
    for label, wide in (("reviewer candidate set", False), ("wider set (+ comma variants)", True)):
        dq = collections.defaultdict(set)
        worst = collections.defaultdict(float)
        for r in rows:
            cands = {""}
            for a in set(r["answers"]):
                for v in (comma_variants(a) if wide else {a}):
                    cands |= {v, cap(v), v + ".", cap(v) + "."}
            for c in cands:
                sm = s_m4c(r["answers"], c)
                for name in ("lmms-eval", "llava server"):
                    d = abs(sm - DEFS[name](r["answers"], c))
                    if d > 1e-12:
                        dq[name].add(r["question_id"])
                        worst[(name, r["question_id"])] = max(worst[(name, r["question_id"])], d)
        for name in ("lmms-eval", "llava server"):
            tot = sum(v for (k, q), v in worst.items() if k == name)
            print(f"{label}: m4c != {name} on {len(dq[name])} questions {sorted(dq[name])}; "
                  f"sum of worst-case |diff| = {tot:.4f} -> {100 * tot / n:.4f} points")
    return rows


def main():
    rows = {s: analyse_split(s) for s in ("test", "dev")}
    by_q = {r["question_id"]: r for r in rows["test"]}
    r = by_q.get(223031001)
    if r:
        print(f"\nqid 223031001 answers {r['answers']}")
        for c in ("Knife, fork", "knife,fork", "knife fork", "Knife and fork"):
            print(f"   pred {c!r}: " + ", ".join(f"{k} {f(r['answers'], c):.4f}" for k, f in DEFS.items()))
    # ---- real R02 answers ----
    preds = [json.loads(line) for line in open(f"{RUN}/answers.jsonl", encoding="utf-8")]
    metrics = json.load(open(f"{RUN}/metrics.json", encoding="utf-8"))
    stored = {json.loads(line)["question_id"]: json.loads(line)["score"] for line in open(f"{RUN}/vqa_scores.jsonl", encoding="utf-8")}
    assert sorted(p["question_id"] for p in preds) == sorted(by_q) and len(preds) == len(by_q)
    print(f"\n=== EXP-0018-R02 real answers: {len(preds)}; metrics.json accuracy {metrics['accuracy']!r} "
          f"{ {t: v['accuracy'] for t, v in metrics['by_answer_type'].items()} }")
    per = {}
    for name, f in DEFS.items():
        tot, typ = [], collections.defaultdict(list)
        sc = {}
        for p in preds:
            g = by_q[p["question_id"]]
            s = f(g["answers"], p["text"])
            sc[p["question_id"]] = s
            tot.append(s)
            typ[g["answer_type"]].append(s)
        per[name] = sc
        print(f"   {name:18s} {100 * sum(tot) / len(tot)!r:22s} " +
              " ".join(f"{t} {100 * sum(v) / len(v):.4f}" for t, v in sorted(typ.items())))
    print(f"   recomputed m4c per-question == vqa_scores.jsonl: {all(abs(per['m4c(score_vqav2)'][q] - s) < 1e-12 for q, s in stored.items())}")
    for name in DEFS:
        d = [(q, by_q[q]['answers'], next(p['text'] for p in preds if p['question_id'] == q), per['m4c(score_vqav2)'][q], per[name][q])
             for q in per[name] if abs(per[name][q] - per['m4c(score_vqav2)'][q]) > 1e-12]
        print(f"   m4c vs {name}: {len(d)} questions differ {d[:6]}")
    texts = [p["text"] for p in preds]
    feats = {"contain ','": lambda t: ',' in t, "no-space ','": lambda t: bool(NOSPACE_COMMA.search(t)),
             "contain '?'": lambda t: '?' in t, "contain \"'s\"": lambda t: "'s" in t,
             "m4c-norm != vqaEval-norm": lambda t: P(t) != off_norm(ws(t)), "capitalised": lambda t: t[:1].isupper(),
             "trailing '.'": lambda t: t.endswith('.'), "multi-line": lambda t: '\n' in t}
    print("   R02 output features: " + "; ".join(f"{k} {sum(1 for t in texts if f(t))}" for k, f in feats.items()))
    print("   R02 outputs where m4c-norm != vqaEval-norm:",
          collections.Counter((t, P(t), off_norm(ws(t))) for t in texts if P(t) != off_norm(ws(t))).most_common(10))
    # ---- other LLaVA-1.5 output files ----
    for name, path in OTHER.items():
        try:
            texts = [json.loads(line)["text"] for line in open(path, encoding="utf-8")]
        except (OSError, KeyError) as e:
            print(f"{name}: unreadable ({e})")
            continue
        ns = [t for t in texts if NOSPACE_COMMA.search(t)]
        cm = [t for t in texts if ',' in t]
        diff = collections.Counter(t for t in texts if P(t) != off_norm(ws(t)))
        print(f"{name}: {len(texts)} outputs; contain ',' {len(cm)}; no-space ',' {len(ns)} {ns[:8]}; "
              f"m4c-norm != vqaEval-norm {sum(diff.values())} {diff.most_common(6)}")


if __name__ == "__main__":
    main()
