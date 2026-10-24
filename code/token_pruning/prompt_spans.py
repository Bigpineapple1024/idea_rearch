"""Token spans of a POPE-style question ("Is there a(n) X in the image?") in the LLaVA prompt ids after the image.

Used by run_eval.py --layer-rows-log (EXP-0011 D2b); the same rule as analyze_text_rows.py (D2): decode the tokenizer's
pieces, find the question with POPE's pattern (its "an" questions spell "imange"), and take the tokens whose non-space
characters fall inside the question sentence and inside the object name.
"""
import re

QUESTION = re.compile(r"Is there an? (.+?) in the (?:image|imange)\?")


def piece_text(piece):
    m = re.fullmatch(r"<0x([0-9A-F]{2})>", piece)
    return chr(int(m.group(1), 16)) if m else piece.replace("▁", " ")


def question_spans(tokenizer, ids):
    """(question token indices, object-name token indices, object name) within ids; ValueError without a question."""
    pieces = [piece_text(p) for p in tokenizer.convert_ids_to_tokens(ids)]
    starts = [0]
    for p in pieces:
        starts.append(starts[-1] + len(p))
    m = QUESTION.search("".join(pieces))
    if m is None:
        raise ValueError(f"no POPE-style question in {''.join(pieces)!r}")

    def covering(lo, hi):
        out = []
        for i, p in enumerate(pieces):
            a, b = starts[i] + len(p) - len(p.lstrip(" ")), starts[i + 1]
            if a < b and a < hi and b > lo:
                out.append(i)
        return out

    return covering(m.start(), m.end()), covering(m.start(1), m.end(1)), m.group(1)
