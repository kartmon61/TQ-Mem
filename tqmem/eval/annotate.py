"""LLM draft annotation of role / time type for temporal questions, with regex cross-check.
  python -m tqmem.eval.annotate --split all --llm openai:gpt-4o-mini --out annotations/locomo_temporal_roles.csv"""
from __future__ import annotations
import argparse, csv, os
from ..llm import make_llm
from ..translate import LLMTranslator, RuleTranslator
from ..bench import locomo

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/locomo10.json"); ap.add_argument("--split", default="all")
    ap.add_argument("--llm", default="mock"); ap.add_argument("--out", default="annotations/locomo_temporal_roles.csv")
    ap.add_argument("--review_out", default="annotations/review_queue.csv")
    ap.add_argument("--workers", type=int, default=8, help="parallel extraction calls (rows keep dataset order)"); a = ap.parse_args()
    tr = LLMTranslator(make_llm(a.llm)) if a.llm != "mock" else RuleTranslator()
    rows = []; n_fail = 0; n_agree = 0
    from concurrent.futures import ThreadPoolExecutor
    pairs = [(c, q) for c in locomo.load(a.data, None if a.split == "all" else a.split) for q in c.qas if q.category == "temporal"]
    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        tqs = list(ex.map(lambda cq: tr.translate(cq[1].question, cq[0].now)[0], pairs))
    for (c, q), tq in zip(pairs, tqs):
        if True:
            needs = int(tq.role != q.role_guess or tq.parse_failed or (tq.mode == "anchored" and not tq.center) or (tq.mode == "range" and not tq.span))
            n_fail += int(tq.parse_failed); n_agree += int(tq.role == q.role_guess)
            rows.append({"question_id": q.qid, "question": q.question, "now": c.now, "role": tq.role, "mode": tq.mode, "center": tq.center or "",
                         "span_start": tq.span[0] if tq.span else "", "span_end": tq.span[1] if tq.span else "", "width": tq.width,
                         "regex_role": q.role_guess, "source": "llm", "needs_review": needs, "note": ""})
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(a.review_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows([r for r in rows if r["needs_review"]])
    from collections import Counter
    print(f"{len(rows)} temporal questions | role agreement with regex {n_agree/len(rows):.3f} | parse_failed {n_fail} | needs_review {sum(r['needs_review'] for r in rows)}")
    print("mode distribution:", dict(Counter(r["mode"] for r in rows)))

if __name__ == "__main__":
    main()
