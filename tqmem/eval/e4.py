"""E4: temporal-expression translation accuracy (RQ4).
Input jsonl: {"id","lang","question","now","gold_role","gold_mode","gold_center","gold_span":[s,e],"gold_width","expr_type": explicit|implicit|vague}
  python -m tqmem.eval.e4 --set data/e4_en_200.jsonl --llm openai:gpt-4o-mini --translator llm --out results/e4/en_A.jsonl"""
from __future__ import annotations
import argparse, json, os, collections
import numpy as np
from ..llm import make_llm
from ..translate import LLMTranslator, RuleTranslator
from ..store import to_day

WIDTH_ORDER = ["exact", "narrow", "medium", "wide"]

def span_iou(a, b):
    (a0, a1), (b0, b1) = a, b; inter = max(0.0, min(a1, b1) - max(a0, b0) + 1); union = (a1 - a0 + 1) + (b1 - b0 + 1) - inter
    return inter / union if union > 0 else 0.0

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--set", required=True); ap.add_argument("--llm", default="mock"); ap.add_argument("--translator", default="rule")
    ap.add_argument("--out", required=True); a = ap.parse_args()
    llm = make_llm(a.llm); tr = LLMTranslator(llm) if a.translator == "llm" else RuleTranslator()
    items = [json.loads(l) for l in open(a.set, encoding="utf-8") if l.strip()]
    os.makedirs(os.path.dirname(a.out), exist_ok=True); rows = []
    with open(a.out, "w", encoding="utf-8") as f:
        for it in items:
            tq, cost = tr.translate(it["question"], it["now"])
            r = {"id": it["id"], "lang": it.get("lang"), "expr_type": it.get("expr_type"), "parse_failed": tq.parse_failed,
                 "role_ok": tq.role == it["gold_role"], "mode_ok": tq.mode == it["gold_mode"],
                 "width_ok": tq.width == it.get("gold_width"), "width_adj_ok": abs(WIDTH_ORDER.index(tq.width) - WIDTH_ORDER.index(it.get("gold_width", tq.width))) <= 1,
                 "center_mae": abs(to_day(tq.center) - to_day(it["gold_center"])) if (tq.center and it.get("gold_center")) else None,
                 "span_iou": span_iou((to_day(tq.span[0]), to_day(tq.span[1])), (to_day(it["gold_span"][0]), to_day(it["gold_span"][1]))) if (tq.span and it.get("gold_span")) else None,
                 "pred": tq.to_dict(), "cost": cost}
            r["all_ok"] = r["role_ok"] and r["mode_ok"] and r["width_ok"] and ((r["center_mae"] == 0) if it.get("gold_center") else True) and ((r["span_iou"] == 1.0) if it.get("gold_span") else True)
            # error type
            if r["all_ok"]: r["error"] = None
            elif r["parse_failed"]: r["error"] = "schema"
            elif not r["role_ok"]: r["error"] = "role"
            elif not r["mode_ok"]: r["error"] = "mode"
            elif r["center_mae"] not in (None, 0) and r["center_mae"] >= 300: r["error"] = "relative_resolution"
            elif not r["width_ok"]: r["error"] = "width"
            else: r["error"] = "other"
            rows.append(r); f.write(json.dumps(r, ensure_ascii=False) + "\n")
    def agg(sel):
        s = [r for r in rows if sel(r)]
        if not s: return {}
        m = lambda k: float(np.mean([r[k] for r in s]))
        c = [r["center_mae"] for r in s if r["center_mae"] is not None]; io = [r["span_iou"] for r in s if r["span_iou"] is not None]
        return {"n": len(s), "role_acc": m("role_ok"), "mode_acc": m("mode_ok"), "width_acc": m("width_ok"), "width_adj_acc": m("width_adj_ok"),
                "center_mae_days": float(np.mean(c)) if c else None, "span_iou": float(np.mean(io)) if io else None, "all_ok": m("all_ok"),
                "parse_fail_rate": m("parse_failed"), "errors": dict(collections.Counter(r["error"] for r in s if r["error"]))}
    print(json.dumps({"all": agg(lambda r: True), "by_lang": {l: agg(lambda r, l=l: r["lang"] == l) for l in {r["lang"] for r in rows}},
                      "by_expr": {t: agg(lambda r, t=t: r["expr_type"] == t) for t in {r["expr_type"] for r in rows}}}, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    main()
