"""E4 (RQ4): agreement between a gold time-condition annotation (csv) and LLM extractions, with no LLM call.

Sources compared against the gold csv: the LLM draft csv and/or the `tq` field of run records (the extraction the experiments used).
  python -m tqmem.eval.e4_agree --gold annotations/independent_100_claude.csv \
      --draft annotations/independent_100_llm_draft.csv --records <run jsonl> [...] --out reports/e4_agreement.md
Metrics: role / mode / (role, mode) accuracy and Cohen's kappa on mode; on gold time conditions: anchored centre absolute error (days),
range span IoU (days), width accuracy.
"""
from __future__ import annotations
import argparse, csv, json, os
from collections import Counter
from ..store import to_day

def norm(d):
    role = (d.get("role") or "").strip() or "constraint"; mode = (d.get("mode") or "").strip() or "none"
    span = d.get("span") or ([d.get("span_start"), d.get("span_end")] if d.get("span_start") else None)
    return {"role": role, "mode": mode if role != "target" else "none", "center": d.get("center") or None,
            "span": span, "width": (d.get("width") or "").strip() or None}

def kappa(a, b):
    n = len(a); po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b); pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0

def iou(s1, s2):
    a0, a1 = sorted(to_day(x) for x in s1); b0, b1 = sorted(to_day(x) for x in s2)
    inter = max(0.0, min(a1, b1) - max(a0, b0) + 1); union = max(a1, b1) - min(a0, b0) + 1
    return inter / union

def compare(gold, pred, name):
    q = [k for k in gold if k in pred]
    g = [gold[k] for k in q]; p = [pred[k] for k in q]
    acc = lambda f: sum(f(x) == f(y) for x, y in zip(g, p)) / len(q)
    out = {"source": name, "n": len(q), "role": acc(lambda d: d["role"]), "mode": acc(lambda d: d["mode"]),
           "role_mode": acc(lambda d: (d["role"], d["mode"])), "kappa_mode": kappa([d["mode"] for d in g], [d["mode"] for d in p])}
    tc = [(x, y, k) for x, y, k in zip(g, p, q) if x["role"] == "constraint" and x["mode"] != "none"]
    out["n_timed"] = len(tc); out["timed_mode"] = sum(x["mode"] == y["mode"] for x, y, _ in tc) / len(tc) if tc else None
    errs = [abs(to_day(x["center"]) - to_day(y["center"])) for x, y, _ in tc if x["mode"] == "anchored" and y["mode"] == "anchored" and y["center"]]
    out["center_mae"] = sum(errs) / len(errs) if errs else None; out["n_center"] = len(errs)
    ious = [iou(x["span"], y["span"]) for x, y, _ in tc if x["mode"] == "range" and y["mode"] == "range" and y["span"]]
    out["span_iou"] = sum(ious) / len(ious) if ious else None; out["n_span"] = len(ious)
    ws = [(x["width"], y["width"]) for x, y, _ in tc if x["mode"] == y["mode"]]
    out["width"] = sum(a == b for a, b in ws) / len(ws) if ws else None
    out["disagree"] = [(k, x["role"] + "/" + x["mode"], y["role"] + "/" + y["mode"]) for x, y, k in zip(g, p, q) if (x["role"], x["mode"]) != (y["role"], y["mode"])]
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--gold", required=True); ap.add_argument("--draft", default=None)
    ap.add_argument("--records", nargs="*", default=[]); ap.add_argument("--labels", nargs="*", default=None); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    gold = {r["question_id"]: norm(r) for r in csv.DictReader(open(a.gold, encoding="utf-8-sig"))}
    res = []
    if a.draft: res.append(compare(gold, {r["question_id"]: norm(r) for r in csv.DictReader(open(a.draft, encoding="utf-8-sig"))}, "LLM 초안 (csv)"))
    for i, p in enumerate(a.records):
        recs = {}
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            if r.get("question_id") in gold and r.get("tq"): recs[r["question_id"]] = norm(r["tq"])
        res.append(compare(gold, recs, a.labels[i] if a.labels else os.path.basename(p)))
    f = lambda v, d=3: "-" if v is None else f"{v:.{d}f}"
    L = [f"# E4 시간 조건 추출 일치도 (정답: `{os.path.basename(a.gold)}`)", "",
         "| 비교 대상 | n | 역할 | 유형 | 역할+유형 | κ(유형) | 시간 조건 문항 n | 그중 유형 일치 | 중심일 오차(일) | 기간 IoU | 폭 일치 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for o in res:
        L.append(f"| {o['source']} | {o['n']} | {f(o['role'])} | {f(o['mode'])} | {f(o['role_mode'])} | {f(o['kappa_mode'])} | {o['n_timed']} | {f(o['timed_mode'])} | "
                 f"{f(o['center_mae'],1)} (n={o['n_center']}) | {f(o['span_iou'])} (n={o['n_span']}) | {f(o['width'])} |")
    for o in res:
        L += ["", f"## 불일치 — {o['source']}", ""] + [f"- {k}: 정답 {gm} / 추출 {pm}" for k, gm, pm in o["disagree"]] or ["- 없음"]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); open(a.out, "w", encoding="utf-8").write("\n".join(L) + "\n"); print("\n".join(L))

if __name__ == "__main__":
    main()
