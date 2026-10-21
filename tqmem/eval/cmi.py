"""E3 V8 (4th round, 2026-09-29): AssoMem-style learned weights (AssoMem 3.2.3) in place of TQ-Mem's static weight table.

Per time type (our extracted mode), the semantic score and the kernel score of every candidate are binned low/medium/high
(global tertiles over the training candidates), the conditional mutual information with the binary label "candidate is a gold
evidence turn" is measured, and the weights are softmax(CMI / T) over {semantic, temporal}. Everything else is TQ-Mem v3
(min-max semantic score, kernel width x2, no floor, no window, full-question search, the same extraction).

The labels are what AssoMem needs and TQ-Mem does not. Protocol mirrors tune_cv: leave one LoCoMo test conversation out, learn on
the other seven (T picked on the training fold's recall@10 from GRID_T), apply to the held-out one; `pooled` (all eight) is used on LME.

  python -m tqmem.eval.cmi --embedder BAAI/bge-large-en-v1.5 \
      --extractions "results/e1/rag+tq-nt.fq.nowin-cv_bedrock_us.openai.gpt-5.6-luna_us-east-1_test_run1.jsonl" --out configs/cmi_weights.json
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
from ..embed import make_embedder
from ..store import MemoryStore, to_day
from .. import kernels as KN
from ..tscore import cmi_weights, minmax
from ..bench import locomo

GRID_T = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]   # CMI is ~1e-3 nats (gold turns are rare), so T must reach that scale
V3 = {"width_scale": 2.0}

def features(item, store):
    """Candidates = every turn (v3 has no window). Returns s (min-max), K (v3 kernel), y (gold evidence)."""
    tq = item["tq"]; nowd = to_day(item["now"])
    q = store.embedder.encode([item["question"]])[0]
    s = minmax(store.sem_scores(q))
    sigma, tau, h = KN.scaled_width(tq.get("width") or "medium", V3["width_scale"])
    c = to_day(tq["center"]) if tq.get("center") else None
    sp = (to_day(tq["span"][0]), to_day(tq["span"][1])) if tq.get("span") else None
    if sp and sp[0] > sp[1]: sp = (sp[1], sp[0])
    K = KN.kernel(tq["mode"], store.days, nowd, c, sp, sigma, tau, h)
    ev = set(item["evidence"]); y = np.array([1 if m.id in ev else 0 for m in store.mems])
    return s, K, y

def recall(items, feats, weights):
    v = []
    for it in items:
        s, K, y = feats[it["question_id"]]; a, b = weights[it["tq"]["mode"]]
        top = np.argsort(-(a * s + b * K))[:10]; v.append(float(y[top].any()) if y.any() else 0.0)
    return float(np.mean(v)) if v else 0.0

def learn(items, feats):
    rows = []
    for it in items:
        s, K, y = feats[it["question_id"]]
        rows += [{"mode": it["tq"]["mode"], "s": float(a), "K": float(b), "y": int(c)} for a, b, c in zip(s, K, y)]
    best = None
    for T in GRID_T:
        w, diag = cmi_weights(rows, T); r = recall(items, feats, w)
        if best is None or r > best[0] + 1e-9: best = (r, T, w, diag)
    return best

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/locomo10.json"); ap.add_argument("--embedder", default="hash")
    ap.add_argument("--extractions", required=True); ap.add_argument("--out", default="configs/cmi_weights.json"); ap.add_argument("--report", default="reports/cmi_weights.md")
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(a.extractions, encoding="utf-8")]
    convs = {c.conv_id: c for c in locomo.load(a.data, "test")}
    emb = make_embedder(a.embedder); stores = {}
    for cid, c in convs.items():
        st = MemoryStore(emb); st.add_many(c.memories); stores[cid] = st
    qa = {q.qid: q for c in convs.values() for q in c.qas}
    items = [{"conv_id": r["conv_id"], "question_id": r["question_id"], "question": r["question"], "now": convs[r["conv_id"]].now,
              "tq": r["tq"], "evidence": qa[r["question_id"]].evidence}
             for r in recs if r.get("tq") and r["tq"].get("role") != "target" and r["tq"].get("mode") not in (None, "none")
             and r["conv_id"] in convs and r["question_id"] in qa]
    feats = {it["question_id"]: features(it, stores[it["conv_id"]]) for it in items}
    print(f"time-conditioned questions: {len(items)}", flush=True)
    out = {"method": "AssoMem 3.2.3 CMI weights, tertile bins, softmax(CMI/T), T from GRID_T by training recall@10", "grid_T": GRID_T,
           "source_extractions": os.path.basename(a.extractions), "folds": {}, "pooled": None}
    lines = ["# V8 CMI weights (leave-one-conversation-out)", "", "| held out | T | anchored (α,β) | range (α,β) | recent (α,β) | train r@10 | held-out r@10 | n |", "|---|---|---|---|---|---|---|---|"]
    fmt = lambda w, m: f"({w[m][0]:.2f}, {w[m][1]:.2f})" if m in w else "-"
    for cid in sorted(convs):
        tr = [it for it in items if it["conv_id"] != cid]; te = [it for it in items if it["conv_id"] == cid]
        r_tr, T, w, diag = learn(tr, feats); r_te = recall(te, feats, w)
        out["folds"][cid] = {"weights": w, "T": T, "cmi": diag, "train_recall10": round(r_tr, 4), "heldout_recall10": round(r_te, 4), "n_heldout": len(te)}
        lines.append(f"| {cid} | {T} | {fmt(w,'anchored')} | {fmt(w,'range')} | {fmt(w,'recent')} | {r_tr:.3f} | {r_te:.3f} | {len(te)} |")
    r_all, T, w, diag = learn(items, feats)
    out["pooled"] = {"weights": w, "T": T, "cmi": diag, "train_recall10": round(r_all, 4)}
    lines += ["", f"pooled (all eight, used on LME): T={T} anchored {fmt(w,'anchored')} range {fmt(w,'range')} recent {fmt(w,'recent')} r@10 {r_all:.3f}",
              "", "CMI (nats) per type, pooled: " + ", ".join(f"{m}: sem {v[0]:.4f} / time {v[1]:.4f} (n={v[2]})" for m, v in diag.items())]
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    os.makedirs(os.path.dirname(a.report), exist_ok=True); open(a.report, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
