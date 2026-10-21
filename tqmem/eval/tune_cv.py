"""Leave-one-conversation-out tuning on LoCoMo test (decision 2026-09-15: the LME temporal dev set is target-type and saturates).

For each held-out conversation, coordinate-descent on the other conversations' time-constrained questions (LLM-extracted mode != none,
extractions reused from an earlier rag+tq run so this needs no LLM calls), then score the held-out conversation with the chosen
parameters. Objective = recall@10 of the RAG store under the v3 configuration: no semantic floor, full-question search, no candidate window.

  python -m tqmem.eval.tune_cv --embedder BAAI/bge-large-en-v1.5 \
      --extractions "results/e1/rag+tq_bedrock_us.openai.gpt-5.6-luna_us-east-1_test_run1.jsonl" --out configs/tuned_cv.json

Writes configs/tuned_cv.json  {"folds": {conv_id: params}, "pooled": {...}, "grid": ...}  and reports/tune_cv.md.
"""
from __future__ import annotations
import argparse, json, os, itertools
import numpy as np
from ..embed import make_embedder
from ..store import MemoryStore
from ..translate import Translator, TemporalQuery
from .. import retrieve as R
from .. import kernels as KN
from ..bench import locomo
from ..bench.metrics import recall_at_k

class CachedTranslator(Translator):
    """Serve the extraction recorded for each question (tq dict from a run record); query=None so the full question is embedded."""
    def __init__(self, table: dict): self.table = table
    def translate(self, question, now):
        d = self.table[question]
        tq = TemporalQuery(query=None, role=d.get("role", "constraint"), mode=d.get("mode", "none"), center=d.get("center"),
                           span=tuple(d["span"]) if d.get("span") else None, width=d.get("width", "medium"))
        return tq, {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "latency_ms": 0.0}

DEFAULT = {"width_scale": 1.0, "weights": {"none": [1.0, 0.0], "recent": [0.7, 0.3], "range": [0.5, 0.5], "anchored": [0.4, 0.6]},
           "h_recent": None, "sem_norm": "raw"}
GRID = {"width_scale": [0.5, 1.0, 2.0], "beta": [0.1, 0.3, 0.5, 0.7], "h_recent": [14, 30, 60, 90], "sem_norm": ["raw", "minmax"]}

def make_tq(store, tr, p):
    w = {k: tuple(v) for k, v in p["weights"].items()}
    return R.TQ(store, tr, weights=w, width_scale=p["width_scale"], theta_sem=None, use_window=False, h_recent=p["h_recent"], sem_norm=p["sem_norm"])

def score(items, p, stores, tr):
    vals = []
    for it in items:
        hits, _, _ = make_tq(stores[it["conv_id"]], tr, p).retrieve(it["question"], it["now"])
        vals.append(recall_at_k([h["id"] for h in hits], it["evidence"]))
    return float(np.mean(vals)) if vals else 0.0

def descend(items, stores, tr, log):
    p = json.loads(json.dumps(DEFAULT)); best = score(items, p, stores, tr)
    def try_set(label, mutate):
        nonlocal p, best
        cand = json.loads(json.dumps(p)); mutate(cand); v = score(items, cand, stores, tr); log.append((label, round(v, 4)))
        if v > best + 1e-9: p, best = cand, v
    for sn in GRID["sem_norm"]: try_set(f"sem_norm={sn}", lambda c, sn=sn: c.update(sem_norm=sn))
    for ws in GRID["width_scale"]: try_set(f"width_scale={ws}", lambda c, ws=ws: c.update(width_scale=ws))
    for mode in ["anchored", "range", "recent"]:
        for b in GRID["beta"]: try_set(f"beta[{mode}]={b}", lambda c, mode=mode, b=b: c["weights"].__setitem__(mode, [round(1 - b, 2), b]))
    for h in GRID["h_recent"]: try_set(f"h_recent={h}", lambda c, h=h: c.update(h_recent=h))
    return p, best

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/locomo10.json"); ap.add_argument("--embedder", default="hash")
    ap.add_argument("--extractions", required=True, help="a run record file whose `tq` field holds the extraction per question")
    ap.add_argument("--out", default="configs/tuned_cv.json"); ap.add_argument("--report", default="reports/tune_cv.md")
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(a.extractions, encoding="utf-8")]
    table = {r["question"]: (r["tq"] or {"mode": "none"}) for r in recs}
    convs = {c.conv_id: c for c in locomo.load(a.data, "test")}
    emb = make_embedder(a.embedder); stores = {}
    for cid, c in convs.items():
        st = MemoryStore(emb); st.add_many(c.memories); stores[cid] = st
    items = [{"conv_id": r["conv_id"], "question": r["question"], "now": convs[r["conv_id"]].now,
              "evidence": next(q.evidence for q in convs[r["conv_id"]].qas if q.qid == r["question_id"])}
             for r in recs if (r["tq"] or {}).get("mode") not in (None, "none") and r["conv_id"] in convs]
    tr = CachedTranslator(table)
    print(f"time-constrained questions: {len(items)} over {len(convs)} conversations", flush=True)
    folds, held, held_default, logs = {}, [], [], {}
    for cid in sorted(convs):
        train = [it for it in items if it["conv_id"] != cid]; test = [it for it in items if it["conv_id"] == cid]
        log = []; p, v_train = descend(train, stores, tr, log); logs[cid] = log
        v_test = score(test, p, stores, tr); v_def = score(test, DEFAULT, stores, tr)
        folds[cid] = {"params": p, "train_recall10": round(v_train, 4), "heldout_recall10": round(v_test, 4), "heldout_default_recall10": round(v_def, 4), "n_heldout": len(test)}
        held += [v_test] * len(test); held_default += [v_def] * len(test)
        print(f"[fold {cid}] train {v_train:.3f} -> held-out {v_test:.3f} (default {v_def:.3f}, n={len(test)}) | {json.dumps(p['weights'])} ws={p['width_scale']} h={p['h_recent']} sn={p['sem_norm']}", flush=True)
    pooled = {"heldout_recall10": round(float(np.mean(held)), 4), "heldout_default_recall10": round(float(np.mean(held_default)), 4), "n": len(held)}
    out = {"method": "leave-one-conversation-out on LoCoMo test, objective recall@10 (RAG store, no theta, full question, no window)",
           "source_extractions": a.extractions, "grid": GRID, "default": DEFAULT, "folds": folds, "pooled": pooled}
    os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    os.makedirs(os.path.dirname(a.report), exist_ok=True)
    with open(a.report, "w", encoding="utf-8") as f:
        f.write(f"# Leave-one-conversation-out tuning (LoCoMo test)\n\npooled held-out recall@10 **{pooled['heldout_recall10']}** vs default {pooled['heldout_default_recall10']} (n={pooled['n']})\n\n")
        f.write("| held-out | n | train | held-out | default | width | β anchored/range/recent | h_recent | sem_norm |\n|---|---|---|---|---|---|---|---|---|\n")
        for cid, fo in folds.items():
            p = fo["params"]; w = p["weights"]
            f.write(f"| {cid} | {fo['n_heldout']} | {fo['train_recall10']} | {fo['heldout_recall10']} | {fo['heldout_default_recall10']} | {p['width_scale']} | {w['anchored'][1]}/{w['range'][1]}/{w['recent'][1]} | {p['h_recent']} | {p['sem_norm']} |\n")
        f.write("\n## coordinate-descent logs\n")
        for cid, log in logs.items(): f.write(f"- {cid}: " + ", ".join(f"{k}={v}" for k, v in log) + "\n")
    print("pooled:", pooled); print(f"wrote {a.out}, {a.report}")

if __name__ == "__main__":
    main()
