"""Weight / width sensitivity curves for TQ-Mem (β curve per temporal mode). Offline: extractions are reused from a rag+tq run record
(CachedTranslator), embeddings come from the disk cache, so no LLM call and (after the first pass) no GPU pass.

For each extracted mode m in {anchored, range, recent}, on the questions extracted with that mode:
  - beta_m in 0.0 .. 0.9 (others at the base values)          -> recall@10 curve   (beta 0 = plain semantic ranking = rag)
  - width_scale in {0.25, 0.5, 1, 2, 3, 4}                      -> recall@10 curve
  - recent only: half-life h in {width table, 14, 30, 60, 90, 180} days
Base = configs/tuned_v3.yaml (the CV-converged v3 parameters) unless --tuned says otherwise. System = rag+tq v3 (no floor, full question,
no candidate window, min-max s_sem).

  python -m tqmem.eval.sensitivity --dataset locomo --split test --embedder BAAI/bge-large-en-v1.5 \
      --extractions results/e1/rag+tq-nt.fq.nowin-cv_bedrock_us.openai.gpt-5.6-luna_us-east-1_test_run1.jsonl --out reports/sensitivity_locomo.md
  python -m tqmem.eval.sensitivity --dataset lme --lme_part test --embedder BAAI/bge-large-en-v1.5 \
      --extractions results/e2/rag+tq-nt.fq.nowin_gateway_gpt-5.6-luna_lme-test_run1.jsonl --out reports/sensitivity_lme.md
"""
from __future__ import annotations
import argparse, json, os, yaml
import numpy as np
from ..embed import make_embedder
from ..store import MemoryStore
from ..bench.metrics import recall_at_k
from .. import retrieve as R
from .tune_cv import CachedTranslator

BETAS = [round(b * 0.1, 1) for b in range(10)]
WIDTHS = [0.25, 0.5, 1.0, 2.0, 3.0, 4.0]
HALFLIVES = [None, 14, 30, 60, 90, 180]

def base_params(path: str | None) -> dict:
    p = {"width_scale": 2.0, "weights": {"none": [1.0, 0.0], "recent": [0.9, 0.1], "range": [0.9, 0.1], "anchored": [0.7, 0.3]}, "h_recent": None, "sem_norm": "minmax"}
    if path:
        t = yaml.safe_load(open(path, encoding="utf-8")) or {}
        for k in ("width_scale", "weights", "h_recent", "sem_norm"):
            if k in t: p[k] = t[k]
    return p

def make_tq(store, tr, p, k=10):
    w = {m: tuple(v) for m, v in p["weights"].items()}
    return R.TQ(store, tr, k=k, weights=w, width_scale=p["width_scale"], theta_sem=None, use_window=False, h_recent=p["h_recent"], sem_norm=p["sem_norm"])

def items_locomo(data, split, records):
    from ..bench import locomo
    out = []
    for conv in locomo.load(data, None if split == "all" else split):
        for qa in conv.qas:
            r = records.get(qa.qid)
            if not r or (r.get("mode") or "none") == "none" or (r.get("tq") or {}).get("role") == "target": continue
            out.append({"qid": qa.qid, "group": conv.conv_id, "question": qa.question, "now": conv.now, "evidence": qa.evidence, "mode": r["mode"], "tq": r["tq"], "memories": conv.memories})
    return out

def items_lme(set_path, split_path, part, records):
    from ..bench import longmemeval as LME
    keep = None
    if split_path and part:
        sp = json.load(open(split_path, encoding="utf-8")); keep = set(sp["test_temporal"]) | set(sp["test_other"]) if part == "test" else set(sp[part])
    out = []
    for it in LME.load(set_path):
        if keep is not None and it.qid not in keep: continue
        r = records.get(it.qid)
        if not r or (r.get("mode") or "none") == "none" or (r.get("tq") or {}).get("role") == "target": continue
        out.append({"qid": it.qid, "group": it.qid, "question": it.question, "now": it.now, "evidence": it.gold_sessions, "mode": r["mode"], "tq": r["tq"], "memories": it.memories})
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="locomo", choices=["locomo", "lme"])
    ap.add_argument("--data", default="data/locomo10.json"); ap.add_argument("--split", default="test")
    ap.add_argument("--set", default="data/longmemeval_s.json"); ap.add_argument("--lme_split", default="data/lme_split.json"); ap.add_argument("--lme_part", default="test")
    ap.add_argument("--extractions", required=True, help="rag+tq run record: the recorded extraction of every question is reused")
    ap.add_argument("--embedder", default="BAAI/bge-large-en-v1.5"); ap.add_argument("--tuned", default="configs/tuned_v3.yaml")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    records = {r["question_id"]: r for r in (json.loads(l) for l in open(a.extractions, encoding="utf-8"))}
    items = items_locomo(a.data, a.split, records) if a.dataset == "locomo" else items_lme(a.set, a.lme_split, a.lme_part, records)
    emb = make_embedder(a.embedder); base = base_params(a.tuned)
    tr = CachedTranslator({it["question"]: it["tq"] for it in items})
    # group items by store (conversation / item) so each store is built once; evaluate every configuration on that store
    groups = {}
    for it in items: groups.setdefault(it["group"], []).append(it)
    configs = {}   # label -> params
    for mode in ("anchored", "range", "recent"):
        for b in BETAS:
            p = json.loads(json.dumps(base)); p["weights"][mode] = [round(1 - b, 2), b]; configs[f"beta[{mode}]={b}"] = p
        for w in WIDTHS:
            p = json.loads(json.dumps(base)); p["width_scale"] = w; configs[f"width[{mode}]={w}"] = p
    for h in HALFLIVES:
        p = json.loads(json.dumps(base)); p["h_recent"] = h; configs[f"h_recent={h}"] = p
    configs["base"] = json.loads(json.dumps(base))
    res = {lab: {} for lab in configs}   # label -> qid -> recall
    for gi, (g, its) in enumerate(groups.items()):
        st = MemoryStore(emb); st.add_many(its[0]["memories"])
        for lab, p in configs.items():
            mode_of = lab.split("[")[1].split("]")[0] if "[" in lab else None
            tq = make_tq(st, tr, p)
            for it in its:
                if mode_of and it["mode"] != mode_of: continue
                if lab.startswith("h_recent") and it["mode"] != "recent": continue
                hits, _, _ = tq.retrieve(it["question"], it["now"])
                res[lab][it["qid"]] = recall_at_k([h["id"] for h in hits], it["evidence"])
        if (gi + 1) % 20 == 0 or gi + 1 == len(groups): print(f"[sensitivity] {gi + 1}/{len(groups)} stores", flush=True)
    n_mode = {m: sum(1 for it in items if it["mode"] == m) for m in ("anchored", "range", "recent")}
    lines = [f"# Sensitivity — {a.dataset} ({'split ' + a.split if a.dataset == 'locomo' else 'part ' + a.lme_part}), rag+tq v3, extractions from {os.path.basename(a.extractions)}",
             f"base = {json.dumps(base)}; questions by extracted mode: {n_mode}. recall@10 on the questions of each mode; beta 0 = plain semantic ranking.", ""]
    def curve(prefix, mode, keys, fmt=lambda v: str(v)):
        row = []
        for v in keys:
            lab = f"{prefix}[{mode}]={v}" if prefix != "h_recent" else f"h_recent={v}"
            vals = list(res[lab].values()); row.append(f"{np.mean(vals):.3f}" if vals else "—")
        return row
    lines.append("## beta curve (weight of the kernel; others at base)\n")
    lines.append("| mode (n) | " + " | ".join(f"β={b}" for b in BETAS) + " |"); lines.append("|---|" + "---|" * len(BETAS))
    for m in ("anchored", "range", "recent"): lines.append(f"| {m} ({n_mode[m]}) | " + " | ".join(curve("beta", m, BETAS)) + " |")
    lines.append("\n## width scale curve (kernel width multiplier; others at base)\n")
    lines.append("| mode (n) | " + " | ".join(f"×{w}" for w in WIDTHS) + " |"); lines.append("|---|" + "---|" * len(WIDTHS))
    for m in ("anchored", "range", "recent"): lines.append(f"| {m} ({n_mode[m]}) | " + " | ".join(curve("width", m, WIDTHS)) + " |")
    lines.append("\n## recent half-life (days; None = width table x scale)\n")
    lines.append("| " + " | ".join(f"h={h}" for h in HALFLIVES) + " |"); lines.append("|" + "---|" * len(HALFLIVES))
    lines.append("| " + " | ".join(curve("h_recent", "recent", HALFLIVES)) + " |")
    text = "\n".join(lines); print(text)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); open(a.out, "w", encoding="utf-8").write(text + "\n")
        json.dump({"base": base, "n_mode": n_mode, "results": {lab: float(np.mean(list(v.values()))) if v else None for lab, v in res.items()}}, open(a.out.replace(".md", ".json"), "w", encoding="utf-8"), indent=1)
        print(f"wrote {a.out}")

if __name__ == "__main__":
    main()
