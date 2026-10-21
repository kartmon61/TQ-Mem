"""E2(c) / RQ2(b): recall by distance between the gold evidence date and the hard-filter window.

For every time-constrained question (hard-filter extraction mode != none) compute
    dist = 0 if some gold evidence day lies inside the HF window [lo, hi], else the days to the nearest boundary,
bin it into  inside (dist = 0) / just outside (0 < dist <= w) / far (dist > w)  with w = the kernel width of that extraction
(sigma for anchored, tau for range, half-life for recent, same width table and scale as the run), and report per bin the recall@10 of
the hard filter and of TQ-Mem with the paired-bootstrap CI of the difference. Reads existing run records only: no model, no LLM.

  python -m tqmem.eval.boundary --dataset locomo --data data/locomo10.json --split test \
      --pair rag+hf=results/e1/rag+hf_..._test_run1.jsonl rag+tq=results/e1/rag+tq-nt.fq.nowin-cv_..._test_run1.jsonl \
      --pair mem0+hf=... mem0+tq=...  --out reports/boundary_locomo.md
  python -m tqmem.eval.boundary --dataset lme --set data/longmemeval_s.json --lme_split data/lme_split.json --lme_part test --pair ... --out reports/boundary_lme.md
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
from ..store import to_day
from ..translate import TemporalQuery
from .. import kernels as KN
from ..retrieve import _resolve
from .report import paired_bootstrap

def load_records(path: str) -> dict:
    return {r["question_id"]: r for r in (json.loads(l) for l in open(path, encoding="utf-8"))}

def gold_days_locomo(data: str, split: str | None) -> tuple[dict, dict]:
    """question_id -> (gold days, now)"""
    from ..bench import locomo
    out = {}
    for conv in locomo.load(data, split):
        day = {m.id: m.day for m in conv.memories}
        for qa in conv.qas:
            out[qa.qid] = ([day[e] for e in qa.evidence if e in day], conv.now)
    return out

def gold_days_lme(set_path: str, split_path: str | None, part: str | None) -> dict:
    from ..bench import longmemeval as LME
    keep = None
    if split_path and part:
        sp = json.load(open(split_path, encoding="utf-8"))
        keep = set(sp["test_temporal"]) | set(sp["test_other"]) if part == "test" else set(sp[part])
    out = {}
    for it in LME.load(set_path):
        if keep is not None and it.qid not in keep: continue
        day = {}
        for m in it.memories: day.setdefault(m.meta["session_id"], m.day)
        out[it.qid] = ([day[s] for s in it.gold_sessions if s in day], it.now)
    return out

def hf_window(tqd: dict, now: str, width_scale: float = 1.0):
    """(lo, hi, w) of the hard-filter window for a recorded extraction, or None for mode none / target."""
    if not tqd or tqd.get("mode") in (None, "none") or tqd.get("role") == "target": return None
    tq = TemporalQuery(query=None, role=tqd.get("role", "constraint"), mode=tqd["mode"], center=tqd.get("center"),
                       span=tuple(tqd["span"]) if tqd.get("span") else None, width=tqd.get("width", "medium"))
    try:
        sigma, tau, h, c, sp = _resolve(tq, now, width_scale)
        lo, hi = KN.hard_window(tq.mode, to_day(now), c, sp, sigma, tau, h)
    except Exception:
        return None
    w = {"anchored": sigma, "range": tau, "recent": h}[tq.mode]
    return lo, hi, w

def distance(gold: list[float], lo: float, hi: float) -> float | None:
    if not gold: return None
    ds = [0.0 if lo <= g <= hi else min(abs(g - lo), abs(g - hi)) for g in gold]
    return min(ds)

def analyse(pairs: list[tuple[str, dict, str, dict]], gold: dict, width_scale: float = 1.0) -> list[str]:
    """pairs: (hf_name, hf_records, tq_name, tq_records). Bins come from the HF extraction of each pair."""
    lines = []
    for hf_name, hf, tq_name, tq in pairs:
        bins = {"inside (dist = 0)": [], "just outside (0 < dist <= w)": [], "far (dist > w)": []}
        n_const = n_nogold = 0
        for qid, r in hf.items():
            if qid not in tq or qid not in gold: continue
            win = hf_window(r.get("tq"), gold[qid][1], width_scale)
            if win is None: continue
            n_const += 1
            d = distance(gold[qid][0], win[0], win[1])
            if d is None: n_nogold += 1; continue
            key = "inside (dist = 0)" if d == 0 else ("just outside (0 < dist <= w)" if d <= win[2] else "far (dist > w)")
            bins[key].append(qid)
        lines.append(f"\n### {tq_name} − {hf_name}  (constraint questions by HF extraction: {n_const}; without gold date: {n_nogold})\n")
        lines.append("| bin | n | recall@10 HF | recall@10 TQ | Δ recall [95% CI] | judge HF | judge TQ | Δ judge [95% CI] |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for key, qs in bins.items():
            if not qs:
                lines.append(f"| {key} | 0 | — | — | — | — | — | — |"); continue
            rh = [hf[q]["recall10"] or 0 for q in qs]; rt = [tq[q]["recall10"] or 0 for q in qs]
            jh = [hf[q]["judge"] or 0 for q in qs]; jt = [tq[q]["judge"] or 0 for q in qs]
            d, lo, hi = paired_bootstrap(rt, rh); dj, loj, hij = paired_bootstrap(jt, jh)
            flag = " *" if (lo > 0 or hi < 0) else ""
            lines.append(f"| {key} | {len(qs)} | {np.mean(rh):.3f} | {np.mean(rt):.3f} | {d:+.3f} [{lo:+.3f}, {hi:+.3f}]{flag} | {np.mean(jh):.3f} | {np.mean(jt):.3f} | {dj:+.3f} [{loj:+.3f}, {hij:+.3f}]{' *' if (loj > 0 or hij < 0) else ''} |")
        lines.append("\n`*` = 95% CI excludes 0. RQ2(b) reads the 'just outside' row: a CI lower bound above 0 means the soft boundary recovers evidence the hard filter cuts. Bins with n < 30 are reported for the CI only (설계서 §2).")
    return lines

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="locomo", choices=["locomo", "lme"])
    ap.add_argument("--data", default="data/locomo10.json"); ap.add_argument("--split", default="test")
    ap.add_argument("--set", default="data/longmemeval_s.json"); ap.add_argument("--lme_split", default="data/lme_split.json"); ap.add_argument("--lme_part", default="test")
    ap.add_argument("--pair", nargs=2, action="append", required=True, metavar="NAME=PATH", help="hf NAME=PATH tq NAME=PATH")
    ap.add_argument("--width_scale", type=float, default=1.0, help="width scale used by the HF run (default 1.0)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    gold = gold_days_locomo(a.data, None if a.split == "all" else a.split) if a.dataset == "locomo" else gold_days_lme(a.set, a.lme_split, a.lme_part)
    pairs = []
    for hf_spec, tq_spec in a.pair:
        hn, hp = hf_spec.split("=", 1); tn, tp = tq_spec.split("=", 1)
        pairs.append((hn, load_records(hp), tn, load_records(tp)))
    lines = [f"# Boundary-distance analysis — {a.dataset} ({'split ' + a.split if a.dataset == 'locomo' else 'part ' + a.lme_part})",
             "dist = days from the nearest gold evidence date to the hard-filter window of the HF extraction (0 inside); w = kernel width of that extraction (sigma / tau / half-life, width table x scale)."]
    lines += analyse(pairs, gold, a.width_scale)
    text = "\n".join(lines); print(text)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); open(a.out, "w", encoding="utf-8").write(text + "\n"); print(f"\nwrote {a.out}")

if __name__ == "__main__":
    main()
