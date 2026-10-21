"""Aggregate results: per-category F1/judge/recall, temporal split by role, cost; paired bootstrap between systems.
  python -m tqmem.eval.report results/e1/*.jsonl --pair rag+tq rag --pair rag+tq rag+hf"""
from __future__ import annotations
import argparse, json, glob, collections
import numpy as np

def load(paths):
    rows = []
    for p in paths:
        for line in open(p, encoding="utf-8"): rows.append(json.loads(line))
    return rows

def paired_bootstrap(a, b, n=1000, seed=42):
    a, b = np.asarray(a, float), np.asarray(b, float); d = a - b
    rng = np.random.default_rng(seed); idx = np.arange(len(d))
    boots = np.array([d[rng.choice(idx, len(idx), replace=True)].mean() for _ in range(n)])
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

def table(rows, metric="f1"):
    by = collections.defaultdict(list)
    for r in rows:
        if r.get(metric) is None: continue
        by[(r["system"], r["category"])].append(r[metric])
        if r["category"] == "temporal": by[(r["system"], f"temporal/{r.get('role','') or r.get('role_guess')}")].append(r[metric])
        if r.get("gold_mode"): by[(r["system"], f"mode/{r['gold_mode']}")].append(r[metric])
        elif r.get("mode"): by[(r["system"], f"mode/{r['mode']}")].append(r[metric])
        by[(r["system"], "all")].append(r[metric])
    systems = sorted({s for s, _ in by}); cats = sorted({c for _, c in by})
    out = [["system"] + cats + ["n"]]
    for s in systems:
        out.append([s] + [f"{np.mean(by[(s,c)]):.3f}" if (s, c) in by else "-" for c in cats] + [str(len(by[(s, 'all')]))])
    return out

def cost_table(rows):
    by = collections.defaultdict(list)
    for r in rows:
        by[r["system"]].append((r["retrieval_llm_calls"], r["retrieval_prompt_tokens"] + r["retrieval_completion_tokens"],
                                r["answer_prompt_tokens"], r["latency_retrieval_ms"]))
    out = [["system", "ret_calls", "ret_tokens", "ans_prompt_tokens", "lat_ret_p50", "lat_ret_p95"]]
    for s, v in sorted(by.items()):
        v = np.array(v, float)
        out.append([s, f"{v[:,0].mean():.2f}", f"{v[:,1].mean():.0f}", f"{v[:,2].mean():.0f}", f"{np.percentile(v[:,3],50):.0f}", f"{np.percentile(v[:,3],95):.0f}"])
    return out

def md(t):
    return "\n".join(["| " + " | ".join(r) + " |" if i != 1 else "|" + "---|" * len(t[0]) + "\n| " + " | ".join(r) + " |" for i, r in enumerate(t)])

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("paths", nargs="+"); ap.add_argument("--pair", nargs=2, action="append", default=[])
    ap.add_argument("--metric", default="f1"); ap.add_argument("--subset", default=None, help="category or temporal/constraint etc.")
    a = ap.parse_args()
    rows = load([p for pat in a.paths for p in glob.glob(pat)])
    print("## token-F1\n" + md(table(rows, "f1")))
    if any("retrieval_tokens" in r for r in rows): print("\n## retrieval tokens (mean)\n" + md(table(rows, "retrieval_tokens")))
    if any(r.get("judge") is not None for r in rows): print("\n## judge\n" + md(table(rows, "judge")))
    print("\n## recall@10\n" + md(table(rows, "recall10")))
    print("\n## cost\n" + md(cost_table(rows)))
    for s1, s2 in a.pair:
        def sel(s):
            d = {}
            for r in rows:
                if r["system"] != s: continue
                key = r["category"] if r["category"] != "temporal" else f"temporal/{r.get('role') or r.get('role_guess')}"
                mkey = f"mode/{r.get('gold_mode') or r.get('mode')}"
                if a.subset and not (key == a.subset or r["category"] == a.subset or mkey == a.subset): continue
                d[r["question_id"]] = r[a.metric]
            return d
        d1, d2 = sel(s1), sel(s2); common = sorted(set(d1) & set(d2))
        if not common: print(f"\n[pair] {s1} vs {s2}: no common questions"); continue
        m, lo, hi = paired_bootstrap([d1[q] for q in common], [d2[q] for q in common])
        print(f"\n[pair] {s1} − {s2} on {a.subset or 'all'} ({len(common)} q): Δ{a.metric}={m:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")

if __name__ == "__main__":
    main()
