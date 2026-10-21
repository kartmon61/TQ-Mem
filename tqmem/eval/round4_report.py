"""Summarise 4th-round runs (2026-09-29) against the replayed TQ-Mem v3 reference on the same time-conditioned questions.

  python -m tqmem.eval.round4_report --dir results/e3r4_ret --out reports/round4_ret.md
  python -m tqmem.eval.round4_report --dir results/e3r4 --out reports/round4.md

For every set: the reference row (from the reference file, restricted to the questions run), the plain store (rag / mem0 on the same
questions, from the round-3 files) and each variant, with paired bootstrap 95% CI of variant - reference.
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np

LABEL = {"ref": "TQ-Mem v3 (replayed)", "notype": "① 유형 구분 제거", "fixw": "② 유형별 가중치 고정", "mul": "③ 곱하기 결합",
         "tvec": "④ 커널 대신 벡터에 시간 값", "asso": "V7b AssoMem 공개 코드 감쇠", "temb": "V7a AssoMem 시간 임베딩",
         "temb-tl": "V7a AssoMem 시간 임베딩 (TimeLLaMA)", "cmi": "V8 CMI 학습 가중치"}
PLAIN = {"locomo_rag": "results/e1/rag_bedrock_us.openai.gpt-5.6-luna_us-east-1_test_run1.jsonl",
         "locomo_mem0": "results/e1/mem0_bedrock_us.openai.gpt-5.6-luna_us-east-1_test_run1.jsonl",
         "lme_rag": "results/e2/rag_gateway_gpt-5.6-luna_lme-test_run1.jsonl",
         "lme_mem0": "results/e2/mem0_gateway_gpt-5.6-luna_lme-test_temporal_run1.jsonl"}
PLAIN_B = {"locomo_rag": "results/e1_famB/rag-cv_bedrock_us.anthropic.claude-haiku-4-5-20251001-v1_0_us-east-1_test_run1.jsonl",
           "locomo_mem0": "results/e1_famB/mem0-cv_bedrock_us.anthropic.claude-haiku-4-5-20251001-v1_0_us-east-1_test_run1.jsonl",
           "lme_rag": "results/e2_famB/rag_gateway_claude-haiku-4-5-20251001_lme-test_run1.jsonl"}
METRICS = ["recall10", "f1", "judge"]

def load(p): return {r["question_id"]: r for r in (json.loads(l) for l in open(p, encoding="utf-8")) if not r.get("error")}

def boot(d, n=1000, seed=42):
    d = np.asarray(d, dtype=float); rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(1); return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)

def classify(fname):
    ds = "lme" if "_lme-" in fname else "locomo"; st = "mem0" if fname.startswith("mem0") else "rag"
    sysn = fname.split("_")[0]; parts = sysn.split("-")
    var = "ref"
    for p in parts[1:]:
        for tok in p.split("."):
            if tok in LABEL: var = tok
    if "-tl" in sysn and var == "temb": var = "temb-tl"
    return f"{ds}_{st}", var

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--family", choices=["A", "B"], default="A", help="B = Claude Haiku 4.5 answers (plain-store rows from the family-B files)"); a = ap.parse_args()
    plain = PLAIN_B if a.family == "B" else PLAIN
    runs = {}
    for f in sorted(glob.glob(os.path.join(a.dir, "*.jsonl"))):
        name = os.path.basename(f)
        if name.startswith("write_cost"): continue
        key, var = classify(name); runs.setdefault(key, {})[var] = f
    lines = [f"# 4차 실험 요약 ({os.path.basename(a.dir)})", "",
             "모든 변형은 TQ-Mem v3 기준 실행의 시간 조건 추출을 그대로 재생하고, 시간 조건이 걸린 문항만 실행했다(나머지 문항은 기준과 순위가 같다).",
             "Δ = 변형 − 기준, 괄호는 문항 단위 paired bootstrap 95% CI(1,000회). **굵게** = CI가 0을 포함하지 않음.", ""]
    for key in ["locomo_rag", "locomo_mem0", "lme_rag", "lme_mem0"]:
        if key not in runs: continue
        vr = runs[key]; any_f = next(iter(vr.values()))
        qids = list(load(any_f))
        ref_src = json.loads(open(any_f, encoding="utf-8").readline()).get("replayed_from")
        ref_path = next((p for p in glob.glob("results/e*/*") if os.path.basename(p) == ref_src), None)
        ref = load(ref_path) if ref_path else {}
        has_ans = any(json.loads(open(f, encoding="utf-8").readline()).get("f1") is not None for f in vr.values())
        mets = METRICS if has_ans else ["recall10"]
        lines += [f"## {key} — {len(qids)}문항", "", f"기준 파일: `{ref_src}`", "",
                  "| 시스템 | " + " | ".join(mets) + " | " + " | ".join(f"Δ{m}" for m in mets) + " |", "|---" * (1 + 2 * len(mets)) + "|"]
        def row(label, recs, base):
            cells = []
            for m in mets:
                v = [recs[q][m] for q in qids if q in recs and recs[q].get(m) is not None]; cells.append(f"{np.mean(v):.3f}" if v else "-")
            for m in mets:
                d = [recs[q][m] - base[q][m] for q in qids if q in recs and q in base and recs[q].get(m) is not None and base[q].get(m) is not None]
                if len(d) < 2 or base is recs: cells.append("—"); continue
                mu, lo, hi = boot(d); s = f"{mu:+.3f} [{lo:+.3f}, {hi:+.3f}]"; cells.append(f"**{s}**" if lo > 0 or hi < 0 else s)
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        base = load(vr["ref"]) if "ref" in vr and has_ans else ref
        if plain.get(key) and os.path.exists(plain[key]): row("(시간 처리 없음) " + key.split("_")[1], load(plain[key]), base)
        if ref: row("TQ-Mem v3 (3차 기준 파일)", ref, base)
        for var in ["ref", "notype", "fixw", "mul", "tvec", "asso", "temb", "temb-tl", "cmi"]:
            if var in vr: row(LABEL[var], load(vr[var]), base)
        # replay fidelity: the replayed reference must rank exactly as the reference file
        if "ref" in vr and ref:
            rr = load(vr["ref"]); same = sum(1 for q in qids if q in ref and rr[q]["recall10"] == ref[q]["recall10"])
            ids = sum(1 for q in qids if q in ref and [h["id"] for h in rr[q]["retrieved"]] == [h["id"] for h in ref[q]["retrieved"]])
            lines += ["", f"재생 검증: recall@10 일치 {same}/{len(qids)}, top-10 id 목록 일치 {ids}/{len(qids)}"]
        lines.append("")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); open(a.out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
