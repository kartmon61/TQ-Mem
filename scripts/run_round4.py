"""4th-round experiments (2026-09-29): ablations (RQ5), AssoMem comparisons V7a/V7b/V8, Mem0 global-decay / iterative-call rows.

Every variant replays the extraction of the TQ-Mem v3 reference run and runs only the time-conditioned questions (the others rank
exactly as the reference). `--stage ret` = retrieval metrics only (no LLM call); `--stage full` = answers + judge on Bedrock.

  python scripts/run_round4.py --stage ret            # free
  python scripts/run_round4.py --stage full           # paid (Luna on Bedrock)
  python scripts/run_round4.py --stage full --only lme_rag --workers 8
"""
from __future__ import annotations
import argparse, subprocess, sys, yaml

VARIANTS = ["", "notype", "fixw", "mul", "tvec", "asso", "temb", "temb-tl", "cmi"]   # "" = the v3 reference itself, replayed; temb-tl = temb with TimeLLaMA
BGE = "BAAI/bge-large-en-v1.5"
LME = ["--dataset", "lme", "--set", "data/longmemeval_s.json"]
SETS = {
    "locomo_rag": dict(base="rag+tq:nt.fq.nowin", ref="results/e1/rag+tq-nt.fq.nowin-cv_bedrock_us.openai.gpt-5.6-luna_us-east-1_test_run1.jsonl",
                       args=["--split", "test", "--tuned_cv", "configs/tuned_cv.json"]),
    "locomo_mem0": dict(base="mem0+tq:nt.fq", ref="results/e1/mem0+tq-nt.fq-cv_bedrock_us.openai.gpt-5.6-luna_us-east-1_test_run1.jsonl",
                        args=["--split", "test", "--tuned_cv", "configs/tuned_cv.json", "--mem0_config", "configs/mem0.yaml"]),
    "lme_rag": dict(base="rag+tq:nt.fq.nowin", ref="results/e2/rag+tq-nt.fq.nowin_gateway_gpt-5.6-luna_lme-test_run1.jsonl",
                    args=LME + ["--lme_part", "test", "--tuned", "configs/tuned_v3.yaml"]),
    "lme_mem0": dict(base="mem0+tq:nt.fq", ref="results/e2/mem0+tq-nt.fq_gateway_gpt-5.6-luna_lme-test_temporal_run1.jsonl",
                     args=LME + ["--lme_part", "test_temporal", "--tuned", "configs/tuned_v3.yaml", "--mem0_config", "configs/mem0_lme.yaml", "--mem0_unit", "session"]),
}
# family B (Claude Haiku 4.5): answer + extraction by Haiku, judge stays Luna; references = round-3 family-B v3 runs
# (Haiku's own extraction). LME Mem0 has no family-B reference (not run in round 3; 27 timed questions, noise-dominated) -> not replicated.
HAIKU = "bedrock_us.anthropic.claude-haiku-4-5-20251001-v1_0_us-east-1"
SETS_B = {
    "locomo_rag": dict(SETS["locomo_rag"], ref=f"results/e1_famB/rag+tq-nt.fq.nowin-cv_{HAIKU}_test_run1.jsonl"),
    "locomo_mem0": dict(SETS["locomo_mem0"], ref=f"results/e1_famB/mem0+tq-nt.fq-cv_{HAIKU}_test_run1.jsonl"),
    "lme_rag": dict(SETS["lme_rag"], ref="results/e2_famB/rag+tq-nt.fq.nowin_gateway_claude-haiku-4-5-20251001_lme-test_run1.jsonl"),
}
# Mem0 rows missing in round 3 (all questions, like the other E1/E2 system rows; no replay)
BASELINES = {
    "locomo_mem0": [("mem0+ga", ["--split", "test", "--mem0_config", "configs/mem0.yaml"]), ("mem0+cl", ["--split", "test", "--mem0_config", "configs/mem0.yaml"])],
    "lme_mem0": [(s, LME + ["--lme_part", "test_temporal", "--mem0_config", "configs/mem0_lme.yaml", "--mem0_unit", "session"]) for s in ("mem0+ga", "mem0+cl")],
}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", choices=["ret", "full"], required=True)
    ap.add_argument("--only", default=None, help="comma list of set names"); ap.add_argument("--variants", default=None, help="comma list (use 'ref' for the replayed reference)")
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--baselines", action="store_true", help="also run the Mem0 ga / cl rows (full stage)")
    ap.add_argument("--dry", action="store_true"); ap.add_argument("--family", choices=["A", "B"], default="A")
    a = ap.parse_args()
    c = yaml.safe_load(open("configs/common.yaml", encoding="utf-8"))
    B = a.family == "B"; sets_def = SETS_B if B else SETS; suf = "_famB" if B else ""
    llm = ["--llm", c["llm_family_B" if B else "llm_family_A"], "--judge_llm", c["judge_llm"], "--embedder", BGE, "--translator", "llm", "--workers", str(a.workers)]
    sets = a.only.split(",") if a.only else list(sets_def)
    vs = [("" if v == "ref" else v) for v in a.variants.split(",")] if a.variants else VARIANTS
    jobs = []
    for name in sets:
        s = sets_def[name]; out = "results/e3r4" + suf + ("_ret" if a.stage == "ret" else "")
        for v in vs:
            sysn = s["base"] + (f".{v.replace('-tl', '')}" if v else "")
            cmd = ["--system", sysn, "--replay", s["ref"], "--only_timed", "--out", out] + s["args"] + (["--cmi", f"configs/cmi_weights{suf}.json"] if v == "cmi" else [])
            if v == "temb-tl": cmd += ["--time_embedder", "timellama"]
            if a.stage == "ret": cmd.append("--retrieval_only")
            jobs.append((f"{name}:{v or 'ref'}", cmd))
        if a.stage == "full" and a.baselines:
            for sysn, extra in BASELINES.get(name, []):
                jobs.append((f"{name}:{sysn}", ["--system", sysn, "--out", ("results/e1" if name.startswith("locomo") else "results/e2") + suf] + extra))
    for label, cmd in jobs:
        full = [sys.executable, "-m", "tqmem.eval.run"] + cmd + llm
        print(f"=== {label}\n    {' '.join(cmd)}", flush=True)
        if a.dry: continue
        r = subprocess.run(full, capture_output=True, text=True, encoding="utf-8", errors="replace")
        tail = [l for l in (r.stdout + r.stderr).splitlines() if l.startswith(("wrote", "[ingest]", "[task error]", "Traceback")) or "Error" in l]
        print("    " + "\n    ".join(tail[-6:]), flush=True)
        if r.returncode != 0: print(f"    !! exit {r.returncode}: {r.stderr[-600:]}", flush=True)

if __name__ == "__main__":
    main()
