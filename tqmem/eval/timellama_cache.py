"""Build the TimeLLaMA time-string cache for V7a (AssoMem paper version: TimeLLaMA embeddings of the query's time condition and of
the memory timestamps, cosine similarity). 2026-09-29.

AssoMem does not say how a sentence vector is read out of TimeLLaMA; we use the mean of the last hidden layer over the tokens,
L2-normalised. Only short strings are embedded (dates and rendered time conditions), so bf16 on CPU is enough.

  python -m tqmem.eval.timellama_cache --refs <reference run jsonl> [...] --out .temb_cache/timellama.jsonl

Strings: every memory date of every LoCoMo test conversation and of every LME haystack whose question is time-conditioned in a
reference file, plus each time-conditioned question's rendered condition (tscore.query_time_text).
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
from ..store import to_day
from ..bench import locomo, longmemeval as LME
from .. import tscore as TS
from .. import kernels as KN  # noqa: F401

MODEL = "chrisyuan45/TimeLlama-7b"

def timed(r):
    tq = r.get("tq") or {}
    return tq.get("role") != "target" and tq.get("mode", "none") != "none" and not tq.get("aggregate") and not r.get("error")

def collect(refs, locomo_path, lme_path):
    texts = set(); lme_ids = set(); convs = {c.conv_id: c for c in locomo.load(locomo_path, "all")}
    nows = {c.conv_id: c.now for c in convs.values()}
    for p in refs:
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if not timed(r): continue
            tq = r["tq"]; now = nows.get(r["conv_id"])
            if r.get("dataset") == "lme": lme_ids.add(r["question_id"]); now = None
            c = to_day(tq["center"]) if tq.get("center") else None
            sp = (to_day(tq["span"][0]), to_day(tq["span"][1])) if tq.get("span") else None
            if sp and sp[0] > sp[1]: sp = (sp[1], sp[0])
            texts.add(("q", r["question_id"], tq["mode"], c, sp, now))
    out = set()
    for c in convs.values(): out |= {TS.date_text(m.day) for m in c.memories}
    lme_now = {}
    if lme_ids:
        for it in LME.load(lme_path, None):
            if it.qid in lme_ids: out |= {TS.date_text(m.day) for m in it.memories}; lme_now[it.qid] = it.now
    for _, qid, mode, c, sp, now in texts:
        out.add(TS.query_time_text(mode, to_day(now or lme_now[qid]), c, sp))
    return sorted(out)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--refs", nargs="+", required=True); ap.add_argument("--out", default=".temb_cache/timellama.jsonl")
    ap.add_argument("--locomo", default="data/locomo10.json"); ap.add_argument("--lme", default="data/longmemeval_s.json")
    ap.add_argument("--device", default="cpu"); ap.add_argument("--batch", type=int, default=16); ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    strings = collect(a.refs, a.locomo, a.lme)
    done = set()
    if os.path.exists(a.out): done = {json.loads(l)["text"] for l in open(a.out, encoding="utf-8")}
    todo = [s for s in strings if s not in done]
    print(f"{len(strings)} strings, {len(todo)} to embed -> {a.out}", flush=True)
    if a.dry or not todo: return
    try:
        import truststore; truststore.inject_into_ssl()
    except Exception: pass
    import torch
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token = tok.pad_token or tok.eos_token; tok.padding_side = "right"
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(a.device).eval()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f, torch.no_grad():
        for i in range(0, len(todo), a.batch):
            b = todo[i:i + a.batch]; enc = tok(b, return_tensors="pt", padding=True).to(a.device)
            h = model(**enc).last_hidden_state.float(); m = enc["attention_mask"].unsqueeze(-1).float()
            v = ((h * m).sum(1) / m.sum(1)).cpu().numpy(); v /= np.linalg.norm(v, axis=1, keepdims=True)
            for t, x in zip(b, v): f.write(json.dumps({"text": t, "vec": [round(float(y), 6) for y in x]}) + "\n")
            f.flush(); print(f"  {min(i + a.batch, len(todo))}/{len(todo)}", flush=True)

if __name__ == "__main__":
    main()
