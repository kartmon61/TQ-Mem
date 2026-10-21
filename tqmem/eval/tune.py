"""Coordinate-descent tuning on dev: width_scale -> beta per mode -> theta_sem. Writes configs/tuned.yaml.

Primary (design): LongMemEval_S temporal-reasoning dev 30 (seed 42, data/lme_split.json):
  python -m tqmem.eval.tune --dataset lme --set data/longmemeval_s.json --lme_split data/lme_split.json \
         --embedder BAAI/bge-large-en-v1.5 --llm bedrock:us.openai.gpt-5.6-luna@us-east-1 --translator llm
Fallback (LME unavailable): LoCoMo dev questions -> configs/tuned.yaml gets `provisional: true`.
Objective = recall@10 of rag+tq over the dev items; the time-condition extraction is cached per question so the grid costs one LLM call per item."""
from __future__ import annotations
import argparse, json, os, yaml, itertools
import numpy as np
from ..embed import make_embedder
from ..llm import make_llm
from ..store import MemoryStore, to_day
from ..translate import LLMTranslator, RuleTranslator, Translator
from .. import retrieve as R
from ..bench import locomo, longmemeval as LME
from ..bench.metrics import recall_at_k

class CachedTranslator(Translator):
    """Same (question, now) -> same extraction; the grid re-scores each item ~13 times but pays for one extraction."""
    def __init__(self, inner: Translator): self.inner, self.cache = inner, {}
    def translate(self, question, now):
        k = (question, now)
        if k not in self.cache: self.cache[k] = self.inner.translate(question, now)
        tq, cost = self.cache[k]
        return tq, dict(cost)

def score(items, tuned, translator):
    """items: dicts with question / evidence (gold ids) / now / store. Objective = mean recall@10 of rag+tq."""
    vals = []
    for it in items:
        w = {k: tuple(v) for k, v in tuned["weights"].items()}
        sysm = R.TQ(it["store"], translator, weights=w, width_scale=tuned["width_scale"], theta_sem=tuned["theta_sem"])
        hits, _, _ = sysm.retrieve(it["question"], it["now"])
        vals.append(recall_at_k([h["id"] for h in hits], it["evidence"]))
    return float(np.mean(vals)) if vals else 0.0

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/locomo10.json"); ap.add_argument("--embedder", default="hash")
    ap.add_argument("--llm", default="mock"); ap.add_argument("--translator", default="rule"); ap.add_argument("--out", default="configs/tuned.yaml")
    ap.add_argument("--dev_set", default="data/dev_tuning.jsonl", help="locomo fallback: jsonl of {conv_id, question, evidence}")
    ap.add_argument("--dataset", default="locomo", choices=["locomo", "lme"])
    ap.add_argument("--set", default="data/longmemeval_s.json", help="lme: longmemeval_s.json")
    ap.add_argument("--lme_split", default="data/lme_split.json", help="lme: split file with the temporal dev ids")
    a = ap.parse_args()
    emb = make_embedder(a.embedder); llm = make_llm(a.llm)
    translator = CachedTranslator(LLMTranslator(llm) if a.translator == "llm" else RuleTranslator())
    items = []
    if a.dataset == "lme":
        dev_ids = set(json.load(open(a.lme_split, encoding="utf-8"))["dev"])
        for it in LME.load(a.set, {"temporal-reasoning"}):
            if it.qid not in dev_ids: continue
            st = MemoryStore(emb); st.add_many(it.memories)
            items.append({"question": it.question, "evidence": it.gold_sessions, "now": it.now, "store": st})
        provisional = False; source = f"lme temporal dev {len(items)} (seed 42)"
    else:
        dev_items = [json.loads(l) for l in open(a.dev_set, encoding="utf-8") if l.strip()]
        by = {}
        for c in locomo.load(a.data, "dev"):
            st = MemoryStore(emb); st.add_many(c.memories); by[c.conv_id] = (c, st)
        for it in dev_items:
            c, st = by[it["conv_id"]]
            items.append({"question": it["question"], "evidence": it["evidence"], "now": c.now, "store": st})
        provisional = True; source = f"locomo dev fallback {len(items)} ({os.path.basename(a.dev_set)})"
    print(f"tuning on {source}", flush=True)
    tuned = {"weights": {"none": [1.0, 0.0], "recent": [0.7, 0.3], "range": [0.5, 0.5], "anchored": [0.4, 0.6]}, "width_scale": 1.0, "theta_sem": 0.5}
    log = []
    # tie-break: keep the incumbent (default) unless a candidate is strictly better
    best = (score(items, tuned, translator), tuned["width_scale"])
    for ws in [0.5, 1.0, 2.0]:
        t = dict(tuned, width_scale=ws); v = score(items, t, translator); log.append(("width_scale", ws, v))
        if v > best[0]: best = (v, ws)
    tuned["width_scale"] = best[1]
    for mode in ["anchored", "range", "recent"]:
        best = (score(items, tuned, translator), tuned["weights"][mode][1])
        for beta in [0.3, 0.5, 0.7]:
            w = dict(tuned["weights"]); w[mode] = [1 - beta, beta]; t = dict(tuned, weights=w); v = score(items, t, translator); log.append((f"beta[{mode}]", beta, v))
            if v > best[0]: best = (v, beta)
        tuned["weights"][mode] = [1 - best[1], best[1]]
    best = (score(items, tuned, translator), tuned["theta_sem"])
    for th in [0.3, 0.5, 0.7]:
        t = dict(tuned, theta_sem=th); v = score(items, t, translator); log.append(("theta_sem", th, v))
        if v > best[0]: best = (v, th)
    tuned["theta_sem"] = best[1]
    tuned["_log"] = [f"{k}={v}: {s:.4f}" for k, v, s in log]
    tuned["provisional"] = provisional; tuned["_source"] = source
    modes = {}
    for (q, now), (tq, _) in translator.cache.items(): modes[tq.mode] = modes.get(tq.mode, 0) + 1
    tuned["_dev_modes"] = modes
    yaml.safe_dump(tuned, open(a.out, "w", encoding="utf-8"), allow_unicode=True)
    print(yaml.safe_dump(tuned, allow_unicode=True))

if __name__ == "__main__":
    main()
