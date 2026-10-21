"""Run one system on LoCoMo (dev/test). Writes results/e1/{system}_{family}[_run{n}].jsonl

Example:
  python -m tqmem.eval.run --system rag+tq --split dev --embedder hash --llm mock --translator rule
  python -m tqmem.eval.run --system rag+tq --split test --embedder BAAI/bge-large-en-v1.5 --llm openai:gpt-4o-mini --translator llm --run 1
"""
from __future__ import annotations
import re
import argparse, json, os, sys, time, yaml, threading
from concurrent.futures import ThreadPoolExecutor
from ..embed import make_embedder
from ..llm import make_llm, LLMResult
from ..store import MemoryStore, to_day, from_day
from ..translate import LLMTranslator, RuleTranslator
from ..gold import GoldTranslator
from ..timedim import TimeDim
from ..bench import supp as SUPP, longmemeval as LME
from .. import retrieve as R
from ..bench import locomo
from ..bench.metrics import token_f1, recall_at_k, ANSWER_SYSTEM, ANSWER_USER, format_memories, JUDGE_SYSTEM, JUDGE_USER
from ..bench.judge_lme import judge_lme

class _FullQuestion:
    """E3 variant 'fq': search with the full question instead of the time-stripped `query` (the strip can drop semantic content —
    'later on the evening of 7 July' -> 'what Maria planned to do' pushed the gold memory from rank 1 to the bottom half, 2026-09-15)."""
    def __init__(self, inner): self.inner = inner
    def translate(self, question, now):
        tq, cost = self.inner.translate(question, now); tq.query = None; return tq, cost

class ReplayTranslator:
    """Reuse the time-condition extraction stored in an earlier result file (field `tq`, keyed by question_id) so that every
    4th-round variant ranks with exactly the same extraction as the TQ-Mem v3 reference run (no extraction noise, no LLM call).
    The reported retrieval cost is the stored one (the extraction is the same call the reference run paid for)."""
    def __init__(self, path: str):
        from ..translate import TemporalQuery
        self.TQ, self.recs, self.local = TemporalQuery, {}, threading.local()
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if r.get("tq") is not None and not r.get("error"): self.recs[r["question_id"]] = r
        self.path = path
    def set_qid(self, qid): self.local.qid = qid
    def timed(self, qid):
        """True if the reference extraction puts a time condition on this question (the only questions whose ranking can change)."""
        r = self.recs.get(qid); tq = (r or {}).get("tq") or {}
        return bool(r) and tq.get("role") != "target" and tq.get("mode", "none") != "none" and not tq.get("aggregate")
    def translate(self, question, now):
        qid = getattr(self.local, "qid", None); r = self.recs.get(qid)
        if r is None: raise KeyError(f"no reference extraction for {qid} in {self.path}")
        d = dict(r["tq"]); d["span"] = tuple(d["span"]) if d.get("span") else None
        tq = self.TQ(**{k: v for k, v in d.items() if k in self.TQ.__dataclass_fields__})
        return tq, {"llm_calls": r.get("retrieval_llm_calls", 0), "prompt_tokens": r.get("retrieval_prompt_tokens", 0),
                    "completion_tokens": r.get("retrieval_completion_tokens", 0), "latency_ms": 0.0}

_AGG = re.compile(r"^\s*(how many|how much|how often|what is the total|total number|in total|altogether|list all|what are all|name all|which of (the|my))\b", re.I)

class _AggregateGuard:
    """E3 variant 'ag': counting / enumeration questions aggregate evidence across sessions. Re-ranking by time inside the k=10 budget
    pushed one needed turn out on LongMemEval (5 judge losses, all "how many ..." multi-session questions, 2026-09-18).
    Rule-based on the question text, no change to the LLM extraction: such questions get mode none (pure semantic ranking) and
    tq.aggregate = True for analysis."""
    def __init__(self, inner): self.inner = inner
    def translate(self, question, now):
        tq, cost = self.inner.translate(question, now)
        if _AGG.search(question):
            tq.aggregate = True
            if tq.mode != "none": tq.mode, tq.center, tq.span = "none", None, None
        return tq, cost

ROUND4 = {"notype", "fixw", "mul", "tvec", "asso", "temb", "cmi"}   # 4th-round ablations + AssoMem comparisons (tscore.py)
TEMB = None   # TimeTextEmbedder for 'temb', set in main()

def build_system(name: str, store, translator, llm, tuned: dict):
    t = tuned or {}
    base, _, var = name.partition(":")
    # E3 mechanism variants, combinable with '.': nt = no semantic floor, fq = search with the full question, nowin = no candidate window
    # (rag only). e.g. rag+tq:nt.fq.nowin, mem0+tq:ntfq (legacy spelling of nt.fq). Named variants (v1, v3, v4, v4nt, v6, nowin) stay as before.
    toks = set(var.replace("ntfq", "nt.fq").split(".")) if var else set()
    if toks and toks <= {"nt", "fq", "nowin", "wr", "ag"} | ROUND4:
        # 4th round (2026-09-29): notype = one Gaussian for all types + one shared weight, fixw = per-type kernels + one shared weight,
        # mul = s * K^gamma, tvec = time value appended to the vector, asso = AssoMem released decay, temb = AssoMem time-text embedding,
        # cmi = per-type weights from conditional mutual information on gold-evidence labels (weights come in via `tuned`, see --cmi)
        r4 = dict(tscore="kernel", combine="add", temb=None)
        if toks & {"notype", "fixw"}:
            from .. import kernels as KN
            w0 = t.get("weights") or KN.WEIGHT_TABLE; bbar = sum(w0[m][1] for m in ("recent", "range", "anchored")) / 3.0
            t = dict(t, weights={"none": (1.0, 0.0)} | {m: (1.0 - bbar, bbar) for m in ("recent", "range", "anchored")})
        if "notype" in toks: r4["tscore"] = "notype"
        if "mul" in toks: r4["combine"] = "mul"
        for k in ("tvec", "asso", "temb"):
            if k in toks: r4["tscore"] = k
        if "temb" in toks: r4["temb"] = TEMB
        if "fq" in toks: translator = _FullQuestion(translator)
        if "ag" in toks: translator = _AggregateGuard(translator)   # aggregate-question guard (see class doc)
        th = None if "nt" in toks else t.get("theta_sem", 0.5)
        if "wr" in toks:   # weak recent: LoCoMo "recently" answers are ~50 days old (median) while the recent kernel half-life is 14 days -> (alpha, beta) = (0.9, 0.1)
            from .. import kernels as KN
            w = dict(t.get("weights") or KN.WEIGHT_TABLE); w["recent"] = (0.9, 0.1); t = dict(t, weights=w)
        if base == "mem0+tq":
            from ..mem0_adapter import Mem0TQ
            return Mem0TQ(store, translator, weights=t.get("weights"), width_scale=t.get("width_scale", 1.0), theta_sem=th, h_recent=t.get("h_recent"), **r4)
        if base == "rag+tq":
            return R.TQ(store, translator, weights=t.get("weights"), width_scale=t.get("width_scale", 1.0), theta_sem=th,
                        theta_mode=t.get("theta_mode", "rel"), use_window="nowin" not in toks, h_recent=t.get("h_recent"), sem_norm=t.get("sem_norm", "raw"), **r4)
    if name.startswith("mem0"):
        # store here is a Mem0Store (see --dataset handling); requires mem0ai + a running vector store
        from ..mem0_adapter import Mem0Retriever, Mem0TQ, Mem0HF, Mem0GA, Mem0CL
        if name == "mem0": return Mem0Retriever(store)
        if name == "mem0+ga": return Mem0GA(store, w_r=t.get("ga_w_r", 1.0))
        if name == "mem0+cl": return Mem0CL(store, llm)
        if name == "mem0+hf": return Mem0HF(store, translator, width_scale=t.get("width_scale", 1.0))
        if name == "mem0+tq": return Mem0TQ(store, translator, weights=t.get("weights"), width_scale=t.get("width_scale", 1.0), theta_sem=t.get("theta_sem", 0.5))
        if name == "mem0+tq:v3": return Mem0TQ(store, translator, weights={"none": (1.0, 0.0), "recent": (0.5, 0.5), "range": (0.5, 0.5), "anchored": (0.5, 0.5)}, width_scale=t.get("width_scale", 1.0), theta_sem=t.get("theta_sem", 0.5))
        raise ValueError(name)
    if name == "rag": return R.RAG(store)
    if name == "rag+ga": return R.GA(store, w_r=t.get("ga_w_r", 1.0))
    if name == "rag+mb": return R.MB(store, S=t.get("mb_S", 30.0), w_e=t.get("mb_w_e", 1.0))
    if name == "rag+hf": return R.HF(store, translator, width_scale=t.get("width_scale", 1.0))
    if name == "rag+cl": return R.CL(store, llm)
    if name == "fc": return R.FullContext(store)
    if name == "rag+tq:v6": return TimeDim(store)
    if name.startswith("rag+tq"):
        kw = dict(weights=t.get("weights"), width_scale=t.get("width_scale", 1.0), theta_sem=t.get("theta_sem", 0.5), theta_mode=t.get("theta_mode", "rel"))
        # E3 variants
        if name == "rag+tq:v1": kw.update(force_mode="recent", force_width="medium")
        if name == "rag+tq:v3": kw.update(weights={m: (0.5, 0.5) for m in ["none", "recent", "range", "anchored"]} | {"none": (1.0, 0.0)})
        if name == "rag+tq:v4": kw.update(combine="mul")
        if name == "rag+tq:v4nt": kw.update(combine="mul", theta_sem=None)
        if name == "rag+tq:nowin": kw.update(use_window=False)
        return R.TQ(store, translator, **kw)
    raise ValueError(name)

def answer(llm, conv, qa, hits, now):
    mem = format_memories(hits, to_day(now))
    if isinstance(hits, list) and len(hits) > 40:   # full context: cap for mock; real runs pass everything
        pass
    r = llm.complete(ANSWER_SYSTEM, ANSWER_USER.format(a=conv.speaker_a, b=conv.speaker_b, now=now, memories=mem, question=qa.question), max_tokens=64)
    return r

def judge(llm, qa, pred):
    r = llm.complete(JUDGE_SYSTEM, JUDGE_USER.format(question=qa.question, gold=qa.answer, pred=pred), max_tokens=16)   # GPT-5.x on Bedrock rejects max_output_tokens < 16
    return 1.0 if "CORRECT" in r.text.upper() and "WRONG" not in r.text.upper() else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="locomo", choices=["locomo", "supp", "lme"])
    ap.add_argument("--data", default="data/locomo10.json"); ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    ap.add_argument("--set", default=None, help="supp: jsonl path (typed/borderline). lme: longmemeval_s.json path")
    ap.add_argument("--lme_types", default=None, help="comma list of LongMemEval question_type to keep")
    ap.add_argument("--lme_split", default="data/lme_split.json", help="lme: split file from make_lme_split (dev / test_temporal / test_other id lists)")
    ap.add_argument("--lme_part", default=None, choices=["dev", "test_temporal", "test_other", "test"],
                    help="lme: keep only the ids of this part of --lme_split (test = test_temporal + test_other); also becomes the file-name tag lme-<part>")
    ap.add_argument("--gold_csv", default=None, help="annotations csv for --translator gold")
    ap.add_argument("--mem0_config", default="configs/mem0.yaml", help="mem0 config yaml (llm/embedder/vector_store)")
    ap.add_argument("--mem0_backend", default="mem0", choices=["mem0", "fake"], help="fake = in-memory stand-in for smoke tests")
    ap.add_argument("--mem0_reingest", action="store_true", help="wipe and re-add memories instead of reusing an earlier ingestion")
    ap.add_argument("--mem0_unit", default="turn", choices=["turn", "session"],
                    help="lme only: Mem0 add per turn (LoCoMo protocol) or per session (one add with the session's messages; ~10x fewer LLM calls)")
    ap.add_argument("--system", required=True); ap.add_argument("--embedder", default="hash")
    ap.add_argument("--llm", default="mock"); ap.add_argument("--judge_llm", default=None)
    ap.add_argument("--translator", default="rule", choices=["rule", "llm", "gold"]); ap.add_argument("--tuned", default=None)
    ap.add_argument("--run", type=int, default=1); ap.add_argument("--out", default="results/e1")
    ap.add_argument("--limit", type=int, default=0, help="cap QA per conversation (smoke tests)")
    ap.add_argument("--skip_adversarial", action="store_true", default=True)
    ap.add_argument("--no_judge", action="store_true")
    ap.add_argument("--workers", type=int, default=1, help="thread pool size: parallel Mem0 ingestion across conversations and parallel LLM calls across questions (records are written in the original order)")
    ap.add_argument("--no_resume", action="store_true", help="start the output file from scratch; default = keep existing records and only run the missing question_ids (record order is then run order, which the report does not depend on)")
    ap.add_argument("--redo_errors", action="store_true", help="when resuming, drop records that carry an `error` field and run those questions again")
    ap.add_argument("--tuned_cv", default=None, help="configs/tuned_cv.json from tune_cv: per-conversation parameters (leave-one-conversation-out); overrides --tuned for locomo")
    ap.add_argument("--fam", default=None, help="override the family tag used in result file names (e.g. to keep resuming the same file after switching the Bedrock routing profile us. -> global. of the same model)")
    ap.add_argument("--replay", default=None, help="result jsonl whose stored extraction (tq) is reused per question_id (4th-round variants; no extraction call)")
    ap.add_argument("--only_timed", action="store_true", help="with --replay: run only questions the reference extraction gives a time condition")
    ap.add_argument("--retrieval_only", action="store_true", help="no answer / judge calls: retrieval metrics only (prediction empty, f1 / judge null)")
    ap.add_argument("--time_embedder", default="store", choices=["store", "timellama"], help="temb: time-text embedder (store = the bge embedder)")
    ap.add_argument("--cmi", default=None, help="configs/cmi_weights.json from tqmem.eval.cmi: per-conversation (locomo) / pooled (lme) weights for ':cmi'")
    a = ap.parse_args()

    emb = make_embedder(a.embedder); llm = make_llm(a.llm); jllm = make_llm(a.judge_llm) if a.judge_llm else llm
    base_tr = LLMTranslator(llm) if a.translator in ("llm", "gold") else RuleTranslator()
    translator = GoldTranslator(a.gold_csv, base_tr) if a.translator == "gold" else base_tr
    replay = ReplayTranslator(a.replay) if a.replay else None
    if replay: translator = replay
    if a.only_timed and not replay: raise SystemExit("--only_timed needs --replay")
    global TEMB
    if ":" in a.system and "temb" in a.system.split(":", 1)[1].split("."):
        from ..tscore import TimeTextEmbedder
        TEMB = TimeTextEmbedder(a.time_embedder, emb, cache_dir=os.environ.get("TQMEM_TEMB_CACHE", ".temb_cache"))
    cmi = json.load(open(a.cmi, encoding="utf-8")) if a.cmi else None
    if a.system.endswith("cmi") and not cmi: raise SystemExit(":cmi needs --cmi configs/cmi_weights.json")
    tuned = yaml.safe_load(open(a.tuned, encoding="utf-8")) if a.tuned else {}
    tuned_cv = json.load(open(a.tuned_cv, encoding="utf-8"))["folds"] if a.tuned_cv else None
    def tuned_for(conv_id):
        """per-conversation params when --tuned_cv is given (each conversation gets the parameters tuned on the other seven)."""
        t = dict(tuned, **tuned_cv[conv_id]["params"]) if tuned_cv and conv_id in tuned_cv else dict(tuned)
        if cmi and a.system.endswith("cmi"):   # locomo: weights learned on the other seven conversations; lme: pooled over LoCoMo
            src = cmi["folds"].get(conv_id) if a.dataset == "locomo" else None
            t["weights"] = {m: tuple(v) for m, v in (src or cmi["pooled"])["weights"].items()}
        return t
    os.makedirs(a.out, exist_ok=True)
    fam = a.fam or a.llm.replace(":", "_").replace("/", "_").replace("@", "_")
    tag = a.split if a.dataset == "locomo" else (f"lme-{a.lme_part}" if a.dataset == "lme" and a.lme_part else os.path.splitext(os.path.basename(a.set))[0])
    sysname = (a.system.replace(':', '-') + ("-gold" if a.translator == "gold" else "") + ("-cv" if a.tuned_cv else "")
               + ("-tl" if a.time_embedder == "timellama" else "") + ("-ret" if a.retrieval_only else ""))
    path = os.path.join(a.out, f"{sysname}_{fam}_{tag}_run{a.run}.jsonl")
    n = 0
    workers = max(1, a.workers)
    done_ids = set()
    if not a.no_resume and os.path.exists(path):
        kept = []
        for line in open(path, encoding="utf-8"):
            try: rec = json.loads(line)
            except Exception: continue
            if a.redo_errors and rec.get("error"): continue
            kept.append(line); done_ids.add(rec["question_id"])
        if a.redo_errors:
            with open(path, "w", encoding="utf-8") as fw: fw.writelines(kept)
        if done_ids: print(f"[resume] {len(done_ids)} records already in {path}; skipping those question_ids" + (" (error records dropped for re-run)" if a.redo_errors else ""), flush=True)
    def one(sysm, llm, qtext, gold, gold_ids, now, meta, store, t_write):
        if replay: replay.set_qid(meta["question_id"])
        t1 = time.time(); hits, cost, tq = sysm.retrieve(qtext, now); t_ret = (time.time() - t1) * 1000
        class _Q: pass
        qa = _Q(); qa.question, qa.answer = qtext, gold
        class _C: pass
        cv = _C(); cv.speaker_a, cv.speaker_b = meta.get("speaker_a", "user"), meta.get("speaker_b", "assistant")
        ans = LLMResult("", 0, 0, 0.0) if a.retrieval_only else answer(llm, cv, qa, hits, now); pred = ans.text.strip()
        rec = dict(meta); rec.update({"system": a.system, "family": fam, "run": a.run, "question": qtext, "gold": gold, "prediction": pred,
                   "f1": None if a.retrieval_only else token_f1(pred, gold), "recall10": recall_at_k([h["id"] for h in hits[:10]], gold_ids),
                   "replayed_from": os.path.basename(a.replay) if a.replay else None,
                   "judge": None if (a.no_judge or a.retrieval_only) else (judge_lme(jllm, meta["category"], qtext, gold, pred, bool(meta.get("abstention"))) if a.dataset == "lme" else judge(jllm, qa, pred)), "tq": tq,
                   "mode": (tq or {}).get("mode"), "role": (tq or {}).get("role"),
                   "retrieval_llm_calls": cost["llm_calls"], "retrieval_prompt_tokens": cost["prompt_tokens"],
                   "retrieval_completion_tokens": cost["completion_tokens"], "retrieval_tokens": cost["prompt_tokens"] + cost["completion_tokens"],
                   "answer_prompt_tokens": ans.prompt_tokens, "answer_completion_tokens": ans.completion_tokens,
                   "latency_retrieval_ms": t_ret + cost["latency_ms"], "latency_answer_ms": ans.latency_ms, "write_ms": t_write, "n_memories": len(store),
                   "retrieved": [{k: h[k] for k in ("id", "date", "s_sem", "K", "score")} for h in hits[:10]]})
        return rec
    n_err = 0
    def guarded(t):
        """Run one task; a task that still fails after all provider retries becomes an error record (prediction "", metrics 0,
        `error` set) instead of aborting the whole run - one request that Luna keeps dropping must not cost 1,300 questions.
        Such records count as wrong; they carry an `error` field and can be re-run later with --no_resume."""
        nonlocal n_err
        try:
            return t()
        except Exception as e:
            n_err += 1; meta = getattr(t, "meta", {})
            print(f"[task error] {meta.get('question_id')}: {type(e).__name__}: {str(e)[:120]}", file=sys.stderr, flush=True)
            return dict(meta, system=a.system, family=fam, run=a.run, question=getattr(t, "question", ""), gold=getattr(t, "gold", ""),
                        prediction="", f1=0.0, recall10=0.0, judge=None if a.no_judge else 0.0, tq=None, mode=None, role=None,
                        retrieval_llm_calls=0, retrieval_prompt_tokens=0, retrieval_completion_tokens=0, retrieval_tokens=0,
                        answer_prompt_tokens=0, answer_completion_tokens=0, latency_retrieval_ms=0.0, latency_answer_ms=0.0,
                        write_ms=0.0, n_memories=0, retrieved=[], error=f"{type(e).__name__}: {str(e)[:200]}")
    def run_tasks(f, tasks):
        """tasks: zero-arg callables returning one record (with .meta/.question/.gold attributes used for error records).
        Run on `workers` threads, written in list order. Provider failures are absorbed by `guarded`."""
        nonlocal n
        if workers == 1 or len(tasks) <= 1:
            for t in tasks: f.write(json.dumps(guarded(t), ensure_ascii=False) + "\n"); f.flush(); n += 1
            return
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for rec in ex.map(guarded, tasks):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush(); n += 1   # flush per record so progress is visible in the file
    mem0_mem = None; ledger_dir = None; wlock = threading.Lock()
    if a.system.startswith("mem0"):
        from ..mem0_adapter import build_mem0, Mem0Store
        cfg = yaml.safe_load(open(a.mem0_config, encoding="utf-8")) if (a.mem0_backend == "mem0" and os.path.exists(a.mem0_config)) else None
        mem0_mem = build_mem0(cfg, backend=a.mem0_backend)
        # resume ledger lives inside the vector-store folder so wiping the store also wipes the ledger; the fake backend keeps no ledger
        vs_path = (((cfg or {}).get("vector_store") or {}).get("config") or {}).get("path", ".mem0_qdrant")
        ledger_dir = None if a.mem0_backend == "fake" else os.path.join(vs_path, "_ledger")
    def write_cost_line(rec: dict):
        with wlock, open(os.path.join(a.out, f"write_cost_{a.system.replace(':','-')}_{fam}.jsonl"), "a", encoding="utf-8") as wf:
            wf.write(json.dumps(rec) + "\n")
    with open(path, "a" if done_ids else "w", encoding="utf-8") as f:
        if a.dataset == "locomo":
            convs = locomo.load(a.data, None if a.split == "all" else a.split)
            def prepare(conv):
                """Build (or reuse) the store for one conversation. Mem0 ingestion is the slow, parallelizable part."""
                if mem0_mem is not None:
                    from ..mem0_adapter import Mem0Store
                    store = Mem0Store(mem0_mem, user_id=f"locomo_{conv.conv_id}", reuse=not a.mem0_reingest, ledger_dir=ledger_dir); t0 = time.time()
                    store.add_turns([(m.id, "user", m.text, from_day(m.day)) for m in conv.memories]); t_write = (time.time() - t0) * 1000
                    write_cost_line({"conv_id": conv.conv_id, **store.write_cost})
                    print(f"[ingest] {conv.conv_id}: n_memories={store.write_cost['n_memories']} reused={store.write_cost['reused']} "
                          f"resumed_from={store.write_cost['resumed_from']} {t_write/1000:.0f}s", flush=True)
                else:
                    store = MemoryStore(emb); t0 = time.time(); store.add_many(conv.memories); t_write = (time.time() - t0) * 1000
                return conv, store, t_write
            if mem0_mem is not None and workers > 1:
                with ThreadPoolExecutor(max_workers=min(workers, len(convs))) as ex: prepared = list(ex.map(prepare, convs))
            else:
                prepared = [prepare(c) for c in convs]
            tasks = []
            for conv, store, t_write in prepared:
                sysm = build_system(a.system, store, translator, llm, tuned_for(conv.conv_id))
                qas = [q for q in conv.qas if not (a.skip_adversarial and q.category == "adversarial")]
                if a.limit: qas = qas[: a.limit]
                for qa in qas:
                    if qa.qid in done_ids: continue
                    if a.only_timed and not replay.timed(qa.qid): continue
                    meta = {"exp": "e1", "dataset": "locomo", "conv_id": conv.conv_id, "question_id": qa.qid, "category": qa.category, "role_guess": qa.role_guess,
                            "speaker_a": conv.speaker_a, "speaker_b": conv.speaker_b}
                    t = (lambda sysm=sysm, qa=qa, conv=conv, meta=meta, store=store, t_write=t_write:
                         one(sysm, llm, qa.question, qa.answer, qa.evidence, conv.now, meta, store, t_write))
                    t.meta, t.question, t.gold = meta, qa.question, qa.answer; tasks.append(t)
            run_tasks(f, tasks)
        elif a.dataset == "supp":
            convs = {c.conv_id: c for c in locomo.load(a.data, "all")}; items = SUPP.load(a.set)
            if a.limit: items = items[: a.limit]
            stores = {}; tasks = []
            for it in items:
                conv = convs[it["conv_id"]]
                if it["conv_id"] not in stores:
                    st = MemoryStore(emb); st.add_many(conv.memories); stores[it["conv_id"]] = (st, build_system(a.system, st, translator, llm, tuned))
                st, sysm = stores[it["conv_id"]]
                meta = {"exp": "e2", "dataset": tag, "conv_id": it["conv_id"], "question_id": it.get("question_id") or f"{tag}-{len(tasks)}", "category": "temporal",
                        "role_guess": "constraint", "gold_mode": it.get("gold_mode"), "gold_width": it.get("gold_width"), "distance_days": it.get("distance_days"),
                        "speaker_a": conv.speaker_a, "speaker_b": conv.speaker_b}
                t = (lambda it=it, sysm=sysm, conv=conv, meta=meta, st=st: one(sysm, llm, it["question"], it["answer"], it["evidence"], conv.now, meta, st, 0.0))
                t.meta, t.question, t.gold = meta, it["question"], it["answer"]; tasks.append(t)
            run_tasks(f, tasks)
        else:  # lme
            types = set(a.lme_types.split(",")) if a.lme_types else None
            items = LME.load(a.set, types)
            if a.lme_part:   # ids fixed by data/lme_split.json (seed 42): temporal-reasoning dev 30 / test 103, every other type test
                sp = json.load(open(a.lme_split, encoding="utf-8"))
                keep = set(sp["test_temporal"]) | set(sp["test_other"]) if a.lme_part == "test" else set(sp[a.lme_part])
                items = [it for it in items if it.qid in keep]
            if a.limit: items = items[: a.limit]
            items = [it for it in items if it.qid not in done_ids]
            if a.only_timed: items = [it for it in items if replay.timed(it.qid)]
            def lme_one(it):
                if mem0_mem is not None:
                    # one Mem0 user per question (each item has its own haystack); turn ids "<session>#<turn>" for the ledger, session id as hit id
                    st = Mem0Store(mem0_mem, user_id=f"lme_{it.qid}", reuse=not a.mem0_reingest, ledger_dir=ledger_dir); t0 = time.time()
                    if a.mem0_unit == "session":
                        sess = {}   # session_id -> (messages in turn order, date); Memory.text is "<role>: <content>"
                        for m in it.memories:
                            role, _, content = m.text.partition(": ")
                            sess.setdefault(m.meta["session_id"], ([], from_day(m.day)))[0].append({"role": role if role in ("user", "assistant") else "user", "content": content or m.text})
                        st.add_sessions([(sid, msgs, d) for sid, (msgs, d) in sess.items()])
                    else:
                        st.add_turns([(f"{m.meta['session_id']}#{m.meta['turn']}", "user", m.text, from_day(m.day), m.meta["session_id"]) for m in it.memories])
                    t_write = (time.time() - t0) * 1000
                    write_cost_line({"conv_id": it.qid, **st.write_cost})
                    print(f"[ingest] {it.qid}: n_memories={st.write_cost['n_memories']} reused={st.write_cost['reused']} resumed_from={st.write_cost['resumed_from']} "
                          f"llm_calls={st.write_cost['llm_calls']} {t_write/1000:.0f}s", flush=True)
                else:
                    st = MemoryStore(emb); t0 = time.time(); st.add_many(it.memories); t_write = (time.time() - t0) * 1000
                sysm = build_system(a.system, st, translator, llm, tuned)
                meta = {"exp": "e2", "dataset": "lme", "conv_id": it.qid, "question_id": it.qid, "category": it.qtype, "role_guess": "constraint", "abstention": it.abstention}
                return one(sysm, llm, it.question, it.answer, it.gold_sessions, it.now, meta, st, t_write)
            lme_tasks = []
            for it in items:
                t = (lambda it=it: lme_one(it))
                t.meta = {"exp": "e2", "dataset": "lme", "conv_id": it.qid, "question_id": it.qid, "category": it.qtype, "role_guess": "constraint", "abstention": it.abstention}
                t.question, t.gold = it.question, it.answer; lme_tasks.append(t)
            run_tasks(f, lme_tasks)
    print(f"wrote {n} records -> {path}" + (f" (+{len(done_ids)} kept from an earlier run)" if done_ids else "") + (f" | {n_err} error records" if n_err else ""))

if __name__ == "__main__":
    main()
