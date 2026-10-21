"""Retrieval systems. Every system implements retrieve(question, now) -> (hits, cost, tq_dict).
hits: list of dicts {id, text, date, score, s_sem, K, alpha, beta} (top-k, best first)."""
from __future__ import annotations
import json, re
import numpy as np
from .store import MemoryStore, to_day, from_day
from .translate import Translator, TemporalQuery
from . import kernels as KN
from . import tscore as TS

def _hit(store, i, score, s_sem=None, K=None, alpha=None, beta=None):
    m = store.mems[i]
    return {"id": m.id, "text": m.text, "date": from_day(m.day), "score": float(score),
            "s_sem": None if s_sem is None else float(s_sem), "K": None if K is None else float(K),
            "alpha": alpha, "beta": beta}

def _zero_cost(): return {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "latency_ms": 0.0}

class Retriever:
    name = "base"
    def __init__(self, store: MemoryStore, k: int = 10): self.store, self.k = store, k
    def retrieve(self, question: str, now: str): raise NotImplementedError

# ---------------- baselines on the RAG store ----------------
class RAG(Retriever):
    """Semantic-only."""
    name = "rag"
    def retrieve(self, question, now):
        q = self.store.embedder.encode([question])[0]
        idx, s = self.store.top_sem(q, self.k)
        return [_hit(self.store, i, v, s_sem=v) for i, v in zip(idx, s)], _zero_cost(), None

def _minmax(x):
    x = np.asarray(x, dtype=np.float64); r = x.max() - x.min()
    return (x - x.min()) / r if r > 0 else np.zeros_like(x)

class GA(Retriever):
    """Generative-Agents-style global exponential recency: 0.995 ** hours."""
    name = "rag+ga"
    def __init__(self, store, k=10, w_r=1.0, pool=100): super().__init__(store, k); self.w_r, self.pool = w_r, pool
    def retrieve(self, question, now):
        q = self.store.embedder.encode([question])[0]
        idx, s = self.store.top_sem(q, min(self.pool, len(self.store)))
        hours = (to_day(now) - self.store.days[idx]) * 24.0
        rec = 0.995 ** np.clip(hours, 0, None)
        sc = _minmax(s) + self.w_r * _minmax(rec)
        o = np.argsort(-sc)[: self.k]
        return [_hit(self.store, idx[j], sc[j], s_sem=s[j], K=rec[j]) for j in o], _zero_cost(), None

class MB(Retriever):
    """MemoryBank-style Ebbinghaus decay exp(-days/S) with fixed strength."""
    name = "rag+mb"
    def __init__(self, store, k=10, S=30.0, w_e=1.0, pool=100): super().__init__(store, k); self.S, self.w_e, self.pool = S, w_e, pool
    def retrieve(self, question, now):
        q = self.store.embedder.encode([question])[0]
        idx, s = self.store.top_sem(q, min(self.pool, len(self.store)))
        days = np.clip(to_day(now) - self.store.days[idx], 0, None)
        dec = np.exp(-days / self.S)
        sc = _minmax(s) + self.w_e * _minmax(dec)
        o = np.argsort(-sc)[: self.k]
        return [_hit(self.store, idx[j], sc[j], s_sem=s[j], K=dec[j]) for j in o], _zero_cost(), None

class FullContext(Retriever):
    name = "fc"
    def retrieve(self, question, now):
        order = self.store._order
        return [_hit(self.store, i, 0.0) for i in order], _zero_cost(), None

# ---------------- systems that use temporal translation ----------------
def _resolve(tq: TemporalQuery, now: str, width_scale: float):
    sigma, tau, h = KN.scaled_width(tq.width, width_scale)
    center = to_day(tq.center) if tq.center else None
    span = (to_day(tq.span[0]), to_day(tq.span[1])) if tq.span else None
    if tq.mode == "range" and span and span[0] > span[1]: span = (span[1], span[0])
    return sigma, tau, h, center, span

class HF(Retriever):
    """TSM-style hard filter then semantic rank (translation reused, window = hard)."""
    name = "rag+hf"
    def __init__(self, store, translator: Translator, k=10, width_scale=1.0):
        super().__init__(store, k); self.tr, self.ws = translator, width_scale
    def retrieve(self, question, now):
        tq, cost = self.tr.translate(question, now)
        q = self.store.embedder.encode([tq.query or question])[0]
        if tq.role == "target" or tq.mode == "none":
            idx, s = self.store.top_sem(q, self.k)
            return [_hit(self.store, i, v, s_sem=v) for i, v in zip(idx, s)], cost, tq.to_dict()
        sigma, tau, h, c, sp = _resolve(tq, now, self.ws)
        lo, hi = KN.hard_window(tq.mode, to_day(now), c, sp, sigma, tau, h)
        cand = self.store.time_range(lo, hi)
        if len(cand) == 0: return [], cost, tq.to_dict()      # hard filter: no expansion
        idx, s = self.store.top_sem(q, min(self.k, len(cand)), cand)
        return [_hit(self.store, i, v, s_sem=v) for i, v in zip(idx, s)], cost, tq.to_dict()

class TQ(Retriever):
    """TQ-Mem: translation -> candidate window -> alpha*s_sem + beta*K (or multiplicative) -> top-k with breakdown."""
    name = "rag+tq"
    def __init__(self, store, translator: Translator, k=10, weights=None, width_scale=1.0, theta_sem=KN.THETA_SEM,
                 theta_mode="rel", combine="add", gamma=None, force_mode=None, force_width=None, use_window=True,
                 h_recent=None, sem_norm="raw", tscore="kernel", temb=None):
        """theta_mode: 'rel' -> keep s_sem >= theta * max(s_sem over candidates); 'abs' -> keep s_sem >= theta.
        h_recent: override the recent half-life (days) regardless of width (tunable; LoCoMo 'recently' answers are ~50 days old).
        sem_norm: 'raw' = cosine as is; 'minmax' = rescale s_sem to [0,1] over the candidates before mixing with K (Mem0TQ does this)."""
        super().__init__(store, k)
        self.tr = translator; self.weights = weights or KN.WEIGHT_TABLE; self.ws = width_scale; self.theta = theta_sem; self.theta_mode = theta_mode
        self.combine = combine; self.gamma = gamma or KN.GAMMA_TABLE
        self.force_mode, self.force_width, self.use_window = force_mode, force_width, use_window
        self.h_recent, self.sem_norm = h_recent, sem_norm
        self.tscore, self.temb = tscore, temb   # temporal score in place of the kernel (tscore.py; 4th-round ablations / AssoMem V7)
    def retrieve(self, question, now):
        tq, cost = self.tr.translate(question, now)
        if self.force_mode: tq.mode = self.force_mode; tq.role = "constraint"
        if self.force_width: tq.width = self.force_width
        q = self.store.embedder.encode([tq.query or question])[0]
        nowd = to_day(now)
        if tq.role == "target" or tq.mode == "none":
            idx, s = self.store.top_sem(q, self.k)
            return [_hit(self.store, i, v, s_sem=v, K=0.0, alpha=1.0, beta=0.0) for i, v in zip(idx, s)], cost, tq.to_dict()
        sigma, tau, h, c, sp = _resolve(tq, now, self.ws)
        if tq.mode == "recent" and self.h_recent: h = float(self.h_recent)
        # candidate reduction with up to 2 expansions
        cand = np.arange(len(self.store))
        if self.use_window:
            for e in range(3):
                lo, hi = KN.candidate_window(tq.mode, nowd, c, sp, sigma, tau, h, expand=e)
                cand = self.store.time_range(lo, hi)
                if len(cand) >= self.k: break
        if len(cand) == 0: return [], cost, tq.to_dict()
        s = self.store.sem_scores(q, cand); s_cos = s
        if self.sem_norm == "minmax" and s.size:
            rng = s.max() - s.min(); s = (s - s.min()) / rng if rng > 0 else np.ones_like(s)
        days = self.store.days[cand]
        K = TS.temporal_score(self.tscore, tq.mode, days, nowd, c, sp, sigma, tau, h, s_cos=s_cos,
                              lo=float(self.store.days.min()), hi=float(self.store.days.max()), temb=self.temb)
        alpha, beta = self.weights[tq.mode]
        if self.tscore == "tvec": sc = K                      # plain vector search on [e; w*t]: the time value replaces the kernel and the mix
        elif self.combine == "add": sc = alpha * s + beta * K
        else: sc = s * np.power(K, self.gamma[tq.mode])
        if self.theta is None: keep = np.ones_like(s, dtype=bool)
        elif self.theta_mode == "rel": keep = s >= self.theta * (s.max() if s.size else 0.0)
        else: keep = s >= self.theta
        sc = np.where(keep, sc, -1.0)
        o = np.argsort(-sc)[: self.k]; o = [j for j in o if sc[j] >= 0]
        return [_hit(self.store, cand[j], sc[j], s_sem=s[j], K=K[j], alpha=alpha, beta=beta) for j in o], cost, tq.to_dict()

class CL(Retriever):
    """Chronos-style iterative tool-call loop (simplified): LLM issues up to `max_rounds` date-ranged searches."""
    name = "rag+cl"
    SYS = ("You search a memory of past conversations. Today is {now}. You may call the tool up to {n} times by replying ONLY with JSON: "
           "{{\"action\":\"search\",\"query\":\"...\",\"date_from\":\"YYYY-MM-DD\"|null,\"date_to\":\"YYYY-MM-DD\"|null}}. "
           "When you have enough evidence reply ONLY with {{\"action\":\"done\"}}.")
    def __init__(self, store, llm, k=10, max_rounds=3): super().__init__(store, k); self.llm, self.max_rounds = llm, max_rounds
    def retrieve(self, question, now):
        cost = _zero_cost(); hits = []; transcript = f"Question: {question}\n"
        for r in range(self.max_rounds):
            res = self.llm.complete(self.SYS.format(now=now, n=self.max_rounds), transcript, max_tokens=120)
            cost["llm_calls"] += 1; cost["prompt_tokens"] += res.prompt_tokens; cost["completion_tokens"] += res.completion_tokens; cost["latency_ms"] += res.latency_ms
            m = re.search(r"\{.*\}", res.text, re.S)
            try: act = json.loads(m.group(0)) if m else {"action": "search", "query": question, "date_from": None, "date_to": None}
            except Exception: act = {"action": "search", "query": question, "date_from": None, "date_to": None}
            if act.get("action") == "done" and hits: break
            qtxt = act.get("query") or question
            q = self.store.embedder.encode([qtxt])[0]
            cand = None
            if act.get("date_from") or act.get("date_to"):
                lo = to_day(act["date_from"]) if act.get("date_from") else -1e9
                hi = to_day(act["date_to"]) if act.get("date_to") else 1e9
                cand = self.store.time_range(lo, hi)
                if len(cand) == 0: cand = None
            idx, s = self.store.top_sem(q, self.k, cand)
            hits = [_hit(self.store, i, v, s_sem=v) for i, v in zip(idx, s)]
            transcript += f"\nRound {r+1} results:\n" + "\n".join(f"[{h['date']}] {h['text'][:120]}" for h in hits) + "\n"
        return hits, cost, None
