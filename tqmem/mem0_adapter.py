"""Mem0 store (fact-level memories) and Mem0+TQ rerank. Requires `pip install mem0ai` and a running vector store.
Design decisions (thesis, Sec. 3.7-3.8):
- session date is passed as metadata["date"]; Mem0's created_at (ingestion time) is NOT used.
- Mem0+TQ = search(limit=100) -> min-max normalize Mem0 score as s_sem -> temporal kernel on metadata date -> alpha*s+beta*K.
"""
from __future__ import annotations
import json, os, time, threading
import numpy as np
from .store import to_day, from_day
from .translate import Translator
from . import kernels as KN
from . import tscore as TS

_OBS = {"date": None}   # session date of the turn being ingested; read by the patched prompt builder below

def _install_observation_date_patch():
    """mem0 2.x's extraction prompt has an 'Observation Date' section ("resolve ALL relative references against it") but the OSS
    `Memory.add` never fills it, so it defaults to today's date and 'last night' in a 2023 session becomes a 2026 date. We wrap the
    prompt builder so that the session date set by Mem0Store.add_turns is used as observation_date (current_date stays today)."""
    import mem0.memory.main as mm
    if getattr(mm, "_tqmem_obs_patched", False): return
    orig = mm.generate_additive_extraction_prompt
    def patched(*a, **kw):
        if kw.get("timestamp") is None and _OBS["date"]: kw["timestamp"] = _OBS["date"]   # timestamp -> observation_date in _resolve_dates
        return orig(*a, **kw)
    mm.generate_additive_extraction_prompt = patched; mm._tqmem_obs_patched = True

def make_thread_safe(memory):
    """Share one Memory across worker threads: serialize the local vector store (qdrant local has no intra-process lock) and the
    embedder (one GPU model); the slow part - the extraction LLM call inside add() - stays parallel. mem0's SQLite history already locks."""
    if getattr(memory, "_tqmem_locked", False): return memory
    lock = threading.RLock()
    def wrap(obj, names):
        for n in names:
            fn = getattr(obj, n, None)
            if callable(fn):
                def locked(*a, _fn=fn, **kw):
                    with lock: return _fn(*a, **kw)
                setattr(obj, n, locked)
    vs = memory.vector_store
    wrap(vs, [n for n in dir(vs) if not n.startswith("_") and callable(getattr(vs, n, None))])
    if getattr(memory, "embedding_model", None) is not None: wrap(memory.embedding_model, ["embed"])
    memory._tqmem_locked = True
    return memory

def build_mem0(cfg: dict | None = None, backend: str = "mem0"):
    """cfg: dict with keys llm, embedder, vector_store (mem0 config format). backend='fake' returns an in-memory stand-in for smoke tests."""
    if backend == "fake":
        return FakeMemory()
    from mem0 import Memory
    _install_observation_date_patch()
    # OpenAI-compatible path on Bedrock (bedrock-runtime /openai/v1 or bedrock-mantle) needs a Bedrock API key as the OpenAI key.
    # mem0's aws_bedrock provider cannot be used for OpenAI models (its provider list has no "openai"), so we stay on the openai provider
    # and inject the key: config api_key -> OPENAI_API_KEY env -> AWS_BEARER_TOKEN_BEDROCK -> short-term token from boto3 credentials.
    import os
    llm_cfg = ((cfg or {}).get("llm") or {}).get("config") or {}
    gw_url = os.environ.get("LLM_GATEWAY_URL", "").rstrip("/")
    gateway = bool(gw_url) and str(llm_cfg.get("openai_base_url", "")).rstrip("/").startswith(gw_url)
    if gateway and not llm_cfg.get("api_key"):
        # OpenAI-compatible LLM gateway: the key from .env (LLM_GATEWAY_KEY) is the OpenAI key for mem0's openai provider
        from .llm import gateway_api_key
        llm_cfg["api_key"] = gateway_api_key()
    if gateway and "gpt-5" in str(llm_cfg.get("model", "")) and "reasoning_effort" not in llm_cfg:
        llm_cfg["reasoning_effort"] = "none"   # mem0 forwards it (_get_supported_params); Luna would otherwise reason on every extraction/update call
    elif "bedrock" in str(llm_cfg.get("openai_base_url", "")) and not llm_cfg.get("api_key") and not os.environ.get("OPENAI_API_KEY"):
        from .llm import bedrock_bearer_token
        region = next((seg for seg in llm_cfg["openai_base_url"].split(".") if seg.startswith(("us-", "eu-", "ap-"))), "us-east-1")
        tok = bedrock_bearer_token(region)
        if not tok:
            raise RuntimeError("Bedrock API key not available: set AWS_BEARER_TOKEN_BEDROCK (long-term key) or valid boto3 credentials")
        llm_cfg["api_key"] = tok
    mem = Memory.from_config(cfg)
    if gateway:
        # optional gateway-specific headers (e.g. prompt caching for mem0's fixed prompt prefixes), as JSON in LLM_GATEWAY_HEADERS
        import json
        headers = json.loads(os.environ.get("LLM_GATEWAY_HEADERS") or "{}")
        if headers:
            mem.llm.client = mem.llm.client.with_options(default_headers=headers)
    return make_thread_safe(mem)

class FakeMemory:
    """Minimal stand-in with the subset of the mem0 API we use. Stores each turn as one 'fact'; search = hashed cosine."""
    def __init__(self):
        from .embed import HashEmbedder
        import numpy as np
        self.emb = HashEmbedder(); self.np = np; self.db = {}; self._lock = threading.Lock()
    def add(self, messages, user_id, metadata=None):
        text = "\n".join(m["content"] for m in messages); vec = self.emb.encode([text])[0]   # one fake memory per add (turn or session)
        with self._lock: self.db.setdefault(user_id, []).append({"id": f"{user_id}-{len(self.db.get(user_id, []))}", "memory": text, "vec": vec, "metadata": metadata or {}})
        return {"results": [{"event": "ADD"}]}
    def get_all(self, *, filters=None, top_k=20, **kw):
        user_id = (filters or {}).get("user_id")
        return {"results": [{"id": x["id"], "memory": x["memory"], "metadata": x["metadata"]} for x in self.db.get(user_id, [])][:top_k]}
    def delete_all(self, user_id=None, **kw):
        self.db.pop(user_id, None)
    def search(self, query, *, top_k=20, filters=None, threshold=0.1, **kw):
        items = self.db.get((filters or {}).get("user_id"), [])
        if not items: return {"results": []}
        q = self.emb.encode([query])[0]; E = self.np.stack([x["vec"] for x in items]); sc = E @ q
        o = self.np.argsort(-sc)[:top_k]
        return {"results": [{"id": items[i]["id"], "memory": items[i]["memory"], "score": float(sc[i]), "metadata": items[i]["metadata"]} for i in o]}

class Mem0Store:
    def __init__(self, memory, user_id: str, reuse: bool = True, ledger_dir: str | None = ".mem0_ledger"):
        """reuse=True: turns already ingested for user_id (per the ledger) are skipped, so an interrupted ingestion resumes at the
        first missing turn. Set False to wipe and re-ingest. The ledger (one line per ingested dia_id, appended after each add)
        is the source of truth: counting dia_ids inside Mem0 is unreliable because later turns can merge/delete earlier memories.
        ledger_dir=None disables the ledger (old behaviour: any existing memory => skip the whole conversation)."""
        self.m, self.user_id, self.reuse = memory, user_id, reuse
        self.ledger = os.path.join(ledger_dir, f"{user_id}.jsonl") if ledger_dir else None
        self.write_cost = {"llm_calls": 0, "tokens": 0, "n_memories": 0, "write_ms": 0.0, "reused": False, "resumed_from": 0}
    def _all(self) -> list:
        return self.m.get_all(filters={"user_id": self.user_id}, top_k=10**6).get("results", [])
    def _ledger_ids(self) -> list:
        if not self.ledger or not os.path.exists(self.ledger): return []
        return [json.loads(l)["dia_id"] for l in open(self.ledger, encoding="utf-8") if l.strip()]
    def add_turns(self, turns):
        """turns: list of (dia_id, role, text, date_iso[, session_id]). One add per turn keeps Mem0's ADD/UPDATE/DELETE logic intact
        (LoCoMo protocol, 2026-09-14)."""
        return self._add_units([(t[0], [{"role": t[1], "content": t[2]}], t[3], t[4] if len(t) > 4 else None) for t in turns])
    def add_sessions(self, sessions):
        """sessions: list of (session_id, messages[{role, content}], date_iso). One add per session = one extraction call over the whole
        session (mem0's intended `add(messages)` use). ~10x fewer LLM calls than per-turn adds on LongMemEval (2026-09-18); the ledger
        then holds one line per session id, so a run cannot mix the two units for the same user."""
        return self._add_units([(sid, msgs, d, sid) for sid, msgs, d in sessions])
    def _add_units(self, units):
        """units: (unit_id, messages, date_iso, session_id|None). Resume from the ledger; metadata date/dia_id(/sid) on every memory."""
        existing = self._all()
        if existing and not self.reuse:
            self.m.delete_all(user_id=self.user_id)
            if self.ledger and os.path.exists(self.ledger): os.remove(self.ledger)
            existing = []
        done = set(self._ledger_ids()) if (self.reuse and self.ledger) else set()
        if existing and self.reuse and not self.ledger:      # no ledger: legacy conversation-level reuse
            self.write_cost["n_memories"] = len(existing); self.write_cost["reused"] = True; return
        todo = [u for u in units if u[0] not in done]
        if not todo:
            self.write_cost["n_memories"] = len(existing); self.write_cost["reused"] = True; return
        self.write_cost["resumed_from"] = len(units) - len(todo)
        if self.ledger: os.makedirs(os.path.dirname(self.ledger), exist_ok=True)
        t0 = time.time()
        for unit_id, messages, d, sid in todo:
            md = {"date": d, "dia_id": unit_id}
            if sid: md["sid"] = sid   # LongMemEval: session id (gold is session-level) -> becomes the hit id in _hits_from
            _OBS["date"] = d   # session date -> mem0 'Observation Date' (see _install_observation_date_patch)
            self.m.add(messages, user_id=self.user_id, metadata=md)
            self.write_cost["llm_calls"] += 1   # Mem0 makes >=1 extraction call per add; refine with provider usage hooks if available
            if self.ledger:
                with open(self.ledger, "a", encoding="utf-8") as lf: lf.write(json.dumps({"dia_id": unit_id, "date": d}) + "\n")
        self.write_cost["write_ms"] = (time.time() - t0) * 1000
        self.write_cost["n_memories"] = len(self._all())
    def __len__(self): return self.write_cost["n_memories"]
    def search(self, query: str, limit: int):
        # threshold=0.0: keep the full top-`limit` pool (mem0 default 0.1 would silently drop low-similarity candidates that TQ may rescore)
        r = self.m.search(query, top_k=limit, filters={"user_id": self.user_id}, threshold=0.0)
        return r.get("results", r)

def _hits_from(results, scores=None):
    out = []
    for i, x in enumerate(results):
        md = x.get("metadata") or {}
        out.append({"id": md.get("sid") or md.get("dia_id") or x.get("id"), "text": x.get("memory") or x.get("text", ""), "date": md.get("date"),
                    "score": float(scores[i]) if scores is not None else float(x.get("score", 0.0)), "s_sem": float(x.get("score", 0.0)), "K": None, "alpha": None, "beta": None})
    return out

class Mem0Retriever:
    name = "mem0"
    def __init__(self, store: Mem0Store, k=10): self.st, self.k = store, k
    def retrieve(self, question, now):
        return _hits_from(self.st.search(question, self.k)), {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "latency_ms": 0.0}, None

class Mem0TQ:
    name = "mem0+tq"
    def __init__(self, store: Mem0Store, translator: Translator, k=10, pool=100, weights=None, width_scale=1.0, theta_sem=KN.THETA_SEM, theta_mode="rel", h_recent=None,
                 combine="add", tscore="kernel", temb=None):
        self.st, self.tr, self.k, self.pool = store, translator, k, pool
        self.combine, self.tscore, self.temb = combine, tscore, temb   # 4th-round ablations / AssoMem V7 (tscore.py)
        self.h_recent = h_recent   # override the recent half-life (days); tuned by tune_cv
        self.weights = weights or KN.WEIGHT_TABLE; self.ws, self.theta, self.theta_mode = width_scale, theta_sem, theta_mode
    def retrieve(self, question, now):
        tq, cost = self.tr.translate(question, now)
        res = self.st.search(tq.query or question, self.pool)
        if not res: return [], cost, tq.to_dict()
        raw = np.array([float(x.get("score", 0.0)) for x in res]); rng = raw.max() - raw.min()
        s = (raw - raw.min()) / rng if rng > 0 else np.ones_like(raw)
        if tq.role == "target" or tq.mode == "none":
            o = np.argsort(-s)[: self.k]; return _hits_from([res[j] for j in o], s[o]), cost, tq.to_dict()
        days = np.array([to_day((x.get("metadata") or {}).get("date", now)) for x in res]); nowd = to_day(now)
        sigma, tau, h = KN.scaled_width(tq.width, self.ws)
        if tq.mode == "recent" and self.h_recent: h = float(self.h_recent)
        c = to_day(tq.center) if tq.center else None; sp = (to_day(tq.span[0]), to_day(tq.span[1])) if tq.span else None
        # tvec: Mem0 scores are cosine similarities (qdrant), time normalised over the 100-candidate pool
        K = TS.temporal_score(self.tscore, tq.mode, days, nowd, c, sp, sigma, tau, h, s_cos=raw,
                              lo=float(days.min()), hi=float(days.max()), temb=self.temb); a, b = self.weights[tq.mode]
        if self.tscore == "tvec": sc = K
        elif self.combine == "add": sc = a * s + b * K
        else: sc = s * np.power(K, KN.GAMMA_TABLE[tq.mode])
        keep = np.ones_like(s, dtype=bool) if self.theta is None else (s >= (self.theta * s.max() if self.theta_mode == "rel" else self.theta))
        sc = np.where(keep, sc, -1.0); o = [j for j in np.argsort(-sc)[: self.k] if sc[j] >= 0]
        hits = _hits_from([res[j] for j in o], sc[o])
        for n, j in enumerate(o): hits[n].update({"s_sem": float(s[j]), "K": float(K[j]), "alpha": a, "beta": b})
        return hits, cost, tq.to_dict()


class Mem0GA:
    """Mem0 + Generative-Agents-style global recency on the same top-100 pool (E1 1차 row missing in round 3): 0.995 ** hours,
    min-max(score) + w_r * min-max(recency) - the same formula as rag+ga."""
    name = "mem0+ga"
    def __init__(self, store: Mem0Store, k=10, pool=100, w_r=1.0): self.st, self.k, self.pool, self.w_r = store, k, pool, w_r
    def retrieve(self, question, now):
        cost = {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "latency_ms": 0.0}
        res = self.st.search(question, self.pool)
        if not res: return [], cost, None
        raw = np.array([float(x.get("score", 0.0)) for x in res])
        days = np.array([to_day((x.get("metadata") or {}).get("date", now)) for x in res])
        rec = 0.995 ** np.clip((to_day(now) - days) * 24.0, 0, None)
        sc = TS.minmax(raw) + self.w_r * TS.minmax(rec); o = np.argsort(-sc)[: self.k]
        hits = _hits_from([res[j] for j in o], sc[o])
        for n, j in enumerate(o): hits[n]["K"] = float(rec[j])
        return hits, cost, None

class Mem0CL:
    """Mem0 + Chronos-style iterative tool calls (same prompt and loop as rag+cl): each round the LLM issues a query with an optional
    date range; the range filters Mem0's top-100 pool by metadata date. Last round's top-k is returned, as in rag+cl."""
    name = "mem0+cl"
    def __init__(self, store: Mem0Store, llm, k=10, pool=100, max_rounds=3): self.st, self.llm, self.k, self.pool, self.max_rounds = store, llm, k, pool, max_rounds
    def retrieve(self, question, now):
        import re as _re
        from .retrieve import CL
        cost = {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "latency_ms": 0.0}; hits = []; transcript = f"Question: {question}\n"
        for r in range(self.max_rounds):
            res = self.llm.complete(CL.SYS.format(now=now, n=self.max_rounds), transcript, max_tokens=120)
            cost["llm_calls"] += 1; cost["prompt_tokens"] += res.prompt_tokens; cost["completion_tokens"] += res.completion_tokens; cost["latency_ms"] += res.latency_ms
            m = _re.search(r"\{.*\}", res.text, _re.S)
            try: act = json.loads(m.group(0)) if m else {"action": "search", "query": question}
            except Exception: act = {"action": "search", "query": question}
            if act.get("action") == "done" and hits: break
            pool = self.st.search(act.get("query") or question, self.pool)
            if act.get("date_from") or act.get("date_to"):
                lo = to_day(act["date_from"]) if act.get("date_from") else -1e9; hi = to_day(act["date_to"]) if act.get("date_to") else 1e9
                inside = [x for x in pool if lo <= to_day((x.get("metadata") or {}).get("date", now)) <= hi]
                if inside: pool = inside
            hits = _hits_from(pool[: self.k])
            transcript += f"\nRound {r+1} results:\n" + "\n".join(f"[{h['date']}] {h['text'][:120]}" for h in hits) + "\n"
        return hits, cost, None

class Mem0HF:
    """Mem0 + TSM-style hard filter on the same top-100 pool (E1 1차, E3 V2 on Mem0)."""
    name = "mem0+hf"
    def __init__(self, store: Mem0Store, translator: Translator, k=10, pool=100, width_scale=1.0):
        self.st, self.tr, self.k, self.pool, self.ws = store, translator, k, pool, width_scale
    def retrieve(self, question, now):
        tq, cost = self.tr.translate(question, now)
        res = self.st.search(tq.query or question, self.pool)
        if not res: return [], cost, tq.to_dict()
        raw = np.array([float(x.get("score", 0.0)) for x in res])
        if tq.role == "target" or tq.mode == "none":
            o = np.argsort(-raw)[: self.k]; return _hits_from([res[j] for j in o], raw[o]), cost, tq.to_dict()
        days = np.array([to_day((x.get("metadata") or {}).get("date", now)) for x in res]); nowd = to_day(now)
        sigma, tau, h = KN.scaled_width(tq.width, self.ws)
        c = to_day(tq.center) if tq.center else None; sp = (to_day(tq.span[0]), to_day(tq.span[1])) if tq.span else None
        lo, hi = KN.hard_window(tq.mode, nowd, c, sp, sigma, tau, h)
        keep = (days >= lo) & (days <= hi)
        if not keep.any(): return [], cost, tq.to_dict()
        sc = np.where(keep, raw, -1e9); o = [j for j in np.argsort(-sc)[: self.k] if keep[j]]
        return _hits_from([res[j] for j in o], raw[o]), cost, tq.to_dict()
