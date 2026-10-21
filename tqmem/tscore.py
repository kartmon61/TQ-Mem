"""Alternative temporal scores for the 4th-round ablations (2026-09-29). Each one replaces the kernel K in
score = alpha * s_sem + beta * K while everything else (extraction, candidates, normalisation, k) stays as in TQ-Mem v3.

  kernel  TQ-Mem: mode-specific continuous kernel (kernels.kernel)
  notype  ablation 1: no time-type distinction - one Gaussian at the condition's centre, its extent as the width
  tvec    ablation 4: the time value appended to the embedding (vector search on [e; w*t]), no kernel
  asso    V7b: AssoMem released code (scoring/temporal.py) - query-agnostic multi-scale recency decay, taus 3/90/365 days
  temb    V7a: AssoMem paper - cosine similarity between embeddings of the query's time condition and the memory's date text
"""
from __future__ import annotations
import datetime as _dt
import hashlib, json, math, os
import numpy as np
from . import kernels as KN

ASSO_TAUS = (3.0, 90.0, 365.0)          # facebookresearch/AssoMem scoring/temporal.py default decay constants (days)

def notype_kernel(mode, t, now, center, span, sigma, tau, halflife):
    """One Gaussian for every mode. centre: anchored = centre, range = midpoint, recent = now.
    width: anchored = sigma, range = max(half the span, tau), recent = the half-life."""
    if mode == "none": return np.zeros_like(t, dtype=np.float64)
    if mode == "anchored": c, s = center, sigma
    elif mode == "range":
        l, r = span; c, s = (l + r) / 2.0, max((r - l) / 2.0, tau)
    elif mode == "recent": c, s = now, halflife
    else: raise ValueError(mode)
    return np.exp(-((t - c) ** 2) / (2 * max(s, 1e-6) ** 2))

def asso_decay(t, now, taus=ASSO_TAUS):
    """AssoMem released temporal score: mean of exp(-age/tau); age < 0 -> 1.0 as in their code."""
    age = now - np.asarray(t, dtype=np.float64)
    out = np.mean([np.exp(-np.clip(age, 0, None) / tt) for tt in taus], axis=0)
    return np.where(age < 0, 1.0, out)

def query_time_point(mode, now, center, span):
    """Single day coordinate for the query's time condition (tvec / temb). None for mode none."""
    if mode == "anchored": return center
    if mode == "range": return (span[0] + span[1]) / 2.0
    if mode == "recent": return now
    return None

def tvec_scores(s_cos, t_mem, t_q, lo, hi, w=1.0):
    """Cosine of [e_m; w*tn_m] and [e_q; w*tn_q] with unit e_m, e_q and cos(e_m, e_q) = s_cos (time normalised to [0,1] over [lo, hi]).
    Same as TimeDim (E3 V6) but the query coordinate is the extracted time condition instead of `now`."""
    span = max(hi - lo, 1.0)
    tm = (np.asarray(t_mem, dtype=np.float64) - lo) / span; tq = (t_q - lo) / span
    return (s_cos + w * w * tm * tq) / np.sqrt((1 + (w * tm) ** 2) * (1 + (w * tq) ** 2))

# ---------------- V7a: time-text embeddings ----------------
def _day_to_date(d: float) -> _dt.date:
    from .store import from_day
    return _dt.date.fromisoformat(from_day(int(round(d))))

def date_text(d: float) -> str:
    """Memory side: the timestamp as text (AssoMem embeds the utterance's timestamp with its time tokens)."""
    x = _day_to_date(d); return x.strftime("%B %-d, %Y") if os.name != "nt" else x.strftime("%B %#d, %Y")

def query_time_text(mode, now, center, span) -> str | None:
    """Query side: the extracted time condition rendered as text (the query's temporal tokens)."""
    f = date_text
    if mode == "anchored": return f(center)
    if mode == "range": return f"from {f(span[0])} to {f(span[1])}"
    if mode == "recent": return f"recently, before {f(now)}"
    return None

class TimeTextEmbedder:
    """Embeds short time strings with a cache on disk. backend 'store' = the store's sentence embedder (bge-large);
    backend 'timellama' = mean-pooled last hidden state of chrisyuan45/TimeLlama-7b, read from a precomputed cache only
    (tqmem.eval.timellama_cache builds it offline)."""
    def __init__(self, backend: str = "store", embedder=None, cache_dir: str = ".temb_cache"):
        self.backend, self.embedder, self.cache_dir = backend, embedder, cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        import threading
        tag = backend if backend != "store" else "store_" + str(getattr(embedder, "model_name", type(embedder).__name__)).replace("/", "_")
        self.path = os.path.join(cache_dir, f"{tag}.jsonl")   # one cache per embedder (a hash-embedder test run must not feed a bge run)
        self.cache = {}; self.lock = threading.Lock()
        if os.path.exists(self.path):
            for line in open(self.path, encoding="utf-8"):
                r = json.loads(line); self.cache[r["text"]] = np.asarray(r["vec"], dtype=np.float32)
    def encode(self, texts):
        with self.lock: return self._encode(texts)
    def _encode(self, texts):
        miss = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if miss:
            if self.backend != "store":
                raise RuntimeError(f"{len(miss)} time strings missing from {self.path}; build it with tqmem.eval.timellama_cache first (e.g. {miss[0]!r})")
            V = self.embedder.encode(miss)
            with open(self.path, "a", encoding="utf-8") as f:
                for t, v in zip(miss, V):
                    v = np.asarray(v, dtype=np.float32); v = v / (np.linalg.norm(v) or 1.0); self.cache[t] = v
                    f.write(json.dumps({"text": t, "vec": [round(float(x), 6) for x in v]}) + "\n")
        return np.stack([self.cache[t] for t in texts])
    def sims(self, qtext: str, days) -> np.ndarray:
        texts = [date_text(d) for d in days]
        E = self.encode(texts); q = self.encode([qtext])[0]
        return E @ q

def minmax(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0: return x
    r = x.max() - x.min(); return (x - x.min()) / r if r > 0 else np.ones_like(x)

def temporal_score(kind, mode, days, nowd, c, sp, sigma, tau, h, s_cos=None, lo=None, hi=None, temb=None, w_tvec=1.0):
    """K in [0,1] for one query over candidate day coordinates `days` (tvec returns the combined vector score instead)."""
    if kind == "kernel": return KN.kernel(mode, days, nowd, c, sp, sigma, tau, h)
    if kind == "notype": return notype_kernel(mode, days, nowd, c, sp, sigma, tau, h)
    if kind == "asso": return asso_decay(days, nowd)
    if kind == "temb":
        return minmax(temb.sims(query_time_text(mode, nowd, c, sp), days))
    if kind == "tvec": return tvec_scores(s_cos, days, query_time_point(mode, nowd, c, sp), lo, hi, w_tvec)
    raise ValueError(kind)

# ---------------- V8: CMI weights (AssoMem 3.2.3) ----------------
def _mi_binned(xb: np.ndarray, y: np.ndarray) -> float:
    """Mutual information (nats) between a 3-bin score and a binary label."""
    n = len(y); mi = 0.0
    if n == 0: return 0.0
    for b in range(3):
        pb = np.mean(xb == b)
        if pb == 0: continue
        for v in (0, 1):
            pbv = np.mean((xb == b) & (y == v)); pv = np.mean(y == v)
            if pbv > 0: mi += pbv * math.log(pbv / (pb * pv))
    return mi

def cmi_weights(rows, T: float):
    """rows: dicts {mode, s (minmax semantic score), K, y (1 = candidate is gold evidence)} over training queries' candidates.
    Scores are binned into low/medium/high by global tertiles; CMI_d(type) = I(Y; bin_d | type) per time type (our mode);
    weights = softmax(CMI / T) over {semantic, temporal}. Returns ({mode: (alpha, beta)}, {mode: (cmi_sem, cmi_time, n)})."""
    out, diag = {"none": (1.0, 0.0)}, {}
    if not rows: return out, diag
    S = np.array([r["s"] for r in rows]); K = np.array([r["K"] for r in rows])
    qs, qk = np.quantile(S, [1 / 3, 2 / 3]), np.quantile(K, [1 / 3, 2 / 3])
    for m in ("recent", "range", "anchored"):
        idx = [i for i, r in enumerate(rows) if r["mode"] == m]
        if not idx: continue
        y = np.array([rows[i]["y"] for i in idx]); sb = np.digitize(S[idx], qs); kb = np.digitize(K[idx], qk)
        cs, ck = _mi_binned(sb, y), _mi_binned(kb, y)
        z = np.exp(np.array([cs, ck]) / T); z = z / z.sum()
        out[m] = (float(z[0]), float(z[1])); diag[m] = (cs, ck, len(idx))
    return out, diag
