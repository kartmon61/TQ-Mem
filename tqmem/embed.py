"""Embedding backends. All return L2-normalized float32 arrays of shape (n, d)."""
from __future__ import annotations
import hashlib, os, re, threading
import numpy as np

CACHE_DIR = os.environ.get("TQMEM_EMB_CACHE", ".emb_cache")   # "" disables
CACHE_MIN = 32                                                # only whole haystacks / conversations are cached, never single queries

def cached_encode(cache_dir: str, model_name: str, texts: list[str], compute) -> np.ndarray:
    """Return compute(texts) with an on-disk .npy cache keyed by the exact text list (sha1 of the texts in order).
    Same embedder + same texts -> identical vectors, so the cache cannot change any result; it only saves the GPU pass
    (LongMemEval_S: ~500 turns x 470 items x 3 RAG systems ~ 2 h per system on an RTX 3060, 2026-09-18)."""
    if not cache_dir or len(texts) < CACHE_MIN:
        return compute(texts)
    h = hashlib.sha1(model_name.encode("utf-8"))
    for t in texts: h.update(b"\0"); h.update(t.encode("utf-8"))
    d = os.path.join(cache_dir, model_name.replace("/", "_")); key = os.path.join(d, h.hexdigest() + ".npy")
    if os.path.exists(key):
        v = np.load(key)
        if v.shape[0] == len(texts): return v
    v = compute(texts)
    os.makedirs(d, exist_ok=True)
    tmp = f"{key}.{os.getpid()}.{threading.get_ident()}.tmp.npy"
    np.save(tmp, v); os.replace(tmp, key)   # atomic publish: parallel workers / processes never read a half-written file
    return v

class Embedder:
    dim: int
    def encode(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

class HashEmbedder(Embedder):
    """Deterministic bag-of-words hashing embedder for pipeline tests (no model download)."""
    def __init__(self, dim: int = 512):
        self.dim = dim
    def encode(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in re.findall(r"[a-z0-9가-힣]+", t.lower()):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
        n = np.linalg.norm(out, axis=1, keepdims=True); n[n == 0] = 1
        return out / n

class SentenceTransformerEmbedder(Embedder):
    """bge-large-en-v1.5 (default) via sentence-transformers. Fixed for all systems (2026-09-14 decision)."""
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5", device: str | None = None, batch_size: int = 64, cache_dir: str | None = None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_embedding_dimension() if hasattr(self.model, "get_embedding_dimension") else self.model.get_sentence_embedding_dimension()
        self.batch_size = batch_size
        self.model_name = model_name
        self.cache_dir = CACHE_DIR if cache_dir is None else cache_dir
        self._lock = threading.Lock()   # one model shared by worker threads (--workers): serialize GPU calls
    def _encode(self, texts):
        with self._lock:
            v = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True, convert_to_numpy=True)
        return v.astype(np.float32)
    def encode(self, texts):
        return cached_encode(self.cache_dir, self.model_name, list(texts), self._encode)

def make_embedder(name: str, **kw) -> Embedder:
    if name == "hash":
        return HashEmbedder(**kw)
    return SentenceTransformerEmbedder(model_name=name, **kw)
