"""E3 V6: scalar time_dim baseline. Appends a normalized time scalar to the embedding and does pure vector search.
This is the 'time inside the embedding' family (category 4) in its simplest form."""
from __future__ import annotations
import numpy as np
from .store import MemoryStore, to_day
from .retrieve import Retriever, _hit, _zero_cost

class TimeDim(Retriever):
    name = "rag+tq:v6"
    def __init__(self, store: MemoryStore, k=10, w=1.0):
        super().__init__(store, k); self.w = w
        d = store.days; self.dmin, self.dmax = d.min(), d.max()
        tnorm = (d - self.dmin) / max(self.dmax - self.dmin, 1.0)
        self.E2 = np.concatenate([store.E, (self.w * tnorm)[:, None]], axis=1).astype(np.float32)
        self.E2 /= np.linalg.norm(self.E2, axis=1, keepdims=True)
    def retrieve(self, question, now):
        q = self.store.embedder.encode([question])[0]
        # query time coordinate = now (the only time signal a query-agnostic scalar can use)
        qt = (to_day(now) - self.dmin) / max(self.dmax - self.dmin, 1.0)
        q2 = np.concatenate([q, [self.w * qt]]).astype(np.float32); q2 /= np.linalg.norm(q2)
        s = self.E2 @ q2; o = np.argsort(-s)[: self.k]
        return [_hit(self.store, i, s[i], s_sem=float(self.store.E[i] @ q)) for i in o], _zero_cost(), None
