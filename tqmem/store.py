"""Memory store: texts, day coordinates, normalized embeddings; exact cosine search + sorted time index."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, datetime
import numpy as np
from .embed import Embedder

def to_day(d) -> float:
    if isinstance(d, datetime): d = d.date()
    if isinstance(d, date): return float(d.toordinal())
    return float(datetime.strptime(str(d)[:10], "%Y-%m-%d").toordinal())

def from_day(x: float) -> str:
    return date.fromordinal(int(round(x))).isoformat()

@dataclass
class Memory:
    id: str
    text: str
    day: float          # absolute day coordinate
    meta: dict = field(default_factory=dict)

class MemoryStore:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.mems: list[Memory] = []
        self.E: np.ndarray | None = None
        self.days: np.ndarray | None = None
        self._order: np.ndarray | None = None   # indices sorted by day
        self._sorted_days: np.ndarray | None = None

    def add_many(self, mems: list[Memory]):
        """Write path: one embedding per memory, no LLM call, no precomputed time score."""
        self.mems.extend(mems)
        self._build()

    def _build(self):
        self.E = self.embedder.encode([m.text for m in self.mems])
        self.days = np.array([m.day for m in self.mems], dtype=np.float64)
        self._order = np.argsort(self.days, kind="stable")
        self._sorted_days = self.days[self._order]

    def __len__(self): return len(self.mems)

    # --- semantic axis
    def sem_scores(self, q_vec: np.ndarray, idx: np.ndarray | None = None) -> np.ndarray:
        E = self.E if idx is None else self.E[idx]
        return np.clip(E @ q_vec, 0.0, None)   # cosine on normalized vectors, negatives clipped

    def top_sem(self, q_vec: np.ndarray, k: int, idx: np.ndarray | None = None):
        s = self.sem_scores(q_vec, idx)
        base = np.arange(len(self.mems)) if idx is None else idx
        o = np.argsort(-s)[:k]
        return base[o], s[o]

    # --- time axis
    def time_range(self, lo: float, hi: float) -> np.ndarray:
        """Indices of memories with lo <= day <= hi via binary search on the sorted time index."""
        a = np.searchsorted(self._sorted_days, lo, side="left")
        b = np.searchsorted(self._sorted_days, hi, side="right")
        return self._order[a:b]
