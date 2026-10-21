"""LongMemEval_S loader (schema per the official repo; data is HF-hosted and must be downloaded separately).
Each item has its own haystack -> one MemoryStore per question."""
from __future__ import annotations
import json
from dataclasses import dataclass
from datetime import datetime
from ..store import Memory, to_day

@dataclass
class LMEItem:
    qid: str; qtype: str; question: str; answer: str; now: str; memories: list[Memory]; gold_sessions: list[str]; abstention: bool

def _parse_date(s: str) -> str:
    s = s.strip()
    for fmt in ("%Y/%m/%d (%a) %H:%M", "%Y/%m/%d", "%Y-%m-%d", "%Y-%m-%d %H:%M"):
        try: return datetime.strptime(s, fmt).date().isoformat()
        except ValueError: pass
    return s[:10]

def load(path: str, types: set[str] | None = None) -> list[LMEItem]:
    out = []
    for it in json.load(open(path, encoding="utf-8")):
        if types and it["question_type"] not in types: continue
        mems = []
        for si, (sess, d, sid) in enumerate(zip(it["haystack_sessions"], it["haystack_dates"], it["haystack_session_ids"])):
            day = to_day(_parse_date(d))
            for ti, turn in enumerate(sess):
                mems.append(Memory(id=f"{sid}", text=f'{turn["role"]}: {turn["content"]}', day=day, meta={"session_id": sid, "turn": ti}))
        out.append(LMEItem(it["question_id"], it["question_type"], it["question"], str(it["answer"]), _parse_date(it["question_date"]),
                           mems, list(it["answer_session_ids"]), it["question_id"].endswith("_abs")))
    return out
