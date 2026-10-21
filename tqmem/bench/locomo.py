"""LoCoMo loader. Verified against locomo10.json (2026-09-08): 10 conversations, 1,986 QA.
category: 1=multi-hop(282) 2=temporal(321) 3=open-domain(96) 4=single-hop(841) 5=adversarial(446)."""
from __future__ import annotations
import json, re
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from ..store import Memory, to_day

CATEGORY = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
DEV_IDS = ["conv-26", "conv-30"]          # first two by sample_id order

@dataclass
class QA:
    conv_id: str; qid: str; question: str; answer: str; category: str; evidence: list[str]
    role_guess: str = "constraint"        # heuristic; replaced by human annotation when available

@dataclass
class Conversation:
    conv_id: str; speaker_a: str; speaker_b: str
    memories: list[Memory]; qas: list[QA]; now: str; span_days: int; n_sessions: int

_WHEN = re.compile(r"^(when|how long|how many (days|weeks|months|years)|what (date|year|month|time)|which (year|month|day))\b", re.I)

def parse_dt(s: str) -> datetime:
    return datetime.strptime(s.strip(), "%I:%M %p on %d %B, %Y")

def load(path: str, split: str | None = None) -> list[Conversation]:
    data = json.load(open(path, encoding="utf-8"))
    convs = []
    for c in data:
        cid = c["sample_id"]
        if split == "dev" and cid not in DEV_IDS: continue
        if split == "test" and cid in DEV_IDS: continue
        conv = c["conversation"]; mems = []; dates = []
        k = 1
        while f"session_{k}" in conv:
            d = parse_dt(conv[f"session_{k}_date_time"]).date(); dates.append(d)
            for t in conv[f"session_{k}"]:
                mems.append(Memory(id=t["dia_id"], text=f'{t["speaker"]}: {t["text"]}', day=to_day(d),
                                   meta={"session": k, "speaker": t["speaker"]}))
            k += 1
        now = (max(dates) + timedelta(days=1)).isoformat()
        qas = []
        for i, q in enumerate(c["qa"]):
            cat = CATEGORY[q["category"]]
            qas.append(QA(cid, f"{cid}-q{i}", q["question"], str(q.get("answer", "")), cat, list(q.get("evidence", [])),
                          role_guess=("target" if _WHEN.match(q["question"].strip()) else "constraint")))
        convs.append(Conversation(cid, conv["speaker_a"], conv["speaker_b"], mems, qas, now, (max(dates) - min(dates)).days, k - 1))
    return convs

def stats(convs):
    from collections import Counter
    out = {"n_conv": len(convs), "n_qa": sum(len(c.qas) for c in convs), "by_category": Counter(q.category for c in convs for q in c.qas),
           "temporal_role_guess": Counter(q.role_guess for c in convs for q in c.qas if q.category == "temporal"),
           "turns": {c.conv_id: len(c.memories) for c in convs}, "span_days": {c.conv_id: c.span_days for c in convs}}
    return out
