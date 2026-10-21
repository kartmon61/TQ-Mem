"""Supplementary sets (typed 200, borderline 60, dev tuning) share one JSONL schema:
{"conv_id", "question", "answer", "evidence": [dia_ids], "gold_mode", "gold_center"|"gold_span", "gold_width", "set": "typed|borderline|dev"}
Borderline generator: turns a 'when' question with a known evidence date into a question whose time expression sits
sigma..2sigma (anchored) or 1..3 days outside a month range (range) away from the evidence date. Drafts need human wording."""
from __future__ import annotations
import json, csv, random, calendar
from datetime import date, timedelta
from .locomo import Conversation
from ..store import from_day

def load(path: str):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

def draft_borderline(convs: list[Conversation], out_csv: str, n_each: int = 30, seed: int = 42):
    rng = random.Random(seed); rows = []
    for c in convs:
        byid = {m.id: m for m in c.memories}
        for q in c.qas:
            if q.category != "temporal" or q.role_guess != "target": continue
            ev = [byid[e] for e in q.evidence if e in byid]
            if not ev: continue
            d = date.fromordinal(int(ev[0].day))
            # anchored: center 15 days away (medium sigma=10 -> 1.5 sigma), phrase 'around <mid-month>'
            c_anch = d + timedelta(days=rng.choice([-18, -15, 15, 18]))
            # range: previous or next calendar month relative to d if d is within 3 days of a month edge
            if d.day <= 3: r_start = (d.replace(day=1) - timedelta(days=1)).replace(day=1); r_end = d.replace(day=1) - timedelta(days=1)
            elif d.day >= calendar.monthrange(d.year, d.month)[1] - 2:
                nm = (d.replace(day=28) + timedelta(days=4)).replace(day=1); r_start, r_end = nm, nm.replace(day=calendar.monthrange(nm.year, nm.month)[1])
            else: r_start = r_end = None
            rows.append({"conv_id": c.conv_id, "original_when_question": q.question, "original_answer": q.answer, "evidence": ";".join(q.evidence),
                         "evidence_date": d.isoformat(), "evidence_text": ev[0].text[:160],
                         "anchored_center_suggest": c_anch.isoformat(), "anchored_question (EDIT: 'around <date>')": "",
                         "range_span_suggest": f"{r_start}..{r_end}" if r_start else "", "range_question (EDIT: 'during <month>')": "",
                         "answer (EDIT)": "", "distance_days": ""})
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    return len(rows)
