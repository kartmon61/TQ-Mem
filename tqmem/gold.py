"""Gold translator (E3 V5): returns human-annotated temporal queries from annotations/*.csv instead of calling an LLM.
CSV columns: question_id, question, now, role, mode, center, span_start, span_end, width, note"""
from __future__ import annotations
import csv
from .translate import Translator, TemporalQuery

class GoldTranslator(Translator):
    def __init__(self, csv_path: str, fallback: Translator):
        self.by_q = {}
        for r in csv.DictReader(open(csv_path, encoding="utf-8")):
            if not r.get("mode") or r["mode"] == "ambiguous": continue
            span = (r["span_start"], r["span_end"]) if r.get("span_start") else None
            self.by_q[r["question"].strip()] = TemporalQuery(query=r.get("query") or r["question"], role=r["role"], mode=r["mode"] if r["role"] != "target" else "none",
                                                              center=r.get("center") or None, span=span, width=r.get("width") or "medium")
        self.fallback = fallback
    def translate(self, question, now):
        tq = self.by_q.get(question.strip())
        cost = {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "latency_ms": 0.0}
        return (tq, cost) if tq else self.fallback.translate(question, now)
