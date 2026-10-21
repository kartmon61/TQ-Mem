"""Long-horizon stress set: scale session gaps by k and shift absolute dates in questions."""
from __future__ import annotations
import re, copy, calendar
from datetime import date, timedelta
from ..store import to_day, from_day
from .locomo import Conversation

_MONTH_YEAR = re.compile(r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b")
_DAY_MONTH_YEAR = re.compile(r"\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b")

def stretch(conv: Conversation, k: float) -> Conversation:
    c = copy.deepcopy(conv)
    d1 = min(m.day for m in c.memories)
    def shift(day): return d1 + k * (day - d1)
    for m in c.memories: m.day = shift(m.day)
    c.now = from_day(max(m.day for m in c.memories) + 1)
    c.span_days = int(max(m.day for m in c.memories) - d1)
    def repl_dmy(m):
        d = date(int(m.group(3)), list(calendar.month_name).index(m.group(2)), int(m.group(1)))
        nd = date.fromordinal(int(round(shift(to_day(d))))); return nd.strftime("%-d %B %Y")
    def repl_my(m):
        d = date(int(m.group(2)), list(calendar.month_name).index(m.group(1)), 15)
        nd = date.fromordinal(int(round(shift(to_day(d))))); return nd.strftime("%B %Y")
    for q in c.qas:
        q.question = _DAY_MONTH_YEAR.sub(repl_dmy, q.question)
        q.question = _MONTH_YEAR.sub(repl_my, q.question)
    return c
