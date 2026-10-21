"""Step 1: translate a natural-language question into a structured temporal query."""
from __future__ import annotations
import json, re, calendar
from dataclasses import dataclass, asdict
from datetime import date, timedelta
from .store import to_day, from_day
from .llm import LLM

MODES = {"recent", "anchored", "range", "none"}
WIDTHS = {"exact", "narrow", "medium", "wide"}

@dataclass
class TemporalQuery:
    query: str
    role: str = "constraint"        # constraint | target
    mode: str = "none"              # recent | anchored | range | none
    center: str | None = None       # YYYY-MM-DD
    span: tuple[str, str] | None = None
    width: str = "medium"
    parse_failed: bool = False
    aggregate: bool = False         # E3 'ag': counting/enumeration question, kernel switched off (run.py _AggregateGuard)
    def to_dict(self): return asdict(self)

SYSTEM_PROMPT_EN = """You convert a user's memory-search question into a structured temporal query.
Today's date is {now}.

Output ONLY a JSON object with this schema:
{{"query": string,
 "temporal": {{"role": "constraint"|"target",
              "mode": "recent"|"anchored"|"range"|"none",
              "center": "YYYY-MM-DD"|null,
              "span": ["YYYY-MM-DD","YYYY-MM-DD"]|null,
              "width": "exact"|"narrow"|"medium"|"wide"}}}}

Rules:
1. "query" is the question with all time expressions removed, keeping only what to search for.
2. role = "target" if the question asks WHEN something happened (when / how long ago / what date). Otherwise "constraint".
3. mode = "none" if there is no time expression, or if role = "target".
4. mode = "recent" for lately, recently, these days, nowadays.
5. mode = "anchored" for a single point in time (a date, a month, "around March"). Put the resolved date in "center" (middle of the period for months/seasons/years).
6. mode = "range" for a period (a season, "all of last year", "during the trip", "between X and Y"). Put start and end in "span".
7. width: "exact" for an exact date; "narrow" for early/mid/late-month or a specific week; "medium" for around/sometime/about a month; "wide" for a season, a year, or vague words.
8. Resolve relative expressions (last year, two months ago) against today's date.

Examples:
Q: What have I been reading lately?
A: {{"query":"what I have been reading","temporal":{{"role":"constraint","mode":"recent","center":null,"span":null,"width":"medium"}}}}
Q: Where did I go around March last year?
A: {{"query":"where I went","temporal":{{"role":"constraint","mode":"anchored","center":"{ly}-03-15","span":null,"width":"medium"}}}}
Q: What did I do all last summer?
A: {{"query":"what I did","temporal":{{"role":"constraint","mode":"range","center":null,"span":["{ly}-06-01","{ly}-08-31"],"width":"narrow"}}}}
Q: What's my blood type?
A: {{"query":"my blood type","temporal":{{"role":"constraint","mode":"none","center":null,"span":null,"width":"exact"}}}}
Q: When did I move to the new apartment?
A: {{"query":"moved to the new apartment","temporal":{{"role":"target","mode":"none","center":null,"span":null,"width":"exact"}}}}
Q: What did Caroline eat on 24 October 2023?
A: {{"query":"what Caroline ate","temporal":{{"role":"constraint","mode":"anchored","center":"2023-10-24","span":null,"width":"exact"}}}}
"""

def _fallback(question: str) -> TemporalQuery:
    return TemporalQuery(query=question, role="constraint", mode="none", width="medium", parse_failed=True)

OPEN_START, OPEN_END = "0001-01-01", "9999-12-31"   # sentinels for an open-ended range span (see parse_json)

def parse_json(text: str, question: str) -> TemporalQuery:
    t = text.strip()
    t = re.sub(r"^```(?:json)?", "", t).strip(); t = re.sub(r"```$", "", t).strip()
    m = re.search(r"\{.*\}", t, re.S)
    if not m: return _fallback(question)
    try:
        obj = json.loads(m.group(0))
        tp = obj.get("temporal", {})
        role = tp.get("role", "constraint"); mode = tp.get("mode", "none"); width = tp.get("width", "medium")
        if role not in ("constraint", "target") or mode not in MODES or width not in WIDTHS: return _fallback(question)
        if role == "target": mode = "none"
        span = tp.get("span"); span = tuple(span) if span else None
        if span is not None and len(span) == 2:
            # open interval: null / "null" / "" on one side ("before today" -> [null, today], "since 2022" -> [2022-01-01, null]).
            # A sentinel far date keeps the range kernel sig((t-l)/tau)*sig((r-t)/tau) one-sided (the far factor is 1) and the
            # hard / candidate windows unbounded on that side; no consumer needs a special case. Both sides open -> mode none.
            lo, hi = [None if (x is None or str(x).strip().lower() in ("", "null", "none")) else str(x) for x in span]
            if lo is None and hi is None: span = None
            else: span = (lo or OPEN_START, hi or OPEN_END)
        tq = TemporalQuery(query=obj.get("query") or question, role=role, mode=mode, center=tp.get("center"), span=span, width=width)
        if mode == "anchored" and not tq.center: return _fallback(question)
        if mode == "range" and (not span or len(span) != 2): return _fallback(question)
        return tq
    except Exception:
        return _fallback(question)

class Translator:
    def translate(self, question: str, now: str) -> tuple[TemporalQuery, dict]:
        """Returns (TemporalQuery, cost dict with llm_calls/prompt_tokens/completion_tokens/latency_ms)."""
        raise NotImplementedError

class LLMTranslator(Translator):
    def __init__(self, llm: LLM, retries: int = 1):
        self.llm = llm; self.retries = retries
    def translate(self, question, now):
        ly = str(int(now[:4]) - 1)
        sysm = SYSTEM_PROMPT_EN.format(now=now, ly=ly)
        cost = {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "latency_ms": 0.0}
        tq = _fallback(question)
        for _ in range(self.retries + 1):
            r = self.llm.complete(sysm, f"Q: {question}\nA:", max_tokens=200)
            cost["llm_calls"] += 1; cost["prompt_tokens"] += r.prompt_tokens
            cost["completion_tokens"] += r.completion_tokens; cost["latency_ms"] += r.latency_ms
            tq = parse_json(r.text, question)
            if not tq.parse_failed: break
        return tq, cost

_MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
_MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})

class RuleTranslator(Translator):
    """Regex translator for English, used for tests and as the gold-free ablation baseline (no LLM)."""
    def translate(self, question, now):
        q = question.strip(); ql = q.lower(); nowd = date.fromisoformat(now); cost = {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "latency_ms": 0.0}
        if re.match(r"^(when|how long|how many (days|weeks|months|years)|what (date|year|month|time)|which (year|month|day))\b", ql):
            return TemporalQuery(query=q, role="target", mode="none", width="exact"), cost
        if re.search(r"\b(lately|recently|these days|nowadays)\b", ql):
            return TemporalQuery(query=re.sub(r"\b(lately|recently|these days|nowadays)\b", "", q), role="constraint", mode="recent", width="medium"), cost
        # explicit day: "24 October 2023" / "October 24, 2023"
        m = re.search(r"\b(\d{1,2})\s+([a-z]+)\s+(\d{4})\b", ql) or re.search(r"\b([a-z]+)\s+(\d{1,2}),?\s+(\d{4})\b", ql)
        if m:
            g = m.groups(); 
            try:
                if g[0].isdigit(): d, mo, y = int(g[0]), _MONTHS.get(g[1]), int(g[2])
                else: mo, d, y = _MONTHS.get(g[0]), int(g[1]), int(g[2])
                if mo: return TemporalQuery(query=q.replace(m.group(0), ""), role="constraint", mode="anchored", center=date(y, mo, d).isoformat(), width="exact"), cost
            except ValueError: pass
        # range: "during/throughout/between ... month year"
        m = re.search(r"\b(during|throughout|between|in the (first|second|third|fourth) week of)\b.*?\b([a-z]+)\s*(\d{4})?", ql)
        m2 = re.search(r"\b([a-z]+)\s+(\d{4})\b", ql)
        if m2 and m2.group(1) in _MONTHS:
            mo, y = _MONTHS[m2.group(1)], int(m2.group(2)); last = calendar.monthrange(y, mo)[1]
            if re.search(r"\b(during|throughout|all of)\b", ql):
                return TemporalQuery(query=q.replace(m2.group(0), ""), role="constraint", mode="range", span=(date(y, mo, 1).isoformat(), date(y, mo, last).isoformat()), width="medium"), cost
            wk = re.search(r"\b(first|second|third|fourth|beginning|end|early|mid|late)\b", ql)
            width = "narrow" if wk else "medium"
            cd = {"first": 4, "beginning": 4, "early": 5, "second": 11, "third": 18, "mid": 15, "fourth": 25, "late": 25, "end": 27}.get(wk.group(1), 15) if wk else 15
            return TemporalQuery(query=q.replace(m2.group(0), ""), role="constraint", mode="anchored", center=date(y, mo, min(cd, last)).isoformat(), width=width), cost
        return TemporalQuery(query=q, role="constraint", mode="none", width="medium"), cost
