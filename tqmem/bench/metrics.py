"""Token-F1 (LoCoMo normalization), recall@k, judge prompt."""
from __future__ import annotations
import re, string
from collections import Counter

def normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())

def token_f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g: return float(p == g)
    common = Counter(p) & Counter(g); n = sum(common.values())
    if n == 0: return 0.0
    pr, rc = n / len(p), n / len(g)
    return 2 * pr * rc / (pr + rc)

def recall_at_k(hit_ids: list[str], gold_ids: list[str]) -> float:
    return float(any(g in set(hit_ids) for g in gold_ids)) if gold_ids else 0.0

JUDGE_SYSTEM = "You are a strict but fair grader."
JUDGE_USER = """Question: {question}
Gold answer: {gold}
Predicted answer: {pred}
Is the predicted answer correct given the gold answer? Consider paraphrases and partial dates (e.g., "May 2023" for "8 May 2023") as correct if they contain the essential information.
Reply with exactly one word: CORRECT or WRONG."""

ANSWER_SYSTEM = "You answer questions about a past conversation using retrieved memory snippets. Be concise."
ANSWER_USER = """The conversation is between {a} and {b}. Today's date is {now}. Below are retrieved memory snippets, each with its date.
Answer the question in a short phrase. If the answer cannot be determined from the snippets, say "Not mentioned".

Memories:
{memories}

Question: {question}
Answer:"""

def format_memories(hits: list[dict], now_day: float) -> str:
    from ..store import to_day
    lines = []
    for h in hits:
        ago = int(round(now_day - to_day(h["date"])))
        lines.append(f"[{h['date']}, {ago} days ago] {h['text']}")
    return "\n".join(lines) if lines else "(no memories retrieved)"
