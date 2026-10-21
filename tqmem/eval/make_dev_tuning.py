"""Draft constraint-type tuning questions for dev conversations by inverting LoCoMo 'when' questions.
Writes a CSV the author edits into data/dev_tuning.jsonl (target: 40 items, ~10 per mode).
  python -m tqmem.eval.make_dev_tuning --out annotations/dev_tuning_draft.csv"""
import argparse, csv
from ..bench import locomo
from ..store import from_day

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/locomo10.json"); ap.add_argument("--out", default="annotations/dev_tuning_draft.csv"); a = ap.parse_args()
    convs = locomo.load(a.data, "dev"); rows = []
    for c in convs:
        byid = {m.id: m for m in c.memories}
        for q in c.qas:
            if q.category != "temporal" or q.role_guess != "target": continue
            ev = [byid[e] for e in q.evidence if e in byid]
            if not ev: continue
            d = from_day(ev[0].day)
            rows.append({"conv_id": c.conv_id, "original_when_question": q.question, "original_answer": q.answer, "evidence": ";".join(q.evidence),
                         "session_date": d, "evidence_text": ev[0].text[:160],
                         "constraint_question (EDIT)": "", "answer (EDIT)": "", "gold_mode (anchored/range/recent)": "", "gold_center": d, "gold_span_start": "", "gold_span_end": "", "gold_width (exact/narrow/medium/wide)": ""})
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} draft rows -> {a.out}")

if __name__ == "__main__":
    main()
