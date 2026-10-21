"""End-to-end smoke tests on real LoCoMo dev data without any model download or API key."""
import json, subprocess, sys, os
import numpy as np
from tqmem.embed import HashEmbedder
from tqmem.store import MemoryStore, to_day, from_day
from tqmem.translate import RuleTranslator, parse_json
from tqmem import kernels as KN
from tqmem import retrieve as R
from tqmem.bench import locomo
from tqmem.bench.metrics import token_f1, recall_at_k

DATA = "data/locomo10.json"

def test_loader_counts():
    cs = locomo.load(DATA, "all"); s = locomo.stats(cs)
    assert s["n_conv"] == 10 and s["n_qa"] == 1986
    assert s["by_category"]["temporal"] == 321 and s["by_category"]["adversarial"] == 446

def test_kernels_shapes_and_limits():
    t = np.array([0.0, 5.0, 10.0, 30.0]); now = 60.0
    K = KN.kernel("anchored", t, now, 10.0, None, 10.0, 7.0, 14.0); assert abs(K[2] - 1.0) < 1e-9 and K[3] < K[2]
    K = KN.kernel("range", t, now, None, (0.0, 10.0), 10.0, 7.0, 14.0); assert abs(K[1] - K[1]) < 1e-9 and K[3] < K[1]
    K = KN.kernel("recent", t, now, None, None, 10.0, 7.0, 14.0); assert K[3] > K[0]
    # tau -> 0 approaches hard filter
    K = KN.kernel("range", np.array([5.0, 10.5]), now, None, (0.0, 10.0), 10.0, 1e-6, 14.0); assert K[0] > 0.99 and K[1] < 0.01

def test_parse_json_and_fallback():
    tq = parse_json('{"query":"x","temporal":{"role":"constraint","mode":"anchored","center":"2023-05-07","span":null,"width":"exact"}}', "q")
    assert tq.mode == "anchored" and tq.center == "2023-05-07" and not tq.parse_failed
    assert parse_json("garbage", "q").parse_failed and parse_json("garbage", "q").mode == "none"

def test_retrievers_run_on_dev():
    cs = locomo.load(DATA, "dev"); c = cs[0]
    st = MemoryStore(HashEmbedder()); st.add_many(c.memories); tr = RuleTranslator()
    for sysm in [R.RAG(st), R.GA(st), R.MB(st), R.HF(st, tr), R.TQ(st, tr), R.TQ(st, tr, combine="mul"), R.FullContext(st)]:
        hits, cost, tq = sysm.retrieve("What did Caroline attend in early May 2023?", c.now)
        assert isinstance(hits, list) and "llm_calls" in cost
    hits, _, tq = R.TQ(st, tr).retrieve("What did Caroline attend in early May 2023?", c.now)
    assert tq["mode"] == "anchored" and all(h["K"] is not None for h in hits)

def test_time_index_window():
    cs = locomo.load(DATA, "dev"); c = cs[0]; st = MemoryStore(HashEmbedder()); st.add_many(c.memories)
    lo, hi = to_day("2023-05-01"), to_day("2023-05-10"); idx = st.time_range(lo, hi)
    assert all(lo <= st.days[i] <= hi for i in idx) and len(idx) > 0

def test_metrics():
    assert token_f1("7 May 2023", "7 May 2023") == 1.0 and token_f1("sushi", "pizza") == 0.0
    assert recall_at_k(["D1:3", "D2:1"], ["D2:1"]) == 1.0

def test_cli_run_and_report(tmp_path):
    out = str(tmp_path)
    r = subprocess.run([sys.executable, "-m", "tqmem.eval.run", "--system", "rag+tq", "--split", "dev", "--embedder", "hash", "--llm", "mock", "--translator", "rule", "--no_judge", "--limit", "5", "--out", out], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    files = [f for f in os.listdir(out) if f.endswith(".jsonl")]; assert files
    rec = json.loads(open(os.path.join(out, files[0])).readline())
    for k in ("f1", "recall10", "retrieval_llm_calls", "retrieved", "tq"): assert k in rec

def test_supp_gold_timedim_e4(tmp_path):
    out = str(tmp_path)
    for args in [["--system", "rag+tq", "--dataset", "supp", "--set", "data/dev_tuning.jsonl"],
                 ["--system", "rag+tq:v6", "--split", "dev", "--limit", "3"],
                 ["--system", "rag+tq", "--split", "dev", "--limit", "3", "--translator", "gold", "--gold_csv", "annotations/gold_sample.csv"]]:
        r = subprocess.run([sys.executable, "-m", "tqmem.eval.run", "--embedder", "hash", "--llm", "mock", "--no_judge", "--out", out] + (args if "--translator" in args else args + ["--translator", "rule"]), capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
    r = subprocess.run([sys.executable, "-m", "tqmem.eval.e4", "--set", "data/e4_sample.jsonl", "--llm", "mock", "--translator", "rule", "--out", os.path.join(out, "e4.jsonl")], capture_output=True, text=True)
    assert r.returncode == 0 and '"mode_acc": 1.0' in r.stdout

def test_lme_loader_schema(tmp_path):
    from tqmem.bench import longmemeval as LME
    item = {"question_id": "q1_abs", "question_type": "temporal-reasoning", "question": "When?", "answer": "x", "question_date": "2023/05/20 (Sat) 10:00",
            "haystack_dates": ["2023/05/01 (Mon) 09:00", "2023/05/10 (Wed) 09:00"], "haystack_session_ids": ["s1", "s2"],
            "haystack_sessions": [[{"role": "user", "content": "a"}], [{"role": "assistant", "content": "b"}]], "answer_session_ids": ["s2"]}
    p = tmp_path / "lme.json"; p.write_text(json.dumps([item]))
    items = LME.load(str(p)); assert items[0].abstention and len(items[0].memories) == 2 and items[0].now == "2023-05-20"

def test_borderline_draft(tmp_path):
    from tqmem.bench import supp
    cs = locomo.load(DATA, "dev"); n = supp.draft_borderline(cs, str(tmp_path / "b.csv")); assert n > 0

def test_mem0_fake_path(tmp_path):
    out = str(tmp_path)
    for sysn in ["mem0", "mem0+hf", "mem0+tq"]:
        r = subprocess.run([sys.executable, "-m", "tqmem.eval.run", "--system", sysn, "--split", "dev", "--embedder", "hash", "--llm", "mock", "--translator", "rule", "--no_judge", "--limit", "5", "--mem0_backend", "fake", "--out", out], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
    assert any(f.startswith("write_cost_mem0") for f in os.listdir(out))

def test_annotate_rule(tmp_path):
    r = subprocess.run([sys.executable, "-m", "tqmem.eval.annotate", "--split", "dev", "--llm", "mock", "--out", str(tmp_path / "a.csv"), "--review_out", str(tmp_path / "r.csv")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr and "temporal questions" in r.stdout


def test_mem0_ledger_resume(tmp_path):
    """Interrupted ingestion resumes at the first missing turn; nothing is re-added; --mem0_reingest wipes ledger and store."""
    from tqmem.mem0_adapter import FakeMemory, Mem0Store
    turns = [(f"D{i}", "user", f"turn {i} text", "2023-05-0%d" % (i % 9 + 1)) for i in range(6)]
    mem = FakeMemory(); led = str(tmp_path / "ledger")
    s1 = Mem0Store(mem, "u1", ledger_dir=led); s1.add_turns(turns[:4])          # "interrupted" after 4 turns
    assert s1.write_cost["llm_calls"] == 4 and len(mem.get_all(filters={"user_id": "u1"}, top_k=10**6)["results"]) == 4
    s2 = Mem0Store(mem, "u1", ledger_dir=led); s2.add_turns(turns)              # resume: only 2 remaining turns added
    assert s2.write_cost["llm_calls"] == 2 and s2.write_cost["resumed_from"] == 4 and len(mem.get_all(filters={"user_id": "u1"}, top_k=10**6)["results"]) == 6
    s3 = Mem0Store(mem, "u1", ledger_dir=led); s3.add_turns(turns)              # complete: reused, no adds
    assert s3.write_cost["reused"] and s3.write_cost["llm_calls"] == 0
    s4 = Mem0Store(mem, "u1", reuse=False, ledger_dir=led); s4.add_turns(turns)  # reingest wipes and re-adds all
    assert s4.write_cost["llm_calls"] == 6 and len(mem.get_all(filters={"user_id": "u1"}, top_k=10**6)["results"]) == 6


def test_make_llm_gateway(monkeypatch):
    """gateway:<model> -> OpenAI-compatible client on LLM_GATEWAY_URL/v1 with the key from the environment; no key -> clear error."""
    from tqmem import llm as L
    monkeypatch.delenv("LLM_GATEWAY_KEY", raising=False); monkeypatch.delenv("LLM_GATEWAY_API_KEY", raising=False)
    monkeypatch.setenv("LLM_GATEWAY_URL", "https://gateway.example.test/")
    import pytest
    with pytest.raises(RuntimeError, match="LLM_GATEWAY_KEY"):
        L.make_llm("gateway:gpt-5.6-luna")
    monkeypatch.setenv("LLM_GATEWAY_KEY", "vk-test")
    g = L.make_llm("gateway:gpt-5.6-luna")
    assert isinstance(g, L.OpenAICompatLLM) and str(g.client.base_url).rstrip("/") == "https://gateway.example.test/v1" and g.client.api_key == "vk-test"
    assert g._kwargs(64, 0.0) == {"max_completion_tokens": 64, "reasoning_effort": "none"}          # GPT-5.x: no temperature, effort none
    assert g._kwargs(4, 0.0)["max_completion_tokens"] == 16
    h = L.make_llm("gateway:claude-haiku-4-5-20251001")
    assert h._kwargs(64, 0.0) == {"temperature": 0.0, "max_tokens": 64}                              # Claude: temperature 0


def test_judge_lme_prompts():
    from tqmem.bench.judge_lme import anscheck_prompt, judge_lme
    from tqmem.llm import MockLLM
    p = anscheck_prompt("temporal-reasoning", "Q?", "18 days", "19 days")
    assert "off-by-one" in p and p.endswith("Answer yes or no only.") and "Correct Answer: 18 days" in p
    assert "Rubric:" in anscheck_prompt("single-session-preference", "Q?", "rubric", "resp")
    assert "updated answer" in anscheck_prompt("knowledge-update", "Q?", "A", "R")
    assert "unanswerable" in anscheck_prompt("multi-session", "Q?", "A", "R", abstention=True)
    import pytest
    with pytest.raises(NotImplementedError):
        anscheck_prompt("unknown-type", "Q?", "A", "R")
    class Yes:
        def complete(self, system, user, max_tokens=16, temperature=0.0):
            class R: text = "Yes."; prompt_tokens = 1; completion_tokens = 1; latency_ms = 0.0
            return R()
    class No(Yes):
        def complete(self, *a, **k):
            r = super().complete(*a, **k); r.text = "no"; return r
    assert judge_lme(Yes(), "multi-session", "Q?", "A", "R") == 1.0 and judge_lme(No(), "multi-session", "Q?", "A", "R") == 0.0


def test_lme_part_filter(tmp_path):
    """--lme_part keeps the split ids only, names the file lme-<part>, and uses the official per-type judge (mock answers -> judge 0/1)."""
    import json
    sess = [[{"role": "user", "content": "I adopted a dog named Rex."}, {"role": "assistant", "content": "Nice."}]]
    items = []
    for i, (qt, abs_) in enumerate([("temporal-reasoning", False), ("multi-session", False), ("multi-session", True), ("knowledge-update", False)]):
        items.append({"question_id": f"q{i}" + ("_abs" if abs_ else ""), "question_type": qt, "question": "What is the dog's name?", "answer": "Rex",
                      "question_date": "2023/05/20 (Sat) 10:00", "haystack_dates": ["2023/05/01 (Mon) 10:00"], "haystack_session_ids": ["s1"],
                      "haystack_sessions": sess, "answer_session_ids": ["s1"]})
    data = tmp_path / "lme.json"; data.write_text(json.dumps(items), encoding="utf-8")
    split = tmp_path / "split.json"; split.write_text(json.dumps({"dev": ["q0"], "test_temporal": [], "test_other": ["q1", "q2_abs"]}), encoding="utf-8")
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, "-m", "tqmem.eval.run", "--dataset", "lme", "--set", str(data), "--lme_split", str(split), "--lme_part", "test",
                        "--system", "rag", "--embedder", "hash", "--llm", "mock", "--translator", "rule", "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    fs = os.listdir(out); assert fs == ["rag_mock_lme-test_run1.jsonl"], fs
    recs = [json.loads(l) for l in open(out / fs[0], encoding="utf-8")]
    assert sorted(x["question_id"] for x in recs) == ["q1", "q2_abs"] and all(x["judge"] in (0.0, 1.0) for x in recs)
    assert [x for x in recs if x["question_id"] == "q2_abs"][0]["abstention"] is True


def test_cached_encode(tmp_path):
    """Second call with the same texts is served from disk without computing; a different list (or short lists) is not cached."""
    import numpy as np
    from tqmem.embed import cached_encode, CACHE_MIN
    calls = []
    def compute(texts):
        calls.append(len(texts)); return np.random.RandomState(len(texts)).rand(len(texts), 4).astype(np.float32)
    texts = [f"turn {i}" for i in range(CACHE_MIN + 5)]
    a = cached_encode(str(tmp_path), "m/x", texts, compute); b = cached_encode(str(tmp_path), "m/x", texts, compute)
    assert calls == [len(texts)] and np.array_equal(a, b) and os.listdir(tmp_path / "m_x")
    cached_encode(str(tmp_path), "m/x", texts[:-1], compute); assert len(calls) == 2            # different list -> recomputed
    cached_encode(str(tmp_path), "m/x", texts[:3], compute); assert len(calls) == 3              # short list -> never cached
    cached_encode(str(tmp_path), "m/x", texts[:3], compute); assert len(calls) == 4
    cached_encode("", "m/x", texts, compute); assert len(calls) == 5                             # cache disabled


def test_parse_json_open_span():
    """Open-ended range spans become one-sided: null start -> OPEN_START, null end -> OPEN_END; both null -> fallback (mode none)."""
    import numpy as np
    from tqmem.translate import parse_json, OPEN_START, OPEN_END
    from tqmem.store import to_day
    from tqmem import kernels as KN
    tq = parse_json('{"query":"airlines I flew with","temporal":{"role":"constraint","mode":"range","center":null,"span":[null,"2023-03-02"],"width":"medium"}}', "Q")
    assert tq.mode == "range" and tq.span == (OPEN_START, "2023-03-02")
    tq2 = parse_json('{"query":"x","temporal":{"role":"constraint","mode":"range","center":null,"span":["2022-01-01","null"],"width":"wide"}}', "Q")
    assert tq2.span == ("2022-01-01", OPEN_END)
    tq3 = parse_json('{"query":"x","temporal":{"role":"constraint","mode":"range","center":null,"span":[null,null],"width":"wide"}}', "Q")
    assert tq3.mode == "none"
    lo, hi = to_day(tq.span[0]), to_day(tq.span[1]); t = np.array([to_day("2021-06-01"), to_day("2023-03-01"), to_day("2023-04-01")])
    k = KN.kernel("range", t, to_day("2023-03-02"), None, (lo, hi), 10.0, 10.0, 14.0)
    if k is not None:
        assert k[0] > 0.99 and k[1] > 0.5 and k[2] < 0.05          # one-sided: everything before the end is in, after is out
    assert KN.hard_window("range", to_day("2023-03-02"), None, (lo, hi), 10.0, 10.0, 14.0) == (lo, hi) and lo < to_day("1900-01-01")


def test_aggregate_guard():
    from tqmem.eval.run import _AggregateGuard, _AGG
    from tqmem.translate import TemporalQuery
    class Inner:
        def translate(self, q, now): return TemporalQuery(query=q, mode="range", span=("2023-02-01", "2023-05-01"), width="medium"), {"llm_calls": 1}
    g = _AggregateGuard(Inner())
    tq, _ = g.translate("How many babies were born to friends in the last few months?", "2023-05-13")
    assert tq.mode == "none" and tq.span is None and tq.aggregate is True and tq.to_dict()["aggregate"] is True
    tq2, _ = g.translate("What did I do in March?", "2023-05-13")
    assert tq2.mode == "range" and tq2.aggregate is False
    assert _AGG.search("In total, how much did I spend?") and not _AGG.search("When did I fly to Boston?")


def test_lme_mem0_fake_path(tmp_path):
    """LongMemEval + Mem0 (fake backend): per-item store, session id as hit id so session-level gold scores recall, write-cost file."""
    import json
    sess = [[{"role": "user", "content": "I adopted a dog named Rex."}, {"role": "assistant", "content": "Nice."}],
            [{"role": "user", "content": "The weather is cold today."}, {"role": "assistant", "content": "Stay warm."}]]
    items = [{"question_id": "q0", "question_type": "single-session-user", "question": "What is the dog's name?", "answer": "Rex",
              "question_date": "2023/05/20 (Sat) 10:00", "haystack_dates": ["2023/05/01 (Mon) 10:00", "2023/05/02 (Tue) 10:00"],
              "haystack_session_ids": ["s1", "s2"], "haystack_sessions": sess, "answer_session_ids": ["s1"]}]
    data = tmp_path / "lme.json"; data.write_text(json.dumps(items), encoding="utf-8"); out = tmp_path / "out"
    for sysn in ["mem0", "mem0+tq:nt.fq"]:
        r = subprocess.run([sys.executable, "-m", "tqmem.eval.run", "--dataset", "lme", "--set", str(data), "--system", sysn, "--embedder", "hash", "--llm", "mock",
                            "--translator", "rule", "--no_judge", "--mem0_backend", "fake", "--out", str(out)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
    fs = sorted(os.listdir(out)); assert any(f.startswith("write_cost_mem0_") for f in fs), fs
    # per-session unit: one add per session (2 sessions -> 2 llm_calls), hit ids are still session ids
    out2 = tmp_path / "out2"
    r = subprocess.run([sys.executable, "-m", "tqmem.eval.run", "--dataset", "lme", "--set", str(data), "--system", "mem0", "--embedder", "hash", "--llm", "mock",
                        "--translator", "rule", "--no_judge", "--mem0_backend", "fake", "--mem0_unit", "session", "--out", str(out2)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    wc = [json.loads(l) for l in open(next(out2 / f for f in os.listdir(out2) if f.startswith("write_cost_mem0_")), encoding="utf-8")]
    assert wc[0]["llm_calls"] == 2 and wc[0]["n_memories"] == 2, wc
    rec2 = [json.loads(l) for l in open(next(out2 / f for f in os.listdir(out2) if f.startswith("mem0_mock")), encoding="utf-8")][0]
    assert rec2["recall10"] == 1.0 and all(h["id"] in ("s1", "s2") for h in rec2["retrieved"])
    recs = [json.loads(l) for l in open(out / "mem0_mock_longmemeval_s_run1.jsonl" if (out / "mem0_mock_longmemeval_s_run1.jsonl").exists() else out / "mem0_mock_lme_run1.jsonl", encoding="utf-8")]
    assert recs[0]["recall10"] == 1.0 and all(h["id"] in ("s1", "s2") for h in recs[0]["retrieved"]), recs[0]["retrieved"]

def test_round4_scores():
    """4th-round temporal scores (2026-09-29): shapes, ranges and the defining properties of each."""
    from tqmem import tscore as TS
    t = np.arange(0.0, 400.0, 1.0)
    g = TS.notype_kernel("range", t, 399.0, None, (100.0, 140.0), 4.0, 3.0, 7.0)
    assert g.argmax() == 120 and abs(g[100] - np.exp(-0.5)) < 1e-9          # one Gaussian at the midpoint, sd = half the span
    assert TS.notype_kernel("recent", t, 399.0, None, None, 4.0, 3.0, 7.0).argmax() == 399
    d = TS.asso_decay(t, 399.0)
    assert d[-1] == 1.0 and np.all(np.diff(d) > 0)                          # AssoMem released code: query-agnostic recency only
    assert TS.asso_decay(np.array([500.0]), 399.0)[0] == 1.0                  # negative age -> 1.0 as in their code
    v = TS.tvec_scores(np.full(t.shape, 0.5), t, 200.0, 0.0, 399.0)
    assert np.all(v <= 1.0 + 1e-9)
    w, diag = TS.cmi_weights([{"mode": "anchored", "s": float(i % 7) / 7, "K": float(i % 2), "y": i % 2} for i in range(60)], T=0.1)
    assert w["anchored"][1] > w["anchored"][0] and abs(sum(w["anchored"]) - 1) < 1e-9   # K predicts the label -> temporal weight wins

def test_round4_replay_variants(tmp_path):
    """--replay reuses a reference run's extraction; every 4th-round variant runs on RAG and (fake) Mem0; --only_timed keeps only
    time-conditioned questions; --retrieval_only makes no answer / judge call."""
    out = str(tmp_path)
    base = [sys.executable, "-m", "tqmem.eval.run", "--split", "dev", "--embedder", "hash", "--llm", "mock", "--translator", "rule", "--no_judge", "--limit", "8", "--out", out]
    r = subprocess.run(base + ["--system", "rag+tq:nt.fq.nowin"], capture_output=True, text=True); assert r.returncode == 0, r.stderr
    ref = os.path.join(out, [f for f in os.listdir(out) if f.startswith("rag+tq-nt.fq.nowin_")][0])
    # the rule translator almost never finds a time condition on dev: give 3 questions a real one so every variant scores time
    lines = [json.loads(l) for l in open(ref, encoding="utf-8")]; now = "2023-08-01"
    forced = [{"mode": "anchored", "center": "2023-05-08", "span": None, "width": "narrow"},
              {"mode": "range", "center": None, "span": ["2023-05-01", "2023-06-30"], "width": "medium"},
              {"mode": "recent", "center": None, "span": None, "width": "medium"}]
    for r, f in zip(lines, forced): r["tq"] = dict(r["tq"], role="constraint", query=None, **f)
    with open(ref, "w", encoding="utf-8") as fw: fw.writelines(json.dumps(r) + "\n" for r in lines)
    recs = {r["question_id"]: r for r in lines}
    env = dict(os.environ, TQMEM_TEMB_CACHE=os.path.join(out, "temb"))
    cmi = {"folds": {}, "pooled": {"weights": {"none": [1, 0], "recent": [.8, .2], "range": [.8, .2], "anchored": [.6, .4]}}}
    json.dump(cmi, open(os.path.join(out, "cmi.json"), "w"))
    for var in ["notype", "fixw", "mul", "tvec", "asso", "temb", "cmi"]:
        for sysn in [f"rag+tq:nt.fq.nowin.{var}", f"mem0+tq:nt.fq.{var}"]:
            cmd = base + ["--system", sysn, "--replay", ref, "--only_timed", "--retrieval_only", "--cmi", os.path.join(out, "cmi.json")]
            if sysn.startswith("mem0"): cmd += ["--mem0_backend", "fake"]
            r = subprocess.run(cmd, capture_output=True, text=True, env=env); assert r.returncode == 0, (sysn, r.stderr[-800:])
            f = [x for x in os.listdir(out) if x.startswith(sysn.replace(":", "-") + "-ret_")][0]
            got = open(os.path.join(out, f), encoding="utf-8").readlines(); assert len(got) == 3, (sysn, len(got))
            for line in got:
                rec = json.loads(line)
                assert rec["prediction"] == "" and rec["f1"] is None and rec["replayed_from"] == os.path.basename(ref)
                assert rec["tq"]["mode"] == recs[rec["question_id"]]["tq"]["mode"] != "none" and rec["retrieved"]
    assert os.listdir(os.path.join(out, "temb"))   # temb really embedded time strings, into the test's own cache
    for sysn in ["mem0+ga", "mem0+cl"]:
        r = subprocess.run(base + ["--system", sysn, "--mem0_backend", "fake"], capture_output=True, text=True); assert r.returncode == 0, (sysn, r.stderr[-800:])
