# TQ-Mem

Code for the master's thesis
**"Time as a First-Class Query Dimension: Query-Adaptive Temporal Kernel Retrieval for Long-Term LLM Agent Memory"**
(국문: 시간을 1급 질의 차원으로: 장기 사용 LLM 에이전트 기억을 위한 질의 적응형 시간 커널 검색).

TQ-Mem treats the time condition of a question as part of the query instead of a hard filter.
An LLM translates each question into a temporal query (role, mode, center/span, width grade), and every memory is
scored by a soft temporal kernel combined with semantic similarity:

```
score(m) = α_mode · s_sem(q, m) + β_mode · K_mode(t_m)
```

| mode | example | kernel K(t) |
|---|---|---|
| `none` | "What does Caroline do for work?" | 0 (semantic only) |
| `recent` | "What have I been reading lately?" | exponential decay from *now* (half-life by width grade) |
| `anchored` | "Where was Jon on 28 January 2023?" | Gaussian around the anchor date |
| `range` | "Who did Gina team up with during February 2023?" | product of two sigmoids over the span |

Questions that *ask* for a time ("When did …?") get role `target` and no kernel. Because the kernel is soft, evidence
that falls just outside the parsed window can still be retrieved, which a hard time filter cannot do.
TQ-Mem works on a plain dense-retrieval store (RAG) and as a reranker on top of [Mem0](https://github.com/mem0ai/mem0).

## Setup

```bash
git clone https://github.com/kartmon61/TQ-Mem.git && cd TQ-Mem
pip install -r requirements.txt          # Python 3.10+
python scripts/download_data.py          # LoCoMo + LongMemEval_S into data/ (SHA-256 checked)
python -m pytest -q tests                # mock LLM + hashing embedder, no API key needed
```

Credentials are read from the environment or a project `.env` (see `.env.example`); they are never passed on the command line.
LLM backends are chosen by a spec string (`tqmem/llm.py::make_llm`):

| spec | backend |
|---|---|
| `bedrock:<model_id>@<region>` | Amazon Bedrock Converse API (used for the reported LoCoMo runs) |
| `openai:<model>` | OpenAI API |
| `vllm:<model>@<base_url>` | any OpenAI-compatible server |
| `gateway:<model>` | OpenAI-compatible gateway at `LLM_GATEWAY_URL` |
| `mock` | deterministic stub for tests |

All LLM calls use temperature 0 where the model accepts it. Models in the thesis: GPT-5.6 Luna (family A, also the judge) and
Claude Haiku 4.5 (family B); embedder `BAAI/bge-large-en-v1.5`. Check credentials and models before a paid run with
`python scripts/preflight.py`.

## Systems

| `--system` | description |
|---|---|
| `rag` | semantic top-k |
| `rag+ga` | Generative-Agents-style recency |
| `rag+hf` | hard time filter on the extracted window (TSM-style) |
| `rag+cl` | iterative LLM retrieval loop (Chronos-style) |
| `rag+tq:nt.fq.nowin` | **TQ-Mem** on the RAG store |
| `fc` | full context (upper reference) |
| `mem0`, `mem0+hf` | Mem0 and Mem0 + hard filter |
| `mem0+tq:nt.fq` | **TQ-Mem** reranking Mem0's top-100 |

Variant suffixes (combinable with `.`): `nt` no semantic floor, `fq` embed the full question, `nowin` no candidate window,
`wr` weak recent. Plain `rag+tq` / `mem0+tq` keep the initial design for the ablation.

## Reproducing the experiments

```bash
A=bedrock:global.openai.gpt-5.6-luna@us-east-1
E=BAAI/bge-large-en-v1.5

# LoCoMo test, TQ-Mem (RAG store) with leave-one-conversation-out parameters
python -m tqmem.eval.run --system rag+tq:nt.fq.nowin --split test --embedder $E --llm $A \
    --translator llm --tuned_cv configs/tuned_cv.json --judge_llm $A

# baselines
for s in rag rag+ga rag+hf rag+cl fc; do
  python -m tqmem.eval.run --system $s --split test --embedder $E --llm $A --translator llm --judge_llm $A
done

# Mem0 pair
python -m tqmem.eval.run --system mem0+tq:nt.fq --split test --embedder $E --llm $A --translator llm \
    --tuned_cv configs/tuned_cv.json --mem0_config configs/mem0.yaml --judge_llm $A

# LongMemEval_S test (parameters fixed from LoCoMo CV, not tuned on LongMemEval)
python -m tqmem.eval.run --dataset lme --set data/longmemeval_s.json --lme_split data/lme_split.json --lme_part test \
    --system rag+tq:nt.fq.nowin --tuned configs/tuned_v3.yaml --embedder $E --llm $A --translator llm --judge_llm $A

# tables + paired bootstrap 95% CI
python -m tqmem.eval.report "results/e1/*_test_run1.jsonl" --pair rag+tq-nt.fq.nowin-cv rag --subset temporal/constraint
```

Other entry points: `tqmem.eval.tune_cv` (leave-one-conversation-out tuning → `configs/tuned_cv.json`),
`tqmem.eval.boundary` (recall by distance of the gold evidence to the hard-filter window),
`tqmem.eval.sensitivity` (β and width curves), `tqmem.eval.e4` / `e4_agree` (time-condition extraction accuracy),
`scripts/run_round4.py` (mechanism ablations replaying the same extractions).
Results are written as `results/<exp>/{system}_{llm}_{tag}_run{n}.jsonl`, one record per question.

## Main results

LoCoMo test (n = 1,307), family A. *Time-constrained* = questions whose extracted mode is not `none` (n = 246).

| system | recall@10 | token-F1 | judge acc. | time-constrained recall@10 / judge |
|---|---|---|---|---|
| rag | 0.733 | 0.455 | 0.568 | 0.614 / 0.577 |
| rag+hf | 0.700 | 0.419 | 0.535 | 0.541 / 0.480 |
| rag+cl | 0.759 | 0.451 | 0.564 | 0.711 / 0.634 |
| **rag+tq (TQ-Mem)** | **0.771** | **0.472** | **0.591** | **0.817 / 0.732** |
| mem0 | 0.831 | 0.482 | 0.647 | 0.752 / 0.654 |
| mem0+hf | 0.774 | 0.442 | 0.603 | 0.524 / 0.476 |
| **mem0+tq (TQ-Mem)** | **0.851** | **0.486** | **0.668** | **0.858 / 0.748** |

On time-constrained questions TQ-Mem improves over plain RAG by ΔF1 +0.119 [+0.082, +0.160] and over Mem0 by Δrecall@10
+0.106 [+0.069, +0.142] (paired bootstrap 95% CI), without loss on single-hop questions, and it uses one LLM call per
question versus 2.9 for the iterative loop (`rag+cl`). When the gold evidence lies just outside the hard-filter window,
recall@10 goes from 0.000 (hard filter) to 0.725 (RAG) / 0.878 (Mem0).
On LongMemEval_S temporal-reasoning (n = 103), TQ-Mem raises recall@10 by +0.040 [+0.010, +0.081] over RAG.
Full tables are in the thesis.

## Repository layout

```
tqmem/
  translate.py     question -> temporal query (LLM prompt, JSON parser, rule fallback)
  kernels.py       width / weight tables, kernels, candidate and hard windows
  retrieve.py      RAG, GA, MB, HF, CL, TQ (ours), full context
  mem0_adapter.py  Mem0 store and Mem0 + TQ rerank
  store.py embed.py llm.py tscore.py timedim.py gold.py env.py
  bench/           LoCoMo / LongMemEval loaders, metrics, judge prompts
  eval/            run, tune, tune_cv, report, boundary, sensitivity, e4, ...
configs/           models, mem0 configs, tuned parameters (tuned_cv.json, tuned_v3.yaml)
data/              dev tuning questions, LongMemEval split ids (benchmarks are downloaded)
annotations/       author-checked time-condition labels for LoCoMo / LongMemEval temporal questions
scripts/           download_data.py, preflight.py, run_round4.py
tests/
```

## Data and licenses

The code is released under the MIT License. The benchmarks are **not** included; `scripts/download_data.py` fetches
[LoCoMo](https://github.com/snap-research/locomo) (Maharana et al., 2024) and
[LongMemEval](https://github.com/xiaowu0162/LongMemEval) (Wu et al., 2025), each under its own license.
The files in `annotations/` and `data/` derived from these benchmarks (question ids, question text, time-condition labels)
follow the license of the source benchmark.

## Citation

```bibtex
@mastersthesis{lee2026tqmem,
  author = {Lee, Jinho},
  title  = {Time as a First-Class Query Dimension: Query-Adaptive Temporal Kernel Retrieval for Long-Term LLM Agent Memory},
  year   = {2026}
}
```
