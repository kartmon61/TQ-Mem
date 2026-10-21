"""Preflight: verify credentials (AWS chain, or the llm-gateway key when configs/common.yaml uses gateway: specs),
both LLM families, the OpenAI-compatible path mem0 uses, the embedder and the mem0 config before any paid run. Read-only except one tiny completion per model (a few tokens).

    python scripts/preflight.py            # everything
    python scripts/preflight.py --skip-embed
"""
from __future__ import annotations
import argparse, os, sys, time, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tqmem  # noqa: F401  -- loads the project .env into the environment BEFORE any boto3 session is created

def ok(msg): print(f"  OK   {msg}")
def bad(msg): print(f"  FAIL {msg}")

def check_credentials():
    print("[1] AWS credentials (boto3 chain)")
    import boto3
    try:
        ident = boto3.session.Session().client("sts").get_caller_identity()
        ok(f"account {ident['Account']} arn {ident['Arn']}"); return True
    except Exception as e:
        bad(f"{type(e).__name__}: {str(e)[:160]}"); return False

def check_llm(spec: str, label: str):
    print(f"[2] {label}: {spec}")
    from tqmem.llm import make_llm
    try:
        llm = make_llm(spec); t0 = time.time()
        r = llm.complete("You are helpful.", "Say OK.", max_tokens=16)
        ok(f"text={r.text.strip()!r} prompt_tokens={r.prompt_tokens} completion_tokens={r.completion_tokens} {time.time()-t0:.1f}s")
        return r.prompt_tokens > 0 and r.completion_tokens > 0
    except Exception as e:
        bad(f"{type(e).__name__}: {str(e)[:300]}"); return False

def check_openai_compat(model: str, region: str):
    """Same key/path mem0 will use (bedrock-runtime /openai/v1 + Bedrock API key)."""
    print(f"[3] OpenAI-compatible path for mem0: {model} @ bedrock-runtime.{region}/openai/v1")
    from tqmem.llm import bedrock_bearer_token, OpenAICompatLLM
    tok = bedrock_bearer_token(region)
    if not tok:
        bad("no Bedrock API key (AWS_BEARER_TOKEN_BEDROCK unset and short-term token generation failed)"); return False
    ok(f"bearer token available ({'long-term env' if os.environ.get('AWS_BEARER_TOKEN_BEDROCK') else 'short-term from boto3'})")
    try:
        r = OpenAICompatLLM(model, base_url=f"https://bedrock-runtime.{region}.amazonaws.com/openai/v1", api_key=tok).complete("You are helpful.", "Say OK.", max_tokens=16)
        ok(f"text={r.text.strip()!r} tokens {r.prompt_tokens}/{r.completion_tokens}"); return True
    except Exception as e:
        bad(f"{type(e).__name__}: {str(e)[:300]}"); return False

def check_gateway(models: list[str]):
    """LLM gateway: key present (never printed), GET /v1/models lists the requested models, GET /v1/usage/me budget."""
    print("[1] llm-gateway")
    from tqmem.llm import gateway_url, gateway_api_key
    import urllib.request, json
    try:
        key = gateway_api_key()
    except Exception as e:
        bad(str(e)); return False
    ok(f"LLM_GATEWAY_KEY present (...{key[-4:]}), url {gateway_url()}")
    def get(path):
        req = urllib.request.Request(gateway_url() + path, headers={"Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read().decode("utf-8")), dict(r.headers)
    try:
        body, hdr = get("/v1/models")
        ids = sorted(m.get("id", "") for m in body.get("data", []))
        missing = [m for m in models if m not in ids]
        if missing: bad(f"models not offered by the gateway: {missing}; offered: {ids}"); return False
        ok(f"/v1/models offers {len(ids)} models incl. {models}")
        budget = {k: v for k, v in hdr.items() if k.lower().startswith("x-budget")}
        if budget: ok(f"budget headers {budget}")
    except Exception as e:
        bad(f"/v1/models: {type(e).__name__}: {str(e)[:200]}"); return False
    try:
        body, _ = get("/v1/usage/me")
        ok(f"/v1/usage/me: {json.dumps(body, ensure_ascii=False)[:300]}")
    except Exception as e:
        print(f"  note /v1/usage/me not readable: {type(e).__name__}: {str(e)[:120]}")
    return True

def check_gateway_compat(model: str):
    """Same path/key mem0 will use: OpenAI-compatible chat completions on the gateway with the mem0 model."""
    print(f"[3] OpenAI-compatible path for mem0: {model} @ llm-gateway /v1")
    from tqmem.llm import gateway_url, gateway_api_key, OpenAICompatLLM
    try:
        llm = OpenAICompatLLM(model, base_url=gateway_url() + "/v1", api_key=gateway_api_key(), timeout=60.0)
        r = llm.complete("You are helpful.", "Say OK.", max_tokens=16)
        ok(f"text={r.text.strip()!r} tokens {r.prompt_tokens}/{r.completion_tokens} budget_remaining={llm.budget_remaining}"); return True
    except Exception as e:
        bad(f"{type(e).__name__}: {str(e)[:300]}"); return False

def check_embedder(name: str):
    print(f"[4] embedder {name}")
    try:
        from tqmem.embed import SentenceTransformerEmbedder
        t0 = time.time(); e = SentenceTransformerEmbedder(name); v = e.encode(["hello"])
        ok(f"dim {v.shape[-1]} {time.time()-t0:.0f}s"); return True
    except Exception as e:
        bad(f"{type(e).__name__}: {str(e)[:200]}"); return False

def check_mem0_config(path: str):
    print(f"[5] mem0 config {path}")
    try:
        from mem0.configs.base import MemoryConfig
        cfg = yaml.safe_load(open(path, encoding="utf-8")); MemoryConfig(**cfg)
        ok(f"schema valid: llm={cfg['llm']['provider']}/{cfg['llm']['config']['model']} vs={cfg['vector_store']['provider']}"); return True
    except Exception as e:
        bad(f"{type(e).__name__}: {str(e)[:200]}"); return False

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--common", default="configs/common.yaml"); ap.add_argument("--mem0", default="configs/mem0.yaml")
    ap.add_argument("--skip-embed", action="store_true"); a = ap.parse_args()
    c = yaml.safe_load(open(a.common, encoding="utf-8"))
    m = yaml.safe_load(open(a.mem0, encoding="utf-8"))["llm"]["config"]
    specs = [c["llm_family_A"], c["llm_family_B"], c.get("judge_llm", c["llm_family_A"])]
    if all(sp.startswith("gateway:") for sp in specs):   # 2026-09-18: llm-gateway route (no AWS credentials involved)
        results = [check_gateway(sorted({sp.split(":", 1)[1] for sp in specs} | {m["model"]}))]
        results.append(check_llm(c["llm_family_A"], "family A (gateway chat completions)"))
        results.append(check_llm(c["llm_family_B"], "family B (gateway chat completions)"))
        results.append(check_gateway_compat(m["model"]))
    else:
        results = [check_credentials()]
        results.append(check_llm(c["llm_family_A"], "family A (Converse)"))
        results.append(check_llm(c["llm_family_B"], "family B (Converse)"))
        region = next((seg for seg in m["openai_base_url"].split(".") if seg.startswith(("us-", "eu-", "ap-"))), "us-east-1")
        results.append(check_openai_compat(m["model"], region))
    if not a.skip_embed: results.append(check_embedder(c["embedder"]))
    results.append(check_mem0_config(a.mem0))
    print("\nPREFLIGHT", "PASS" if all(results) else f"FAIL ({results.count(False)} of {len(results)})")
    sys.exit(0 if all(results) else 1)

if __name__ == "__main__":
    main()
