"""LLM clients with usage accounting. OpenAI-compatible endpoint (also vLLM) and a mock for tests."""
from __future__ import annotations
import json, os, sys, time
from dataclasses import dataclass, field

@dataclass
class LLMResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    model: str = ""

class LLM:
    name: str = "base"
    def complete(self, system: str, user: str, max_tokens: int = 256, temperature: float = 0.0) -> LLMResult:
        raise NotImplementedError

_RETRYABLE = ("Timeout", "Throttling", "ThrottlingException", "TooManyRequests", "ServiceUnavailable", "ModelNotReady", "RateLimit", "429", "503", "ConnectionError", "RemoteProtocolError")

def _retry(fn, attempts: int = 4, base: float = 2.0):
    """Retry transient provider failures (read timeouts, throttling, 5xx) with exponential backoff; re-raise anything else at once."""
    import random
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            name = type(e).__name__ + " " + str(e)
            if i == attempts - 1 or not any(k in name for k in _RETRYABLE):
                raise
            wait = base * (2 ** i) + random.random()
            print(f"[llm retry {i+1}/{attempts-1}] {type(e).__name__}: {str(e)[:90]} -> sleep {wait:.0f}s", file=sys.stderr, flush=True)
            time.sleep(wait)

def _fixed_sampling(model: str) -> bool:
    """OpenAI GPT-5.x on Bedrock (verified 2026-09-14 with us.openai.gpt-5.6-luna): Converse rejects `temperature`; Chat Completions rejects
    `max_tokens` (use `max_completion_tokens`) and any temperature other than the default 1. So for these models the temperature-0 rule of
    the design cannot be enforced; sampling stays at the provider default and this is reported in the observations."""
    return "gpt-5" in model or "openai." in model

def bedrock_bearer_token(region: str = "us-east-1") -> str | None:
    """Bedrock API key for the OpenAI-compatible paths. Order: AWS_BEARER_TOKEN_BEDROCK env (long-term key) ->
    short-term token generated from the boto3 credential chain (aws-bedrock-token-generator, 12h). None if neither works."""
    tok = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
    if tok:
        return tok
    try:
        from aws_bedrock_token_generator import provide_token
        return provide_token(region=region)
    except Exception:
        return None

def gateway_url() -> str:
    """Base URL of an OpenAI-compatible LLM gateway, from LLM_GATEWAY_URL (no default)."""
    u = os.environ.get("LLM_GATEWAY_URL")
    if not u:
        raise RuntimeError("LLM_GATEWAY_URL is not set: add `LLM_GATEWAY_URL=https://<your-gateway>` to the project .env")
    return u.rstrip("/")

def gateway_api_key() -> str:
    """API key of the LLM gateway. Read once from the environment (project .env: LLM_GATEWAY_KEY=...);
    it is never taken from a command line or written to any file by the code."""
    k = os.environ.get("LLM_GATEWAY_KEY") or os.environ.get("LLM_GATEWAY_API_KEY")
    if not k:
        raise RuntimeError("LLM_GATEWAY_KEY is not set: add `LLM_GATEWAY_KEY=...` to the project .env")
    return k

class OpenAICompatLLM(LLM):
    """Works for OpenAI API, vLLM's OpenAI-compatible server and an OpenAI-compatible LLM gateway /v1/chat/completions (set base_url).
    Non-streaming only (the gateway does not stream Claude on this path, and we never stream)."""
    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None, timeout: float = 120.0):
        from openai import OpenAI
        # max_retries 2 inside the SDK (429/5xx/timeouts with backoff) + tqmem._retry on top for the transient names in _RETRYABLE
        self.client = OpenAI(base_url=base_url, api_key=api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"), timeout=timeout, max_retries=2)
        self.model = model; self.name = model
        self.budget_remaining: str | None = None   # llm-gateway x-budget-remaining (USD) of the last response, when the server sends it
    def _kwargs(self, max_tokens: int, temperature: float) -> dict:
        # GPT-5.x: reasoning by default, temperature fixed at 1, `max_tokens` rejected -> max_completion_tokens (>= 16) + effort none.
        # Claude/others through the same path: temperature 0 + max_tokens.
        if _fixed_sampling(self.model):
            return {"max_completion_tokens": max(max_tokens, 16), "reasoning_effort": "none"}
        return {"temperature": temperature, "max_tokens": max_tokens}
    def complete(self, system, user, max_tokens=256, temperature=0.0):
        t0 = time.time()
        def call():
            raw = self.client.chat.completions.with_raw_response.create(
                model=self.model, **self._kwargs(max_tokens, temperature),
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
            b = raw.headers.get("x-budget-remaining")
            if b is not None: self.budget_remaining = b
            return raw.parse()
        r = _retry(call)
        u = r.usage
        return LLMResult(r.choices[0].message.content or "", u.prompt_tokens, u.completion_tokens,
                         (time.time() - t0) * 1000, self.model)

class BedrockConverseLLM(LLM):
    """Amazon Bedrock Converse API (for Claude and other non-OpenAI models on Bedrock). Uses boto3 credentials from the environment."""
    def __init__(self, model_id: str, region: str = "us-east-1"):
        import boto3
        from botocore.config import Config
        # fresh session: never reuse boto3's default session, whose credentials may have been resolved (from a stale profile) before .env was loaded.
        # read_timeout 120s + adaptive retries: a single 60s read timeout used to abort a whole run (2026-09-14, rag+tq dev at 30/233).
        # tcp_keepalive: the corporate proxy silently drops idle keep-alive connections; without keepalive a reused dead connection
        # hangs until read_timeout (observed 574s answers = several 120s timeouts + backoff, 2026-09-14). 45s is generous for Luna
        # (~3s typical, fc prompts ~15s); botocore retries 3x standard, then tqmem._retry adds backoff on top.
        # Measured 2026-09-14 (GPT-5.6 Luna on Bedrock): ~8% of ordinary ~500-token requests take 30-90s server-side even with
        # reasoning effort "none" (Haiku answers the same prompts in ~1s); they do complete. A short read timeout therefore turns them into
        # repeated timeouts -> error records. So wait up to 100s (180s for full-context prompts), no silent botocore retries, and let
        # tqmem._retry handle the real transient errors (503 ServiceUnavailable on the us. profile, occasional dropped connections).
        mk = lambda rt: boto3.session.Session().client("bedrock-runtime", region_name=region,
                                                       config=Config(read_timeout=rt, connect_timeout=5, tcp_keepalive=True,
                                                                     max_pool_connections=64, retries={"max_attempts": 1}))
        self.client, self.client_long = mk(200), mk(300)   # 23:40 KST (US morning peak): Luna p90 128s at 16 parallel, max 180s, still completes
        self.model = model_id; self.name = model_id
    def complete(self, system, user, max_tokens=256, temperature=0.0):
        t0 = time.time(); pin = pout = 0
        if _fixed_sampling(self.model): max_tokens = max(max_tokens, 16)   # Luna: "max_output_tokens ... Expected a value >= 16" (crashed the judge at 4, 2026-09-14)
        for mt in (max_tokens, max_tokens * 4):
            if _fixed_sampling(self.model):
                # GPT-5.x on Bedrock reasons by default (hidden reasoning tokens ate the 64-token answer budget -> empty text, slow calls, timeouts).
                # effort "none" (verified 2026-09-14 for us.openai.gpt-5.6-luna; "minimal" is rejected) gives the fast non-reasoning behaviour the design assumes.
                cfg, extra = {"maxTokens": mt}, {"additionalModelRequestFields": {"reasoning": {"effort": "none"}}}
            else:
                cfg, extra = {"maxTokens": mt, "temperature": temperature}, {}
            client = self.client_long if len(system) + len(user) > 20000 else self.client
            r = _retry(lambda: client.converse(modelId=self.model, system=[{"text": system}], messages=[{"role": "user", "content": [{"text": user}]}],
                                               inferenceConfig=cfg, **extra))
            text = "".join(b.get("text", "") for b in r["output"]["message"]["content"]); u = r.get("usage", {})
            pin += u.get("inputTokens", 0); pout += u.get("outputTokens", 0)
            if text.strip() or r.get("stopReason") != "max_tokens":
                break   # else: the budget was spent on non-text (reasoning) blocks -> retry once with 4x tokens
        return LLMResult(text, pin, pout, (time.time() - t0) * 1000, self.model)

class MockLLM(LLM):
    """Deterministic stand-in: answers with the first retrieved snippet's text tail; translation handled by RuleTranslator."""
    name = "mock"
    def complete(self, system, user, max_tokens=256, temperature=0.0):
        # crude: return the last non-empty line containing a bracketed date, else 'Not mentioned'
        lines = [l for l in user.split("\n") if l.startswith("[")]
        text = lines[0].split("] ", 1)[-1][:80] if lines else "Not mentioned"
        return LLMResult(text, len(user) // 4, len(text) // 4, 0.0, "mock")

def make_llm(spec: str) -> LLM:
    """spec:
      'mock'
      'openai:<model>'                      OpenAI API, key from OPENAI_API_KEY
      'vllm:<model>@<base_url>'             any OpenAI-compatible server without auth
      'mantle:<model>@<base_url>'           OpenAI-compatible path on Bedrock (runtime /openai/v1 or mantle), key = AWS_BEARER_TOKEN_BEDROCK or short-term token from boto3 creds
      'bedrock:<model_id>@<region>'         Bedrock Converse API (LoCoMo runs 2026-09-14..17; OpenAI models need a us./global. inference profile ID), boto3 credentials
      'gateway:<model>'                     OpenAI-compatible LLM gateway (/v1/chat/completions at LLM_GATEWAY_URL), key = LLM_GATEWAY_KEY from .env
                                            model names as listed by GET /v1/models, e.g. gpt-5.6-luna, claude-haiku-4-5-20251001 (LongMemEval runs, 2026-09-18-)
    """
    if spec == "mock":
        return MockLLM()
    kind, rest = spec.split(":", 1)
    if kind == "openai":
        return OpenAICompatLLM(rest)
    if kind == "vllm":
        model, url = rest.split("@", 1)
        return OpenAICompatLLM(model, base_url=url, api_key="EMPTY")
    if kind == "mantle":
        model, url = rest.split("@", 1)
        region = next((seg for seg in url.split(".") if seg.startswith(("us-", "eu-", "ap-"))), "us-east-1")
        return OpenAICompatLLM(model, base_url=url, api_key=bedrock_bearer_token(region))
    if kind == "bedrock":
        model, region = rest.split("@", 1) if "@" in rest else (rest, "us-east-1")
        return BedrockConverseLLM(model, region)
    if kind == "gateway":
        # 200s read timeout as on Bedrock: ~8% of Luna requests take 30-90s server-side even with reasoning off (2026-09-14)
        return OpenAICompatLLM(rest, base_url=gateway_url() + "/v1", api_key=gateway_api_key(), timeout=200.0)
    raise ValueError(spec)
