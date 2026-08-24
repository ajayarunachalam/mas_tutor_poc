"""
Async Anthropic-compatible client routed through LiteLLM, with MLflow
tracing autologged on every call.

Drop-in replacement for AsyncAnthropic() / OpenRouterClient() — same
`await client.messages.create(model, max_tokens, system, messages)`
call shape agents already use, so no changes needed in tutor_agent.py,
assessment_agent.py, etc. Swap the instantiation in chat_ui.py / main.py
and every call becomes an MLflow trace automatically.

Why LiteLLM here instead of separate provider-specific clients: one
model string convention ("claude-..." for Anthropic, "ollama_chat/..."
for local Ollama, "openrouter/..." for OpenRouter) routes to the right
provider without a different client class per provider. This makes
compat/ollama_client.py and compat/openrouter_client.py optional going
forward — keep them for the one-off sync scripts that already use them
(extract_concepts.py), but new call sites can standardize on this one.

Why MLflow here: `mlflow.litellm.autolog()` wraps every LiteLLM call in
an MLflow trace automatically — no manual span creation. Traces capture
prompt, response, latency, and token usage out of the box. Call
`tag_current_trace(...)` right after a completion to attach session_id /
agent / decision context, so a trace in the MLflow UI links back to the
matching row in audit_events / session_log (same session_id).
"""
from __future__ import annotations

import os

try:
    import litellm
    import mlflow
    _OBSERVABILITY_AVAILABLE = True
except ImportError:
    _OBSERVABILITY_AVAILABLE = False

# ── One-time MLflow setup ───────────────────────────────────────────────────
# Local file-store tracking by default — no separate server process needed.
# `uv run mlflow ui` reads the same ./mlruns directory to view traces.
# Only runs when the packages are actually installed AND the gateway is in
# use — importing this module (e.g. for tag_current_trace) must stay safe
# even with LLM_GATEWAY=direct or before `uv sync` has picked up the new deps.
if _OBSERVABILITY_AVAILABLE and os.getenv("LLM_GATEWAY", "direct").lower() == "litellm":
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "mas-tutor"))
    mlflow.litellm.autolog()
    litellm.api_base = os.getenv("OLLAMA_HOST", "http://localhost:11434")


def _resolve_model(model: str) -> str:
    """Map a bare model name to a LiteLLM-routable string.

    - Already has a provider prefix (contains "/") → pass through.
    - Starts with "claude" → Anthropic, LiteLLM routes this natively
      using ANTHROPIC_API_KEY from the environment.
    - Anything else (e.g. "gemma4:latest") → assume local Ollama.
    """
    if "/" in model:
        return model
    if model.startswith("claude"):
        return model
    return f"ollama_chat/{model}"


def tag_current_trace(**tags: str) -> None:
    """Attach interpretability tags (session_id, agent, decision, ...) to
    whichever MLflow trace is currently active. Call this immediately
    after a `client.messages.create(...)` call. Safe no-op if mlflow isn't
    installed, isn't in use, or there's no active trace — observability
    must never break the tutoring path."""
    if not _OBSERVABILITY_AVAILABLE:
        return
    try:
        mlflow.update_current_trace(tags={k: str(v) for k, v in tags.items()})
    except Exception:
        pass


class _TextBlock:
    def __init__(self, text: str):
        self.text = text


class _Usage:
    def __init__(self, input_tokens: int, output_tokens: int):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _MessagesResponse:
    def __init__(self, text: str, stop_reason: str = "end_turn", usage: _Usage | None = None):
        self.content = [_TextBlock(text)]
        self.stop_reason = stop_reason
        self.usage = usage or _Usage(0, 0)


class _Messages:
    async def create(self, model: str, max_tokens: int, messages: list, system: str = "") -> _MessagesResponse:
        resolved_model = _resolve_model(model)
        oai_messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": m["role"], "content": m["content"]} for m in messages
        ]

        resp = await litellm.acompletion(
            model=resolved_model,
            messages=oai_messages,
            max_tokens=max_tokens,
        )

        choice = resp.choices[0]
        text = choice.message.content or ""
        usage = _Usage(
            input_tokens=getattr(resp.usage, "prompt_tokens", 0),
            output_tokens=getattr(resp.usage, "completion_tokens", 0),
        )
        stop_reason = "max_tokens" if choice.finish_reason == "length" else "end_turn"
        return _MessagesResponse(text, stop_reason=stop_reason, usage=usage)


class LiteLLMClient:
    """Drop-in replacement for AsyncAnthropic() / OpenRouterClient() — routes
    every call through LiteLLM and traces it in MLflow automatically."""

    def __init__(self):
        if not _OBSERVABILITY_AVAILABLE:
            raise ImportError(
                "LLM_GATEWAY=litellm requires the 'litellm' and 'mlflow' packages. "
                "Add them to pyproject.toml dependencies and run `uv sync`."
            )
        self.messages = _Messages()
