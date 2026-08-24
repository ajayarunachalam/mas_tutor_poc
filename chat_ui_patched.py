"""
MAS Education POC — Browser Chat UI
FastAPI app serving an inline HTML chat page on port 8080.

Usage:
    cd ~/projects/mas_education_poc && uv run python chat_ui.py
    Open http://localhost:8080 in Windows 11 browser (Edge or Chrome).
"""
import asyncio
import json
import logging
import os
import random
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
import uvicorn
from anthropic import AsyncAnthropic
from compat.openrouter_client import OpenRouterClient
from compat.litellm_client import LiteLLMClient, tag_current_trace

from agents.dialogue_agent import DialogueTracker
from orchestrator.poc_orchestrator import create_orchestrator, SessionState
from agents.monitor_agent import print_session_report
from db.init_db import init_database
from dashboard import router as dashboard_router


# ── Constants ─────────────────────────────────────────────────────────────────

MAX_SESSIONS = 200
SESSION_TTL_MINUTES = 30
HISTORY_TRIM = 40
LOG_FILE = os.getenv("LOG_FILE", "chat_ui.log")


# ── Concurrency controls (Phase 2 async-migration) ───────────────────────────

# Bound how many LLM turns are in flight simultaneously so excess requests queue
# safely on the semaphore instead of firing unbounded at OpenRouter. In modern
# asyncio (3.10+) a Semaphore lazily binds to the running loop, and only on a
# *contended* acquire — verified empirically that an uncontended size-30
# semaphore is reused across event loops without error, so this module-level
# instance is safe both under the single production uvicorn loop and under the
# per-test TestClient loops (sequential test requests never contend it).
MAX_CONCURRENT_LLM_CALLS = int(os.getenv("MAX_CONCURRENT_LLM_CALLS", "30"))
_llm_semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM_CALLS)

# Size of the default thread pool that asyncio.to_thread() offloads blocking
# work to: up to 4 concurrent ChromaDB query variants per request in
# rag/retrieve.py, plus Neo4j calls in the orchestrator. Python's default is
# min(32, cpu_count + 4) = 6 on this 2-vCPU VPS, which as few as ~2 concurrent
# requests in their RAG-fetch phase would saturate — silently reintroducing the
# queuing the semaphore is meant to make visible. Set explicitly at startup (see
# _configure_thread_pool) since there is no running loop at import time.
MAX_THREAD_POOL_WORKERS = int(os.getenv("MAX_THREAD_POOL_WORKERS", "40"))

# Hard wall-clock budget for a single turn. Bounds the *entire* orchestrator
# invocation (all internal retries included) so no single dependency stall — an
# LLM-provider hang, a slow embedding call, or a wedged tracing/OTel exporter —
# can ever hold a request for minutes again. On expiry the chat handler returns
# a 503 "took too long" (see the asyncio.TimeoutError branch) and the student
# retries. Legit turns are ~8-17s even at 30 concurrent, so 45s is generous
# headroom (~2.6x observed P99) while capping the pathological tail.
TURN_TIMEOUT_SECONDS = int(os.getenv("TURN_TIMEOUT_SECONDS", "45"))


# ── Structured JSON logging ───────────────────────────────────────────────────

_log_handler = logging.FileHandler(LOG_FILE, mode="a")
_log_handler.setFormatter(logging.Formatter("%(message)s"))
_json_logger = logging.getLogger("mas.chat")
_json_logger.setLevel(logging.INFO)
_json_logger.addHandler(_log_handler)
_json_logger.propagate = False


# ── Aggregate metrics (in-memory, reset on restart) ──────────────────────────

_metrics: dict = {
    "total_sessions": 0,
    "total_turns": 0,
    "total_latency_ms": 0,
}


# ── Static FAQ responses (no LLM — zero hallucination risk) ──────────────────

_FAQ_TOPICS = """Here are the topics covered in Cambridge IGCSE Mathematics (0580):

1. **Number** — integers, decimals, fractions, percentages, ratio, proportion, indices, and standard form
2. **Algebra and Graphs** — expressions, equations, inequalities, sequences, functions, and graph plotting
3. **Coordinate Geometry** — straight-line graphs, gradients, midpoints, and the equation of a line
4. **Geometry** — angles, triangles, polygons, circles, constructions, and loci
5. **Mensuration** — area, perimeter, volume, and surface area of 2D and 3D shapes
6. **Trigonometry** — sine, cosine, and tangent in right-angled and non-right-angled triangles
7. **Transformations and Vectors** — reflections, rotations, translations, enlargements, and vector arithmetic
8. **Statistics** — data collection, averages, charts, cumulative frequency, and box plots
9. **Probability** — single and combined events, tree diagrams, and relative frequency

Tell me which topic you'd like to work on, or just ask a maths question and I'll guide you from there."""

_FAQ_HOW_IT_WORKS = """The MAS Tutor uses a **Socratic approach** — rather than giving you the answer, it guides you with questions and hints so you reach the understanding yourself. Research shows this produces much deeper learning than reading solutions.

Here's what happens behind the scenes:
- A **Tutor Agent** asks guiding questions and explains concepts step by step
- An **Assessment Agent** evaluates your answers and tracks your understanding
- A **Planning Agent** maps your progress through the Cambridge IGCSE Maths syllabus
- A **Dialogue Agent** monitors your engagement and adjusts the approach if you get stuck

All responses are grounded in Cambridge IGCSE Mathematics syllabus materials — the tutor will not invent content or go beyond the syllabus.

You advance to the next topic automatically once your mastery of the current one reaches 85%, based on the quality of your answers over several turns."""

_FAQ_HOW_TO_USE = """Here's how to get the most from the MAS Tutor:

1. **Start naturally** — ask a question about any maths topic, or say what you're currently studying (e.g. "I'm working on quadratic equations")
2. **Engage with the questions** — the tutor will ask you questions rather than give answers; think them through and reply honestly
3. **Share your working** — when you think you have an answer, say so: *"I think the answer is..."* or *"My working gives me..."*
4. **Don't worry about being wrong** — mistakes are part of learning; the tutor responds to where you actually are, not where you think you should be
5. **Type 'summary'** at any time to see your session stats: topics covered, current mastery level, and turns completed

The more you engage with the questions — rather than asking for answers — the faster your understanding will build."""

_FAQ_HELP = """I'm the MAS Tutor — a Socratic maths tutor covering Cambridge IGCSE Mathematics.

Here's what you can ask me:
- **"What topics are available?"** — see the full list of topics I cover
- **"How does the tutor work?"** — understand the Socratic approach and multi-agent system
- **"How do I use the tutor?"** — tips for getting the most from your sessions
- **"summary"** — see your session stats at any time
- Or just **ask a maths question** and we'll get started

What would you like to do?"""

_FAQ_PATTERNS: list[tuple[list[str], str]] = [
    (["what topics", "what can you teach", "what subjects", "what do you cover", "what maths", "list topics", "show topics", "available topics"], _FAQ_TOPICS),
    (["how does this work", "how does the tutor work", "how do you work", "what is this", "what is mas", "explain the tutor"], _FAQ_HOW_IT_WORKS),
    (["how do i use", "how to use", "how best to use", "tips for using", "how should i use", "how can i use"], _FAQ_HOW_TO_USE),
]


def _check_faq(text: str) -> str | None:
    """Return a static FAQ response if the input matches a known meta-question, else None."""
    lowered = text.lower().strip()
    for triggers, response in _FAQ_PATTERNS:
        if any(t in lowered for t in triggers):
            return response
    if lowered in {"help", "?", "hi", "hello", "what can you do", "what can you help with"}:
        return _FAQ_HELP
    return None


# Improvement 5: Vague-distress triage — static response with structured topic choices.
# Prevents the orchestrator from making arbitrary wrong topic guesses when the student
# expresses confusion without specifying any topic.
_DISTRESS_TRIAGE_RESPONSE = """That's okay — let's figure out where to start together! 😊

Which area of maths is giving you trouble? Pick the one that feels most urgent:

1. **Algebra** — equations, expressions, factorising
2. **Number** — fractions, percentages, decimals, ratios
3. **Geometry** — angles, shapes, circles, area and volume
4. **Graphs and Functions** — plotting, transformations, functions
5. **Statistics and Probability** — data, averages, tree diagrams
6. **Trigonometry** — sine, cosine, tangent, Pythagoras

Or just describe what you were working on (e.g. "I'm stuck on quadratic equations") and I'll pick it up from there."""

_DISTRESS_TRIGGERS = (
    "i don't understand anything",
    "i don't get any of it",
    "i don't know where to start",
    "i'm lost",
    "i have no idea",
    "everything is confusing",
    "none of it makes sense",
    "i'm completely lost",
    "i can't do maths",
    "i don't understand maths",
    "i don't get maths",
    "i'm bad at maths",
    "i'm terrible at maths",
)


def _check_vague_distress(text: str) -> str | None:
    """Return the triage response if the message is undirected distress with no topic."""
    lowered = text.lower().strip()
    if any(t in lowered for t in _DISTRESS_TRIGGERS):
        return _DISTRESS_TRIAGE_RESPONSE
    return None


# Improvement 7: Detect student resistance to Socratic method (demanding direct answers).
_RESISTANCE_SIGNALS = (
    "just tell me",
    "just say yes",
    "just answer",
    "give me the answer",
    "what is the answer",
    "tell me the answer",
    "just give me",
    "please just tell",
    "please just say",
    "stop asking",
    "i give up",
    "can you just confirm",
    "just confirm",
    "please confirm",
    "just say if",
)


def _is_resistance(text: str) -> bool:
    """Return True if the student is demanding a direct answer rather than engaging."""
    lowered = text.lower()
    return any(s in lowered for s in _RESISTANCE_SIGNALS)


# ── Startup ───────────────────────────────────────────────────────────────────

init_database()
_llm_gateway = os.getenv("LLM_GATEWAY", "direct").lower()
_openrouter_key = os.getenv("OPENROUTER_API_KEY")
if _llm_gateway == "litellm":
    # Routes every call through LiteLLM and traces it in MLflow automatically.
    # Local trace viewing: uv run mlflow ui  (reads ./mlruns)
    anthropic_client = LiteLLMClient()
elif _openrouter_key:
    anthropic_client = OpenRouterClient(api_key=_openrouter_key)
else:
    anthropic_client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

if os.getenv("PHOENIX_ENABLED", "false").lower() == "true":
    try:
        from phoenix.otel import register
        from openinference.instrumentation.langchain import LangChainInstrumentor
        _tp = register(
            project_name="mas-tutor",
            endpoint=os.getenv("PHOENIX_ENDPOINT", "http://localhost:6006/v1/traces"),
        )
        LangChainInstrumentor().instrument(tracer_provider=_tp)
    except ImportError:
        logging.getLogger("mas.chat").warning("PHOENIX_ENABLED=true but phoenix packages not installed")

# ── Rate-limit key: per learner_id, not per IP ───────────────────────────────
# Many students behind one school NAT share a single public IP, so keying the
# limiter on remote address (slowapi's default get_remote_address) puts a whole
# class in one bucket and 429s a third of them. Instead we key on the learner_id
# from the request body.
#
# This MUST be a *sync* function. slowapi calls the key_func synchronously
# (extension.py:502 — `limit_key = lim.key_func(request)`, not awaited); an
# `async def` key_func would hand slowapi an un-awaited coroutine as the bucket
# key, silently disabling rate limiting (every request a unique bucket) plus a
# "coroutine was never awaited" warning. This version reads the JSON body
# FastAPI has *already* parsed and cached on the Request instance (Starlette
# caches it as request._json / request._body) during its own body-parsing phase,
# which runs before slowapi's decorator-mode limiter check — so there is no
# second read of the receive stream, and the route handler still reads the same
# cached body normally. Falls back to IP if the body is absent/unparseable.
def _learner_id_from_cached_body(request: Request) -> str | None:
    data = getattr(request, "_json", None)
    if data is None:
        raw = getattr(request, "_body", None)
        if raw:
            try:
                data = json.loads(raw)
            except (ValueError, TypeError):
                return None
    if isinstance(data, dict):
        lid = data.get("learner_id")
        if isinstance(lid, str) and lid:
            return lid
    return None


def get_learner_id_key(request: Request) -> str:
    lid = _learner_id_from_cached_body(request)
    if lid:
        return f"learner:{lid}"
    return get_remote_address(request)


limiter = Limiter(key_func=get_learner_id_key)
app = FastAPI(title="MAS Education POC")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.include_router(dashboard_router)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


# In-memory session store
active_sessions: dict[str, dict] = {}


# ── Session helpers ───────────────────────────────────────────────────────────

def get_or_create_session(learner_id: str) -> dict:
    if learner_id in active_sessions:
        active_sessions[learner_id]["last_active"] = datetime.utcnow()
        return active_sessions[learner_id]
    if len(active_sessions) >= MAX_SESSIONS:
        raise HTTPException(status_code=503, detail="Service at capacity — try again later")
    session_id = str(uuid.uuid4())[:8]
    tracker = DialogueTracker(learner_id, session_id)
    orchestrator = create_orchestrator(anthropic_client, tracker)
    _metrics["total_sessions"] += 1
    active_sessions[learner_id] = {
        "session_id": session_id,
        "tracker": tracker,
        "orchestrator": orchestrator,
        "history": [],
        "current_topic": None,
        "current_topic_name": "Mathematics",
        "turn_count": 0,
        "last_active": datetime.utcnow(),
        "ability_signal": 0,              # Improvement 4: strong-learner detection
        "guidance_resistance_count": 0,   # Improvement 7: resistance tracking
    }
    _json_logger.info(json.dumps({
        "ts": datetime.utcnow().isoformat(),
        "event": "session_start",
        "learner_id": learner_id,
        "session_id": session_id,
    }))
    return active_sessions[learner_id]


def cleanup_stale_sessions_sync() -> int:
    """Remove sessions idle longer than SESSION_TTL_MINUTES. Returns count removed."""
    cutoff = datetime.utcnow() - timedelta(minutes=SESSION_TTL_MINUTES)
    to_remove = [
        k for k, v in active_sessions.items()
        if v.get("last_active", datetime.utcnow()) < cutoff
    ]
    for k in to_remove:
        _json_logger.info(json.dumps({
            "ts": datetime.utcnow().isoformat(),
            "event": "session_end",
            "reason": "timeout",
            "learner_id": k,
            "session_id": active_sessions[k].get("session_id"),
            "turn_count": active_sessions[k].get("turn_count", 0),
            "final_topic": active_sessions[k].get("current_topic"),
        }))
        del active_sessions[k]
    return len(to_remove)


def trim_session_history(session: dict) -> None:
    """Trim conversation history to the last HISTORY_TRIM messages."""
    if len(session["history"]) > HISTORY_TRIM:
        session["history"] = session["history"][-HISTORY_TRIM:]


async def _invoke_orchestrator(orchestrator, state, max_attempts: int = 3) -> dict:
    """Invoke the LangGraph orchestrator with retry on transient errors.

    Uses orchestrator.ainvoke() so the graph's now-async nodes run on the already-active
    event loop instead of blocking it. This is the actual fix for the LangGraph/asyncio
    conflict the previous sync orchestrator.invoke() worked around: .invoke() internally
    spins up its own loop (asyncio.run), which cannot be awaited from inside a running
    loop; .ainvoke() is awaited directly, so an in-flight LLM call no longer freezes the
    whole process for every other concurrent request.
    Timeout is delegated to OpenRouter/HTTPX client-level settings.
    """
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            # Semaphore bounds concurrent in-flight LLM turns to
            # MAX_CONCURRENT_LLM_CALLS; the permit is held only for the actual
            # ainvoke and released before any retry backoff, so a waiting turn
            # doesn't occupy a slot during another turn's sleep.
            async with _llm_semaphore:
                return await orchestrator.ainvoke(state)
        except Exception as e:
            msg = str(e)
            # Auth/quota errors are permanent — don't retry
            if "credit balance" in msg or "AuthenticationError" in msg or "User not found" in msg:
                raise
            if attempt < max_attempts - 1:
                # Jitter (random.uniform(0, 1)) de-synchronises retries so a
                # transient failure shared across many concurrent users doesn't
                # cause a synchronised retry storm.
                await asyncio.sleep(2 ** attempt + random.uniform(0, 1))
            last_exc = e
    raise last_exc  # type: ignore[misc]


# ── Startup event ─────────────────────────────────────────────────────────────

@app.on_event("startup")
async def _configure_thread_pool():
    """Resize the default ThreadPoolExecutor used by asyncio.to_thread().

    Runs in the startup handler (not at import) because there is no running loop
    at import time, and before any request fires so the lazy 6-thread default is
    never created. On this 2-vCPU box the default (min(32, cpu_count+4) = 6) is
    too small for the app's per-request RAG fan-out.
    """
    loop = asyncio.get_running_loop()
    loop.set_default_executor(
        ThreadPoolExecutor(
            max_workers=MAX_THREAD_POOL_WORKERS,
            thread_name_prefix="mas-worker",
        )
    )
    _json_logger.info(json.dumps({
        "ts": datetime.utcnow().isoformat(),
        "event": "thread_pool_configured",
        "max_workers": MAX_THREAD_POOL_WORKERS,
    }))


@app.on_event("startup")
async def _start_cleanup_task():
    async def _cleanup_loop():
        while True:
            await asyncio.sleep(300)  # every 5 minutes
            cleanup_stale_sessions_sync()
    asyncio.create_task(_cleanup_loop())


# ── API models ────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    learner_id: str = Field(default="demo-student-001", pattern=r"^[a-zA-Z0-9_-]{1,64}$")


class ChatResponse(BaseModel):
    response: str
    engagement_state: str
    mastery: float
    citations: list[str]
    topic: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "mas-tutor"}


@app.get("/admin/metrics")
async def admin_metrics():
    turns = _metrics["total_turns"]
    total_lat = _metrics["total_latency_ms"]
    avg_lat = round(total_lat / turns, 1) if turns > 0 else 0.0
    return {
        "total_sessions": _metrics["total_sessions"],
        "total_turns": turns,
        "avg_latency_ms": avg_lat,
    }


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the inline chat page."""
    return HTMLResponse(content=CHAT_HTML)


@app.get("/terms", response_class=HTMLResponse)
async def terms():
    """Serve the Terms of Use page (EU AI Act Art. 50 transparency + general terms)."""
    return HTMLResponse(content=TERMS_HTML)


@app.get("/privacy", response_class=HTMLResponse)
async def privacy():
    """Serve the Privacy Policy page."""
    return HTMLResponse(content=PRIVACY_HTML)


@app.post("/chat", response_model=ChatResponse)
@limiter.limit("20/minute")
async def chat(request: Request, req: ChatRequest):
    """Process a student message through the LangGraph orchestrator."""
    t_start = time.monotonic()

    faq_response = _check_faq(req.message)
    if faq_response:
        return ChatResponse(
            response=faq_response,
            engagement_state="EXPLORING",
            mastery=0.0,
            citations=[],
            topic="meta",
        )

    # Improvement 5: Vague-distress triage before hitting the orchestrator
    distress_response = _check_vague_distress(req.message)
    if distress_response:
        return ChatResponse(
            response=distress_response,
            engagement_state="EXPLORING",
            mastery=0.0,
            citations=[],
            topic="meta",
        )

    try:
        session = get_or_create_session(req.learner_id)
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"detail": e.detail})
    trim_session_history(session)

    # Improvement 7: Detect resistance before orchestrator call and increment counter
    if _is_resistance(req.message):
        session["guidance_resistance_count"] = session.get("guidance_resistance_count", 0) + 1

    state = SessionState(
        session_id=session["session_id"],
        learner_id=req.learner_id,
        student_input=req.message,
        conversation_history=session["history"],
        current_topic=session.get("current_topic") or "unknown",
        current_topic_name=session.get("current_topic_name") or "Mathematics",
        intent="unknown",
        agent_response="",
        rag_chunks=[],
        citations=[],
        mastery=0.0,
        engagement_state=session["tracker"].state,
        intervention_prompt="",
        should_assess=False,
        advancement_check={},
        turn_count=session["turn_count"],
        ability_signal=session.get("ability_signal", 0),
        guidance_resistance_count=session.get("guidance_resistance_count", 0),
    )

    try:
        result = await asyncio.wait_for(
            _invoke_orchestrator(session["orchestrator"], state),
            timeout=TURN_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        _json_logger.error(json.dumps({
            "ts": datetime.utcnow().isoformat(),
            "event": "timeout",
            "learner_id": req.learner_id,
            "turn": session["turn_count"],
        }))
        return JSONResponse(
            status_code=503,
            content={"detail": "The tutor took too long to respond. Please try again."},
        )
    except Exception as e:
        msg = str(e)
        _json_logger.error(json.dumps({
            "ts": datetime.utcnow().isoformat(),
            "event": "error",
            "learner_id": req.learner_id,
            "turn": session["turn_count"],
            "error_class": type(e).__name__,
            "error_msg": msg[:200],
            "traceback": traceback.format_exc()[-600:],
        }))
        if "credit balance" in msg or "AuthenticationError" in msg or "User not found" in msg:
            detail = "The tutor's AI service is temporarily unavailable. Please try again later."
        else:
            detail = "Something went wrong processing your question. Please try again."
        return JSONResponse(status_code=503, content={"detail": detail})

    session["history"] = result["conversation_history"]
    session["current_topic"] = result["current_topic"]
    session["current_topic_name"] = result["current_topic_name"]
    session["turn_count"] = result["turn_count"]
    session["ability_signal"] = result.get("ability_signal", session.get("ability_signal", 0))
    session["guidance_resistance_count"] = result.get(
        "guidance_resistance_count", session.get("guidance_resistance_count", 0)
    )

    latency_ms = round((time.monotonic() - t_start) * 1000)
    _metrics["total_turns"] += 1
    _metrics["total_latency_ms"] += latency_ms

    _json_logger.info(json.dumps({
        "ts": datetime.utcnow().isoformat(),
        "event": "turn",
        "learner_id": req.learner_id,
        "turn": session["turn_count"],
        "latency_ms": latency_ms,
        "topic": result["current_topic"],
        "mastery": round(result.get("mastery", 0.0), 3),
        "engagement_state": result["engagement_state"],
        "tokens_in": 0,
        "tokens_out": 0,
    }))

    return ChatResponse(
        response=result["agent_response"],
        engagement_state=result["engagement_state"],
        mastery=result.get("mastery", 0.0),
        citations=result.get("citations", [])[:2],
        topic=result["current_topic"],
    )


@app.get("/summary")
async def summary(learner_id: str = "demo-student-001"):
    """Return session summary stats."""
    if learner_id not in active_sessions:
        return {"error": "No active session"}
    session = active_sessions[learner_id]
    return {
        "session_id": session["session_id"],
        "learner_id": learner_id,
        "turn_count": session["turn_count"],
        "current_topic": session["current_topic"],
        "engagement_state": session["tracker"].state,
    }


# ── Inline HTML ───────────────────────────────────────────────────────────────

CHAT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MAS Tutor — GCSE Maths</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: #f0f4f8; height: 100vh; display: flex; flex-direction: column; }

  header { background: #1e40af; color: white; padding: 14px 20px;
           display: flex; align-items: center; justify-content: space-between; }
  header h1 { font-size: 1.1rem; font-weight: 600; }

  #status-bar { display: flex; gap: 12px; align-items: center; font-size: 0.8rem; }
  #engagement-badge { background: #3b82f6; color: white; border-radius: 12px;
                      padding: 3px 10px; font-weight: 600; }
  #mastery-wrap { display: flex; align-items: center; gap: 6px; }
  #mastery-bar { width: 80px; height: 8px; background: rgba(255,255,255,0.3);
                 border-radius: 4px; overflow: hidden; }
  #mastery-fill { height: 100%; background: #86efac; border-radius: 4px;
                  width: 0%; transition: width 0.4s ease; }
  #mastery-pct { font-size: 0.75rem; }

  #messages { flex: 1; overflow-y: auto; padding: 16px; display: flex;
              flex-direction: column; gap: 12px; }

  .bubble-wrap { display: flex; }
  .bubble-wrap.user  { justify-content: flex-end; }
  .bubble-wrap.tutor { justify-content: flex-start; }

  .bubble { max-width: 72%; padding: 10px 14px; border-radius: 16px;
            line-height: 1.5; font-size: 0.9rem; white-space: pre-wrap; }
  .bubble-wrap.user  .bubble { background: white; border-bottom-right-radius: 4px; }
  .bubble-wrap.tutor .bubble { background: #2563eb; color: white; border-bottom-left-radius: 4px; }

  .citations { font-size: 0.75rem; color: #93c5fd; margin-top: 5px; }

  .typing { display: flex; gap: 4px; align-items: center; padding: 10px 14px;
            background: #2563eb; border-radius: 16px; border-bottom-left-radius: 4px;
            width: fit-content; }
  .dot { width: 7px; height: 7px; background: white; border-radius: 50%;
         animation: bounce 1.2s infinite; }
  .dot:nth-child(2) { animation-delay: 0.2s; }
  .dot:nth-child(3) { animation-delay: 0.4s; }
  @keyframes bounce { 0%,60%,100% { transform: translateY(0); }
                      30% { transform: translateY(-6px); } }

  #input-area { padding: 12px 16px; background: white; border-top: 1px solid #e2e8f0;
                display: flex; gap: 10px; }
  #user-input { flex: 1; border: 1px solid #cbd5e1; border-radius: 8px;
                padding: 9px 13px; font-size: 0.9rem; outline: none; }
  #user-input:focus { border-color: #2563eb; box-shadow: 0 0 0 2px #bfdbfe; }
  #send-btn { background: #1e40af; color: white; border: none; border-radius: 8px;
              padding: 9px 18px; font-size: 0.9rem; cursor: pointer; font-weight: 600; }
  #send-btn:hover { background: #1d4ed8; }
  #send-btn:disabled { background: #93c5fd; cursor: not-allowed; }

  #ai-notice { background: #fef3c7; color: #78350f; font-size: 0.78rem;
               padding: 6px 16px; text-align: center; border-bottom: 1px solid #fde68a; }
  #ai-notice a { color: #78350f; font-weight: 600; }

  footer { background: #f8fafc; color: #64748b; font-size: 0.72rem;
           padding: 6px 16px; text-align: center; border-top: 1px solid #e2e8f0; }
  footer a { color: #475569; }
</style>
</head>
<body>

<header>
  <h1>MAS Tutor &mdash; GCSE Maths (Socratic)</h1>
  <div id="status-bar">
    <span id="engagement-badge">EXPLORING</span>
    <div id="mastery-wrap">
      <span>Mastery</span>
      <div id="mastery-bar"><div id="mastery-fill"></div></div>
      <span id="mastery-pct">0%</span>
    </div>
  </div>
</header>

<div id="ai-notice">
  You are chatting with an <strong>AI tutor</strong>, not a human. It can make mistakes &mdash;
  always check important answers with your teacher. See <a href="/terms">Terms</a> &amp;
  <a href="/privacy">Privacy</a>.
</div>

<div id="messages">
  <div class="bubble-wrap tutor">
    <div class="bubble">Hello! I'm your GCSE Maths tutor. What would you like to work on today?
    Try: <em>"I don't understand quadratic equations"</em> or ask me anything about maths.</div>
  </div>
</div>

<div id="input-area">
  <input id="user-input" type="text" placeholder="Ask a maths question… (type 'summary' for session stats)" autocomplete="off" />
  <button id="send-btn">Send</button>
</div>

<footer>
  MAS Tutor is an independent, personal proof-of-concept project &mdash; not an official Cambridge
  University Press &amp; Assessment product. &middot; <a href="/terms">Terms of Use</a> &middot;
  <a href="/privacy">Privacy Policy</a>
</footer>

<script>
const messagesEl = document.getElementById('messages');
const inputEl    = document.getElementById('user-input');
const sendBtn    = document.getElementById('send-btn');
const badge      = document.getElementById('engagement-badge');
const fill       = document.getElementById('mastery-fill');
const pct        = document.getElementById('mastery-pct');

// Unique learner ID — persists across page refreshes via sessionStorage
const learnerId = (function() {
  const key = 'mas_learner_id';
  let id = sessionStorage.getItem(key);
  if (!id) {
    id = crypto.randomUUID();
    sessionStorage.setItem(key, id);
  }
  return id;
})();

function scrollBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function addBubble(role, text, citations) {
  const wrap = document.createElement('div');
  wrap.className = `bubble-wrap ${role}`;
  const bub = document.createElement('div');
  bub.className = 'bubble';
  bub.textContent = text;
  wrap.appendChild(bub);
  if (citations && citations.length > 0) {
    const cit = document.createElement('div');
    cit.className = 'citations';
    cit.textContent = '↳ ' + citations.join(', ');
    wrap.appendChild(cit);
  }
  messagesEl.appendChild(wrap);
  scrollBottom();
  return wrap;
}

function showTyping() {
  const wrap = document.createElement('div');
  wrap.className = 'bubble-wrap tutor';
  wrap.id = 'typing-indicator';
  wrap.innerHTML = '<div class="typing"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>';
  messagesEl.appendChild(wrap);
  scrollBottom();
}

function removeTyping() {
  const t = document.getElementById('typing-indicator');
  if (t) t.remove();
}

function updateStatus(state, mastery) {
  badge.textContent = state || 'EXPLORING';
  const colours = {
    EXPLORING: '#3b82f6', STUCK: '#ef4444',
    BREAKTHROUGH: '#10b981', CONSOLIDATING: '#f59e0b', MASTERED: '#6d28d9'
  };
  badge.style.background = colours[state] || '#3b82f6';
  const m = Math.round((mastery || 0) * 100);
  fill.style.width = m + '%';
  pct.textContent = m + '%';
}

async function sendMessage() {
  const text = inputEl.value.trim();
  if (!text) return;

  // Handle summary locally via GET
  if (text.toLowerCase() === 'summary') {
    inputEl.value = '';
    const res = await fetch('/summary?learner_id=' + encodeURIComponent(learnerId));
    const data = await res.json();
    if (data.error) {
      addBubble('tutor', data.error);
    } else {
      const msg = `Session ID: ${data.session_id}\\nTurns: ${data.turn_count}\\nTopic: ${data.current_topic}\\nEngagement: ${data.engagement_state}`;
      addBubble('tutor', msg);
    }
    return;
  }

  inputEl.value = '';
  sendBtn.disabled = true;
  addBubble('user', text);
  showTyping();

  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, learner_id: learnerId })
    });
    const data = await res.json();
    removeTyping();
    if (res.ok) {
      addBubble('tutor', data.response, data.citations);
      updateStatus(data.engagement_state, data.mastery);
    } else {
      addBubble('tutor', 'Error: ' + (data.detail || 'Unknown error'));
    }
  } catch (err) {
    removeTyping();
    addBubble('tutor', 'Network error — is the server running?');
  } finally {
    sendBtn.disabled = false;
    inputEl.focus();
  }
}

sendBtn.addEventListener('click', sendMessage);
inputEl.addEventListener('keydown', e => { if (e.key === 'Enter') sendMessage(); });
inputEl.focus();
</script>
</body>
</html>
"""


_LEGAL_CSS = """
  * { box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: #f0f4f8; color: #1e293b; margin: 0; }
  header { background: #1e40af; color: white; padding: 14px 20px; }
  header h1 { font-size: 1.1rem; font-weight: 600; margin: 0; }
  main { max-width: 720px; margin: 0 auto; padding: 24px 20px 60px; line-height: 1.6; }
  h2 { font-size: 1.05rem; margin: 28px 0 8px; color: #1e40af; }
  h2:first-of-type { margin-top: 0; }
  p, li { font-size: 0.92rem; color: #334155; }
  ul { padding-left: 20px; }
  #updated { font-size: 0.8rem; color: #64748b; margin-top: 4px; }
  .back { display: inline-block; margin-top: 24px; font-size: 0.85rem; color: #1e40af; }
"""

TERMS_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Terms of Use — MAS Tutor</title>
<style>""" + _LEGAL_CSS + """</style>
</head>
<body>
<header><h1>MAS Tutor &mdash; Terms of Use</h1></header>
<main>
<p id="updated">Last updated: 17 July 2026</p>

<h2>1. What this is</h2>
<p>MAS Tutor is an independent, personal proof-of-concept project built by its developer to explore
AI-assisted GCSE Maths tutoring. It is <strong>not an official product of, and is not endorsed by,
Cambridge University Press &amp; Assessment</strong> or any exam board. It is a pilot/demonstration
tool, not a certified teaching product.</p>

<h2>2. You are talking to an AI</h2>
<p>This service uses a large language model &mdash; an artificial intelligence system &mdash; to
generate every tutoring response you see. <strong>You are at all times interacting with an AI
system, not a human tutor.</strong> No person reads or approves messages before they are sent to
you.</p>

<h2>3. No guarantee of accuracy</h2>
<p>AI systems can make mistakes, including on mathematical working and answers. Always check
important answers against your teacher, your textbook, or official mark schemes before relying on
this tool for coursework, revision, or exam preparation.</p>

<h2>4. Not a substitute for your teacher or school</h2>
<p>This tool does not replace your school's curriculum or a qualified teacher's judgement. Use it
alongside, not instead of, your normal maths teaching.</p>

<h2>5. Age of users and parental guidance</h2>
<p>This tool is aimed at GCSE-age students (typically 14&ndash;16). If you are under 18, we
recommend using it with a parent, guardian, or teacher's awareness. The developer does not verify
the age of users.</p>

<h2>6. Acceptable use</h2>
<p>Please don't use this tool to attempt to extract its underlying instructions or system prompts,
to abuse or overload the service, or to submit harmful or inappropriate content. Rate limits apply
and may result in temporary blocks.</p>

<h2>7. No warranty; limitation of liability</h2>
<p>This service is provided "as is", for pilot and demonstration purposes, without warranty of any
kind. The developer accepts no liability for outcomes arising from reliance on tutor output,
including in exams, coursework, or assessments.</p>

<h2>8. Changes to these terms</h2>
<p>These terms may change as the project develops. Check this page for the current version.</p>

<h2>9. Contact</h2>
<p>Questions about these terms: <a href="mailto:tony@thompson-starkey.co.uk">tony@thompson-starkey.co.uk</a></p>

<a class="back" href="/">&larr; Back to MAS Tutor</a>
</main>
</body>
</html>
"""

PRIVACY_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Privacy Policy — MAS Tutor</title>
<style>""" + _LEGAL_CSS + """</style>
</head>
<body>
<header><h1>MAS Tutor &mdash; Privacy Policy</h1></header>
<main>
<p id="updated">Last updated: 17 July 2026</p>

<h2>1. Overview</h2>
<p>MAS Tutor is a small, independent pilot project. This page explains what data it collects and
what happens to it, in plain language.</p>

<h2>2. What we collect</h2>
<ul>
  <li>A randomly generated session ID, stored only in your browser's <code>sessionStorage</code>
  &mdash; not a persistent account, and no sign-up is required.</li>
  <li>The text of the messages you send to the tutor.</li>
  <li>Timestamps, an engagement-state estimate, and a mastery estimate generated automatically
  from your session, used to run the tutoring logic.</li>
</ul>
<p>We do not ask for or collect your name, email address, date of birth, or any other contact
details.</p>

<h2>3. How your messages are processed</h2>
<p>Your messages are sent to third-party AI model providers (via OpenRouter, using Anthropic Claude
models) in order to generate a tutoring response. Your question is also matched against a maths
curriculum knowledge base (which contains no personal data) to ground the answer in the right
topic.</p>

<h2>4. Storage and retention</h2>
<p>Session data is stored server-side in a database so that a conversation and its mastery tracking
can continue while your session is active, and so the developer can review and improve the
tutoring pipeline. Data is not sold, and is not shared with advertisers or used for marketing.</p>

<h2>5. Cookies</h2>
<p>This tool does not use tracking cookies. Your session identifier lives only in your browser's
<code>sessionStorage</code> and is cleared automatically when you close the tab.</p>

<h2>6. Children's data</h2>
<p>Because this tool is aimed at GCSE-age students, it is designed to collect no more than the
anonymous session ID and message text described above &mdash; the tutor never asks for a name,
date of birth, school, or any other identifying contact detail.</p>

<h2>7. Your rights and contact</h2>
<p>You can ask for your session data to be deleted by emailing
<a href="mailto:tony@thompson-starkey.co.uk">tony@thompson-starkey.co.uk</a> and quoting your
session ID (visible by typing "summary" in the chat).</p>

<h2>8. Changes to this policy</h2>
<p>This policy may change as the project develops. Check this page for the current version.</p>

<a class="back" href="/">&larr; Back to MAS Tutor</a>
</main>
</body>
</html>
"""


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  MAS Education POC — Browser Chat UI")
    print("  Open http://localhost:8080 in your browser")
    print("  (WSL2: use Windows 11 Edge or Chrome)")
    print("  Ctrl+C to stop")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8080)
