"""
Tutor Agent — MAS Education POC
Constitutional Principle C4: EVERY response must be RAG-grounded.
Uses Socratic method — asks questions, doesn't give answers directly.
"""
import os
import json
from datetime import datetime
from anthropic import Anthropic
from rag.retrieve import retrieve, format_context
from rag.math_verify import verify_tutor_claims
from db.db import log_audit, get_conn
from compat.litellm_client import tag_current_trace

LLM_MODEL = os.getenv("LLM_PRIMARY", "claude-sonnet-4-6")
LLM_HAIKU = os.getenv("LLM_SECONDARY", "claude-haiku-4-5-20251001")
MAX_TOKENS = 500  # v2.0 §6.3: 500 token limit for tutoring responses

# Improvement 1: Patterns that indicate internal diagnostic text leaked into a response.
# These strings must never reach the student — strip any sentence containing them.
_METACOMMENTARY_PATTERNS = (
    "student appears",
    "the student has",
    "student is discussing",
    "rather than identifying",
    "student confuses",
    "student is confusing",
    "the student appears",
    "let's look at this differently —",
    "let me reconsider",
    "different topic",
    "different from the",
)


def _strip_metacommentary(text: str) -> str:
    """Remove any sentence containing internal diagnostic phrases. Returns cleaned text."""
    lines = text.split("\n")
    clean = []
    for line in lines:
        lowered = line.lower()
        if any(p in lowered for p in _METACOMMENTARY_PATTERNS):
            continue
        clean.append(line)
    result = "\n".join(clean).strip()
    # If stripping removed everything, return a safe fallback prompt
    return result if result else "What part of this topic would you like to explore?"


def _select_model(student_input: str) -> str:
    """Improvement 6: Use haiku for short, symbol-free messages to reduce latency.
    Only activates when LLM_SECONDARY is explicitly set in the environment —
    falls back to primary model if not configured."""
    secondary = os.getenv("LLM_SECONDARY", "")
    if not secondary or secondary == LLM_MODEL:
        return LLM_MODEL
    has_math = any(c in student_input for c in ["=", "^", "+", "-", "*", "/", "²", "√", "∫", "∑"])
    if len(student_input) < 60 and not has_math:
        return secondary
    return LLM_MODEL

TUTOR_SYSTEM_PROMPT = """You are a Socratic mathematics tutor for GCSE students (ages 14-16).

ABSOLUTE RULES:
1. You MUST only use information from the CURRICULUM CONTEXT provided below to answer.
2. If the context does not contain enough information, say: "Let me check that with your teacher."
3. Never give the answer directly — guide the student to discover it through questions.
4. Ask ONE question at a time. Never ask multiple questions in one response.
5. Use encouraging language. Celebrate effort, not just correct answers.
6. Keep responses concise (under 150 words).
7. NEVER narrate your reasoning or emit meta-commentary. Do not write anything like "Student confuses X with Y", "Let me reconsider", or "Looking at this differently —". Output only what you would say directly to the student.

TEACHING APPROACH:
- Start with what the student already knows
- Break complex problems into small steps
- Use worked examples from the curriculum context
- When a student is stuck, ask "What do you know about...?" before explaining

CITATION REQUIREMENT:
End every factual response with: "Source: [citation label from context]"
"""


async def get_tutor_response(
    student_input: str,
    conversation_history: list[dict],
    session_id: str,
    learner_id: str,
    current_topic: str = None,
    lo_id: str = None,
    anthropic_client: Anthropic = None
) -> dict:
    """
    Generate a RAG-grounded Socratic tutor response.

    Returns:
    {
        "response": str,
        "rag_chunks_used": list[dict],
        "citations": list[str],
        "grounded": bool
    }
    """
    if anthropic_client is None:
        anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # C4: Retrieve grounded context FIRST — mandatory
    # Phase 5: pass lo_id for KG-guided chunk prioritisation
    # Fix: pass the last tutor message as context_hint so short student replies
    # (e.g. "I think it is the vector?") are interpreted in conversation context.
    last_tutor_msg = next(
        (m["content"] for m in reversed(conversation_history) if m.get("role") == "assistant"),
        None
    )
    rag_results = await retrieve(
        query=student_input,
        subject="mathematics",
        topic=current_topic,
        lo_id=lo_id,
        context_hint=last_tutor_msg,
        anthropic_client=anthropic_client
    )

    rag_context = format_context(rag_results)
    citations = [r["citation_label"] for r in rag_results]
    grounded = len(rag_results) > 0

    if not grounded:
        # RAG unavailable (collection empty or all chunks below threshold).
        # For the testing phase, fall back to LLM-only Socratic tutoring rather than
        # blocking. This allows end-to-end tutor evaluation before the ChromaDB
        # ingest pipeline is run on the VPS.
        log_audit(session_id, learner_id, "tutor-agent",
                  "rag_fallback_llm", "no RAG results — using LLM knowledge (testing mode)",
                  {"query": student_input, "topic": current_topic})
        _fallback_system = (
            "You are a Socratic mathematics tutor for Cambridge IGCSE students (ages 14-16). "
            "Use the Socratic method — guide the student to discover answers through questions "
            "rather than telling them directly. Ask ONE question at a time. "
            "Keep responses under 150 words. Use encouraging language. "
            "NEVER give the answer directly — always ask a guiding question instead. "
            "NEVER narrate your reasoning or emit meta-commentary. Output only what you would say directly to the student."
        )
        fallback_messages = list(conversation_history[-6:])
        fallback_messages.append({"role": "user", "content": student_input})
        try:
            fb_response = await anthropic_client.messages.create(
                model=_select_model(student_input),  # Improvement 6: haiku for short messages
                max_tokens=MAX_TOKENS,
                system=_fallback_system,
                messages=fallback_messages,
            )
            response_text = _strip_metacommentary(fb_response.content[0].text)  # Improvement 1
        except Exception:
            response_text = "I'm having trouble with that one — could you try asking me a specific maths question?"
        return {
            "response": response_text,
            "rag_chunks_used": [],
            "citations": [],
            "grounded": False,
        }

    # Build messages with RAG context injected
    messages = list(conversation_history[-6:])  # last 3 turns (6 messages)
    messages.append({
        "role": "user",
        "content": f"""CURRICULUM CONTEXT (use ONLY this information):
---
{rag_context}
---

STUDENT: {student_input}

Remember: Ask ONE guiding question. Don't give the answer directly. Cite your source."""
    })

    response = await anthropic_client.messages.create(
        model=LLM_MODEL,
        max_tokens=MAX_TOKENS,
        system=TUTOR_SYSTEM_PROMPT,
        messages=messages
    )
    # Links this MLflow trace to the matching audit_events/session_log row —
    # no-op if not running with LLM_GATEWAY=litellm. See compat/litellm_client.py.
    tag_current_trace(session_id=session_id, learner_id=learner_id,
                       agent="tutor-agent", grounded=str(grounded), topic=current_topic or "")

    response_text = _strip_metacommentary(response.content[0].text)  # Improvement 1

    # Phase B: Verify Tutor's assertive math claims before delivering to student.
    # Conservative: only checks assertive language ("so x = N", "therefore x = N").
    # Exploratory Socratic suggestions ("try x = N") are intentionally excluded.
    math_check = verify_tutor_claims(response_text, student_input)
    math_verified = math_check.verified   # True / False / None
    math_corrected = False

    if math_verified is False:
        # Tutor stated a provably wrong value — attempt one correction regeneration
        try:
            correction_messages = list(messages) + [
                {"role": "assistant", "content": response_text},
                {"role": "user", "content": (
                    "[INTERNAL NOTE: Your previous response appears to contain a mathematical "
                    "error in a calculation or substitution result. Please carefully re-check "
                    "any arithmetic in your response before re-generating.]"
                )}
            ]
            correction_response = await anthropic_client.messages.create(
                model=LLM_MODEL,
                max_tokens=MAX_TOKENS,
                system=TUTOR_SYSTEM_PROMPT,
                messages=correction_messages
            )
            response_text = correction_response.content[0].text
            math_corrected = True
        except Exception as e:
            print(f"  Tutor correction regeneration failed ({e}), using original response")
            # Fail-safe: keep original response, log but don't crash

    # Audit log (C6)
    log_audit(
        session_id=session_id,
        learner_id=learner_id,
        agent="tutor-agent",
        decision="generate_response",
        reasoning=f"RAG-grounded response for: {student_input[:80]}",
        technical={
            "model": LLM_MODEL,
            "rag_chunks": len(rag_results),
            "citations": citations,
            "topic": current_topic,
            "tokens_used": response.usage.output_tokens,
            "math_verified": math_verified,
            "math_corrected": math_corrected
        }
    )

    return {
        "response": response_text,
        "rag_chunks_used": rag_results,
        "citations": citations,
        "grounded": True
    }
