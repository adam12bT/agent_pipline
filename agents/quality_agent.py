"""
Quality Agent
--------------
Runs after Security — but only if security passed (see the guard below;
the graph itself already routes a failed security check straight to END,
so this is a defensive double-check, not the primary gate).

Checks the draft's QUALITY, not its safety: source groundedness, coherence,
template compliance, length, tone, and refusal detection. The evidence review
uses the exact RAG/research context preserved by the Generation agent. Unlike
the Security agent, a failure here is
GRADED — it triggers a regeneration attempt (up to MAX_GENERATION_ATTEMPTS)
via the "retry_generation" status rather than a hard stop.

Uses LLM Guard's:
  - `ToxicLanguage` -> tone/appropriateness scoring
  - `NoRefusal`      -> catches the generation model refusing / punting
                        instead of writing the section (a common failure
                        mode for generation agents)
plus the pre-existing template-compliance and word-count checks.

(PII / secrets / malicious-URL scanning moved to agents/security_agent.py
— see that module's docstring for why the split.)

Install:
    pip install llm-guard
"""

import json
import logging
import os
import re

from agents.prompts import QUALITY_GROUNDING_PROMPT_TEMPLATE
from providers import get_provider

logger = logging.getLogger(__name__)

REQUIRED_SECTIONS = [
    "Executive Summary",
    "Understanding of the Requirements",
    "Proposed Approach",
    "Work Plan",
    "Proposed Team",
    "Why Us",
]

MIN_WORD_COUNT = 150
MAX_GENERATION_ATTEMPTS = 3
MIN_GROUNDEDNESS_SCORE = float(os.environ.get("QUALITY_MIN_GROUNDEDNESS", "0.75"))
MIN_COHERENCE_SCORE = float(os.environ.get("QUALITY_MIN_COHERENCE", "0.75"))
QUALITY_EVIDENCE_MAX_CHARS = max(
    5000, int(os.environ.get("QUALITY_EVIDENCE_MAX_CHARS", "40000"))
)
QUALITY_DRAFT_MAX_CHARS = max(
    5000, int(os.environ.get("QUALITY_DRAFT_MAX_CHARS", "30000"))
)
QUALITY_LLM_MODEL = os.environ.get("QUALITY_LLM_MODEL", "").strip() or None
LLM_GUARD_FAIL_CLOSED = os.environ.get(
    "LLM_GUARD_FAIL_CLOSED", "true"
).strip().lower() not in {"0", "false", "no", "off"}

try:
    from llm_guard.output_scanners import NoRefusal, ToxicLanguage

    _LLM_GUARD_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when dep is missing
    _LLM_GUARD_AVAILABLE = False
    logger.warning(
        "llm-guard not installed — skipping toxicity/refusal scanning. "
        "Run `pip install llm-guard` to enable it."
    )

_scanners_cache = None


def llm_guard_available() -> bool:
    return _LLM_GUARD_AVAILABLE


def _get_scanners():
    global _scanners_cache
    if _scanners_cache is None:
        _scanners_cache = {
            "toxicity": ToxicLanguage(threshold=0.7),
            "no_refusal": NoRefusal(threshold=0.5),
        }
    return _scanners_cache


def _run_llm_guard(draft: str) -> dict:
    scanners = _get_scanners()
    findings = {}
    for name, scanner in scanners.items():
        try:
            _, is_valid, risk_score = scanner.scan(prompt="", output=draft)
        except Exception as exc:
            logger.exception("LLM Guard scanner %r failed", name)
            if LLM_GUARD_FAIL_CLOSED:
                findings[f"{name}_scanner_error"] = str(exc)[:300]
            continue
        if not is_valid:
            findings[name] = round(risk_score, 3)
    return findings


def _template_sections(state: dict) -> tuple[list[str], list[str]]:
    requirements = state.get("requirements") or {}
    template = requirements.get("response_template") or {}
    if not isinstance(template, dict):
        return REQUIRED_SECTIONS, REQUIRED_SECTIONS

    raw_required = template.get("required_sections") or []
    raw_ordered = template.get("section_order") or []
    if not isinstance(raw_required, list):
        raw_required = []
    if not isinstance(raw_ordered, list):
        raw_ordered = []

    required = [
        str(section).strip()
        for section in raw_required
        if str(section).strip()
    ]
    ordered = [
        str(section).strip()
        for section in raw_ordered
        if str(section).strip()
    ]
    return required or REQUIRED_SECTIONS, ordered or required or REQUIRED_SECTIONS


def _check_template_compliance(draft: str, required_sections: list[str]) -> list[str]:
    return [section for section in required_sections if section.lower() not in draft.lower()]


def _check_section_order(draft: str, section_order: list[str]) -> list[str]:
    positions = [draft.lower().find(section.lower()) for section in section_order]
    present_positions = [position for position in positions if position >= 0]
    if len(present_positions) < 2 or present_positions == sorted(present_positions):
        return []
    return section_order


def _score(value) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, score))


def _extract_review_json(text: str) -> dict:
    candidate = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, re.DOTALL)
    if fence:
        candidate = fence.group(1)
    else:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            candidate = candidate[start : end + 1]

    parsed = json.loads(candidate)
    if not isinstance(parsed, dict):
        raise ValueError("Grounding evaluator returned JSON that is not an object")

    parsed["groundedness_score"] = _score(parsed.get("groundedness_score"))
    parsed["coherence_score"] = _score(parsed.get("coherence_score"))
    for field in ("unsupported_claims", "contradictions", "coherence_issues", "notes"):
        if not isinstance(parsed.get(field), list):
            parsed[field] = []
    return parsed


def _evaluate_grounding_and_coherence(state: dict, draft: str) -> dict:
    if not draft.strip():
        return {
            "groundedness_score": 0.0,
            "coherence_score": 0.0,
            "unsupported_claims": [],
            "contradictions": [],
            "coherence_issues": ["The generated proposal is empty."],
            "notes": [],
            "evaluation_error": "No draft was available for evaluation.",
        }

    evidence = state.get("generation_evidence") or {}
    if not evidence:
        return {
            "groundedness_score": 0.0,
            "coherence_score": 0.0,
            "unsupported_claims": [],
            "contradictions": [],
            "coherence_issues": ["Generation evidence was not preserved."],
            "notes": [],
            "evaluation_error": "Cannot evaluate grounding without generation evidence.",
        }

    evidence_text = json.dumps(evidence, ensure_ascii=False, default=str)
    prompt = QUALITY_GROUNDING_PROMPT_TEMPLATE.format(
        evidence=evidence_text[:QUALITY_EVIDENCE_MAX_CHARS],
        draft=draft[:QUALITY_DRAFT_MAX_CHARS],
    )
    try:
        response = get_provider().complete(
            prompt,
            temperature=0.0,
            max_tokens=1800,
            model=QUALITY_LLM_MODEL,
        )
        return _extract_review_json(response)
    except Exception as exc:
        logger.exception("Grounding/coherence evaluation failed")
        return {
            "groundedness_score": 0.0,
            "coherence_score": 0.0,
            "unsupported_claims": [],
            "contradictions": [],
            "coherence_issues": [],
            "notes": [],
            "evaluation_error": str(exc)[:500],
        }


def quality_agent(state: dict) -> dict:
    if not state.get("is_verified"):
        return {}
    if not state.get("security_passed", True):
        # Defensive — the graph should never route here on a security
        # failure, but don't silently score a blocked draft if it does.
        return {}

    draft = state.get("draft_proposal", "")
    word_count = len(draft.split())
    required_sections, section_order = _template_sections(state)
    missing_sections = _check_template_compliance(draft, required_sections)
    out_of_order_sections = _check_section_order(draft, section_order)
    quality_findings = _run_llm_guard(draft) if _LLM_GUARD_AVAILABLE else {}
    grounding_review = _evaluate_grounding_and_coherence(state, draft)
    groundedness_score = grounding_review["groundedness_score"]
    coherence_score = grounding_review["coherence_score"]
    grounding_failed = (
        bool(grounding_review.get("evaluation_error"))
        or groundedness_score < MIN_GROUNDEDNESS_SCORE
        or coherence_score < MIN_COHERENCE_SCORE
        or bool(grounding_review.get("contradictions"))
    )

    notes = []
    if word_count < MIN_WORD_COUNT:
        notes.append(f"Draft is short ({word_count} words) — may be incomplete.")
    if missing_sections:
        notes.append(f"Missing expected sections: {missing_sections}")
    if out_of_order_sections:
        notes.append(f"Template sections are out of order: {out_of_order_sections}")
    if quality_findings:
        notes.append(f"LLM Guard flagged: {quality_findings}")
    if grounding_failed:
        notes.append(
            "Grounding/coherence review failed: "
            f"groundedness={groundedness_score:.2f} "
            f"(minimum {MIN_GROUNDEDNESS_SCORE:.2f}), "
            f"coherence={coherence_score:.2f} "
            f"(minimum {MIN_COHERENCE_SCORE:.2f})."
        )
    if grounding_review.get("unsupported_claims"):
        notes.append(
            f"Unsupported claims: {grounding_review['unsupported_claims']}"
        )
    if grounding_review.get("contradictions"):
        notes.append(f"Contradictions: {grounding_review['contradictions']}")

    passed = (
        word_count >= MIN_WORD_COUNT
        and not missing_sections
        and not out_of_order_sections
        and not quality_findings
        and not grounding_failed
    )

    quality_report = {
        "word_count": word_count,
        "missing_sections": missing_sections,
        "out_of_order_sections": out_of_order_sections,
        "required_sections": required_sections,
        "quality_findings": quality_findings,
        "grounding_review": grounding_review,
        "groundedness_threshold": MIN_GROUNDEDNESS_SCORE,
        "coherence_threshold": MIN_COHERENCE_SCORE,
        "notes": notes,
    }

    attempts = state.get("generation_attempts", 0)
    if not passed and attempts < MAX_GENERATION_ATTEMPTS:
        status = "retry_generation"
        logger.info(
            "Quality check failed (attempt %d/%d) — retrying generation. Notes: %s",
            attempts, MAX_GENERATION_ATTEMPTS, notes,
        )
    elif not passed:
        status = "failed"
        logger.warning(
            "Quality check failed after %d attempts — giving up. Notes: %s", attempts, notes,
        )
    else:
        status = "done"
        logger.info("Quality check passed.")

    return {
        "quality_passed": passed,
        "quality_report": quality_report,
        "status": status,
    }
