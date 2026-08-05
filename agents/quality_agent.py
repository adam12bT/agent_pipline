"""
Quality Agent
--------------
Runs after Security — but only if security passed (see the guard below;
the graph itself already routes a failed security check straight to END,
so this is a defensive double-check, not the primary gate).

Checks the draft's QUALITY, not its safety: template compliance, length,
tone, refusal detection. Unlike the Security agent, a failure here is
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

import logging

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
            logger.warning("LLM Guard scanner %r failed, skipping: %s", name, exc)
            continue
        if not is_valid:
            findings[name] = round(risk_score, 3)
    return findings


def _check_template_compliance(draft: str) -> list[str]:
    return [s for s in REQUIRED_SECTIONS if s.lower() not in draft.lower()]


def quality_agent(state: dict) -> dict:
    if not state.get("is_verified"):
        return {}
    if not state.get("security_passed", True):
        # Defensive — the graph should never route here on a security
        # failure, but don't silently score a blocked draft if it does.
        return {}

    draft = state.get("draft_proposal", "")
    word_count = len(draft.split())
    missing_sections = _check_template_compliance(draft)
    quality_findings = _run_llm_guard(draft) if _LLM_GUARD_AVAILABLE else {}

    notes = []
    if word_count < MIN_WORD_COUNT:
        notes.append(f"Draft is short ({word_count} words) — may be incomplete.")
    if missing_sections:
        notes.append(f"Missing expected sections: {missing_sections}")
    if quality_findings:
        notes.append(f"LLM Guard flagged: {quality_findings}")

    passed = word_count >= MIN_WORD_COUNT and not missing_sections and not quality_findings

    quality_report = {
        "word_count": word_count,
        "missing_sections": missing_sections,
        "quality_findings": quality_findings,
        "notes": notes,
    }

    attempts = state.get("generation_attempts", 0)
    if not passed and attempts < MAX_GENERATION_ATTEMPTS:
        status = "retry_generation"
    elif not passed:
        status = "failed"
    else:
        status = "done"

    return {
        "quality_passed": passed,
        "quality_report": quality_report,
        "status": status,
    }