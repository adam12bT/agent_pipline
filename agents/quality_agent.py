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

When explicitly enabled, optionally uses LLM Guard's:
  - `Toxicity`      -> tone/appropriateness scoring
  - `NoRefusal`      -> catches the generation model refusing / punting
                        instead of writing the section (a common failure
                        mode for generation agents)
plus the pre-existing template-compliance and word-count checks.

(PII / secrets / malicious-URL scanning moved to agents/security_agent.py
— see that module's docstring for why the split.)

LLM Guard is disabled by default for lightweight deployments. Groundedness,
coherence, template compliance, section order, and word-count checks remain
active independently of it.
"""

import json
import logging
import os
import re

from json_repair import loads as repair_json_loads

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
MAX_GENERATION_ATTEMPTS = max(
    1, int(os.environ.get("MAX_GENERATION_ATTEMPTS", "1"))
)
MIN_GROUNDEDNESS_SCORE = float(os.environ.get("QUALITY_MIN_GROUNDEDNESS", "0.75"))
MIN_COHERENCE_SCORE = float(os.environ.get("QUALITY_MIN_COHERENCE", "0.75"))
QUALITY_EVIDENCE_MAX_CHARS = min(
    6000,
    max(3000, int(os.environ.get("QUALITY_EVIDENCE_MAX_CHARS", "6000"))),
)
QUALITY_DRAFT_MAX_CHARS = min(
    6000,
    max(3000, int(os.environ.get("QUALITY_DRAFT_MAX_CHARS", "6000"))),
)
QUALITY_EVALUATION_BATCHES = max(
    1, int(os.environ.get("QUALITY_EVALUATION_BATCHES", "1"))
)
QUALITY_MAX_TOKENS = min(
    700,
    max(512, int(os.environ.get("QUALITY_MAX_TOKENS", "700"))),
)
QUALITY_LLM_MODEL = os.environ.get("QUALITY_LLM_MODEL", "").strip() or None
LLM_GUARD_FAIL_CLOSED = os.environ.get(
    "LLM_GUARD_FAIL_CLOSED", "true"
).strip().lower() not in {"0", "false", "no", "off"}
LLM_GUARD_ENABLED = os.environ.get(
    "LLM_GUARD_ENABLED", "false"
).strip().lower() in {"1", "true", "yes", "on"}

if LLM_GUARD_ENABLED:
    try:
        from llm_guard.output_scanners import NoRefusal, Toxicity

        _LLM_GUARD_AVAILABLE = True
    except ImportError:  # pragma: no cover - depends on deployment extras
        _LLM_GUARD_AVAILABLE = False
        logger.warning(
            "LLM Guard was enabled but is unavailable; toxicity/refusal "
            "model scanning will be skipped."
        )
else:
    _LLM_GUARD_AVAILABLE = False
    logger.info("LLM Guard is disabled; toxicity/refusal model scanning is skipped.")

_scanners_cache = None


def llm_guard_available() -> bool:
    return _LLM_GUARD_AVAILABLE


def _get_scanners():
    global _scanners_cache
    if _scanners_cache is None:
        _scanners_cache = {
            "toxicity": Toxicity(threshold=0.7),
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


def _canonical_section_title(value: str) -> str:
    """Normalize client numbering and Markdown decoration for comparison."""
    title = str(value).strip().casefold()
    title = re.sub(r"^\s{0,3}#{1,6}\s*", "", title)
    title = re.sub(r"[*_`]", "", title)
    title = re.sub(r"^\s*(?:section\s+)?\d+(?:\.\d+)*[.)\-:]?\s*", "", title)
    return re.sub(r"\s+", " ", title).strip(" :-–—")


def _section_positions(draft: str, sections: list[str]) -> list[int]:
    normalized_lines = [_canonical_section_title(line) for line in draft.splitlines()]
    positions = []
    for section in sections:
        target = _canonical_section_title(section)
        position = next(
            (
                index
                for index, line in enumerate(normalized_lines)
                if line == target or line.startswith(f"{target}:")
            ),
            -1,
        )
        positions.append(position)
    return positions


def _check_template_compliance(draft: str, required_sections: list[str]) -> list[str]:
    positions = _section_positions(draft, required_sections)
    return [section for section, position in zip(required_sections, positions) if position < 0]


def _check_section_order(draft: str, section_order: list[str]) -> list[str]:
    positions = _section_positions(draft, section_order)
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
    if not candidate:
        raise ValueError("Grounding evaluator returned an empty response")
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, re.DOTALL)
    if fence:
        candidate = fence.group(1)
    else:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0:
            candidate = candidate[start : end + 1] if end > start else candidate[start:]

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        logger.warning(
            "Quality evaluator returned malformed JSON; attempting local repair: %s",
            exc,
        )
        parsed = repair_json_loads(candidate)
    if not isinstance(parsed, dict):
        raise ValueError("Grounding evaluator returned JSON that is not an object")

    parsed["groundedness_score"] = _score(parsed.get("groundedness_score"))
    parsed["coherence_score"] = _score(parsed.get("coherence_score"))
    for field in ("unsupported_claims", "contradictions", "coherence_issues", "notes"):
        if not isinstance(parsed.get(field), list):
            parsed[field] = []
    return parsed


def _empty_review(*, error: str | None = None) -> dict:
    review = {
        "groundedness_score": 0.0,
        "coherence_score": 0.0,
        "unsupported_claims": [],
        "contradictions": [],
        "coherence_issues": [],
        "notes": [],
    }
    if error:
        review["evaluation_error"] = error
    return review


def _review_groups(state: dict, draft: str) -> list[tuple[dict, str]]:
    """Build a small fixed number of matching evidence/draft review groups."""
    evidence = state.get("generation_evidence") or {}
    section_batches = evidence.get("section_batches") or []
    usable = [
        batch
        for batch in section_batches
        if isinstance(batch, dict) and str(batch.get("draft", "")).strip()
    ]
    if not usable:
        return [(evidence, draft)]

    group_count = min(QUALITY_EVALUATION_BATCHES, len(usable))
    groups = []
    group_size = (len(usable) + group_count - 1) // group_count
    for group_index in range(group_count):
        start = group_index * group_size
        selected = usable[start : start + group_size]
        if not selected:
            continue
        group_evidence = {
            "requirements": evidence.get("requirements", {}),
            "research_summary": evidence.get("research_summary", ""),
            "project_references": evidence.get("project_references", ""),
            "cv_excerpts": evidence.get("cv_excerpts", ""),
            "past_proposals": evidence.get("past_proposals", ""),
            "section_batches": [
                {key: value for key, value in batch.items() if key != "draft"}
                for batch in selected
            ],
        }
        group_draft = "\n\n".join(str(batch["draft"]) for batch in selected)
        groups.append((group_evidence, group_draft))
    return groups


def _merge_reviews(reviews: list[dict]) -> dict:
    merged = _empty_review()
    merged["groundedness_score"] = min(
        review["groundedness_score"] for review in reviews
    )
    merged["coherence_score"] = min(
        review["coherence_score"] for review in reviews
    )
    for field in ("unsupported_claims", "contradictions", "coherence_issues", "notes"):
        merged[field] = [
            item for review in reviews for item in review.get(field, [])
        ]
    errors = [
        review.get("evaluation_error")
        for review in reviews
        if review.get("evaluation_error")
    ]
    if errors:
        merged["evaluation_error"] = "; ".join(errors)[:500]
    merged["evaluation_batches"] = len(reviews)
    return merged


def _evaluate_grounding_and_coherence(state: dict, draft: str) -> dict:
    if not draft.strip():
        review = _empty_review(error="No draft was available for evaluation.")
        review["coherence_issues"] = ["The generated proposal is empty."]
        return review

    evidence = state.get("generation_evidence") or {}
    if not evidence:
        review = _empty_review(
            error="Cannot evaluate grounding without generation evidence."
        )
        review["coherence_issues"] = ["Generation evidence was not preserved."]
        return review

    provider = get_provider()
    reviews = []
    for batch_number, (batch_evidence, batch_draft) in enumerate(
        _review_groups(state, draft), start=1
    ):
        evidence_text = json.dumps(batch_evidence, ensure_ascii=False, default=str)
        prompt = QUALITY_GROUNDING_PROMPT_TEMPLATE.format(
            evidence=evidence_text[:QUALITY_EVIDENCE_MAX_CHARS],
            draft=batch_draft[:QUALITY_DRAFT_MAX_CHARS],
        )
        try:
            completion_options = {
                "temperature": 0.0,
                "max_tokens": QUALITY_MAX_TOKENS,
                "model": QUALITY_LLM_MODEL,
                "reasoning_effort": "low",
                "include_reasoning": False,
            }
            try:
                response = provider.complete(
                    prompt,
                    **completion_options,
                    response_format={"type": "json_object"},
                    request_label=f"quality.batch_{batch_number}",
                )
            except Exception as exc:
                # Groq can reject a model-generated response in strict JSON
                # mode before returning any text. There is then nothing for
                # our local JSON repair step to process. Retry exactly once
                # without strict mode and repair/validate the returned text.
                if "json_validate_failed" not in str(exc).casefold():
                    raise
                logger.warning(
                    "Groq strict JSON validation failed for quality batch %d; "
                    "retrying once in text mode",
                    batch_number,
                )
                response = provider.complete(
                    prompt,
                    **completion_options,
                    request_label=f"quality.batch_{batch_number}.json_fallback",
                )
            reviews.append(_extract_review_json(response))
        except Exception as exc:
            logger.exception(
                "Grounding/coherence evaluation batch %d failed", batch_number
            )
            reviews.append(_empty_review(error=str(exc)[:500]))
            break
    return _merge_reviews(reviews)


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
    evaluator_error = bool(grounding_review.get("evaluation_error"))
    grounding_failed = (
        evaluator_error
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
    if grounding_failed and not evaluator_error:
        notes.append(
            "Grounding/coherence review failed: "
            f"groundedness={groundedness_score:.2f} "
            f"(minimum {MIN_GROUNDEDNESS_SCORE:.2f}), "
            f"coherence={coherence_score:.2f} "
            f"(minimum {MIN_COHERENCE_SCORE:.2f})."
        )
    if evaluator_error:
        notes.append(
            "Quality evaluator unavailable; stopping without regenerating the proposal."
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
        "evaluation_available": not evaluator_error,
        "groundedness_threshold": MIN_GROUNDEDNESS_SCORE,
        "coherence_threshold": MIN_COHERENCE_SCORE,
        "notes": notes,
    }

    attempts = state.get("generation_attempts", 0)
    if evaluator_error:
        status = "failed"
        logger.error(
            "Quality evaluator failed; stopping without proposal regeneration: %s",
            grounding_review.get("evaluation_error"),
        )
    elif not passed and attempts < MAX_GENERATION_ATTEMPTS:
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
