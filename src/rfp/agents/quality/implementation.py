"""
Quality Agent Implementation
--------------
Runs after Security — but only if security passed (see the guard below;
the graph itself already routes a failed security check straight to END,
so this is a defensive double-check, not the primary gate).

Checks the draft's QUALITY, not its safety: source groundedness, coherence,
template compliance, length, tone, and refusal detection. The evidence review
uses the exact RAG/research context preserved by the Generation agent. Unlike
the Security agent, it returns a verdict and report only. Retry and terminal
status policy belongs exclusively to the orchestrator.

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

from rfp.prompts import QUALITY_GROUNDING_PROMPT_TEMPLATE
from rfp.default_template import resolve_response_template
from providers import get_provider

logger = logging.getLogger(__name__)

# A non-empty draft is the only universal length rule. Section count and order
# come from the uploaded template or the canonical fallback template.
MIN_WORD_COUNT = 1
MIN_SECTION_BODY_WORDS = max(
    1, int(os.environ.get("QUALITY_MIN_SECTION_BODY_WORDS", "12"))
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
    template = resolve_response_template(requirements)

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
    return required, ordered or required


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


def _duplicate_sections(draft: str, sections: list[str]) -> list[str]:
    """Return template sections whose Markdown heading occurs more than once."""
    heading_titles = [
        _canonical_section_title(line)
        for line in draft.splitlines()
        if re.match(r"^\s{0,3}#{1,2}\s+", line)
    ]
    return [
        section
        for section in sections
        if heading_titles.count(_canonical_section_title(section)) > 1
    ]


def _section_blocks(draft: str, sections: list[str]) -> dict[str, str]:
    """Split a proposal into template sections for localized repair decisions."""
    lines = draft.splitlines()
    starts: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        if not re.match(r"^\s{0,3}#{1,2}\s+", line):
            continue
        normalized = _canonical_section_title(line)
        matched = next(
            (
                section
                for section in sections
                if normalized == _canonical_section_title(section)
            ),
            None,
        )
        if matched and matched not in {section for _, section in starts}:
            starts.append((index, matched))

    blocks = {}
    for position, (start, section) in enumerate(starts):
        end = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
        blocks[section] = "\n".join(lines[start:end]).strip()
    return blocks


def _section_body(block: str) -> str:
    lines = block.strip().splitlines()
    if lines and re.match(r"^\s{0,3}#{1,6}\s+", lines[0]):
        lines = lines[1:]
    return "\n".join(lines).strip()


def _insubstantial_sections(draft: str, sections: list[str]) -> list[str]:
    """Detect present headings that contain no meaningful proposal body."""
    blocks = _section_blocks(draft, sections)
    failed = []
    for section in sections:
        block = blocks.get(section, "")
        body = _section_body(block)
        body_words = len(re.findall(r"\b[\w'-]+\b", body, flags=re.UNICODE))
        if block and body_words < MIN_SECTION_BODY_WORDS:
            failed.append(section)
    return failed


def _claim_text(finding) -> str:
    if isinstance(finding, dict):
        return str(finding.get("claim") or finding.get("text") or "").strip()
    return str(finding or "").strip()


def _sections_containing_claim(
    draft: str,
    sections: list[str],
    finding,
) -> list[str]:
    claim = re.sub(r"\s+", " ", _claim_text(finding).casefold()).strip()
    if len(claim) < 12:
        return []
    blocks = _section_blocks(draft, sections)
    return [
        section
        for section, block in blocks.items()
        if claim in re.sub(r"\s+", " ", block.casefold())
    ]


def _identify_failed_sections(
    *,
    draft: str,
    sections: list[str],
    missing_sections: list[str],
    out_of_order_sections: list[str],
    quality_findings: dict,
    grounding_review: dict,
    word_count: int,
    duplicate_sections: list[str] | None = None,
    incomplete_sections: list[str] | None = None,
) -> list[str]:
    """Map quality failures to the smallest safe set of template sections."""
    failed = (
        set(missing_sections)
        | set(out_of_order_sections)
        | set(duplicate_sections or [])
        | set(incomplete_sections or [])
    )
    unmapped_contradiction = False

    for field in ("unsupported_claims", "contradictions"):
        for finding in grounding_review.get(field, []) or []:
            matches = _sections_containing_claim(draft, sections, finding)
            failed.update(matches)
            if field == "contradictions" and not matches:
                unmapped_contradiction = True

    if word_count < MIN_WORD_COUNT:
        failed.update(sections)

    score_failed = (
        grounding_review.get("groundedness_score", 0.0) < MIN_GROUNDEDNESS_SCORE
        or grounding_review.get("coherence_score", 0.0) < MIN_COHERENCE_SCORE
    )
    if quality_findings or unmapped_contradiction or (score_failed and not failed):
        failed.update(sections)

    return [section for section in sections if section in failed]


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


def _relevant_evidence_excerpt(source, draft: str, max_chars: int) -> str:
    """Select verbatim evidence blocks that overlap most with the draft."""
    text = str(source or "").strip()
    if not text or len(text) <= max_chars:
        return text

    draft_terms = {
        term
        for term in re.findall(r"[a-z0-9]+", draft.casefold())
        if len(term) >= 4
    }
    blocks = [
        block.strip()
        for block in re.split(r"\n{2,}|(?=<document_metadata>)", text)
        if block.strip()
    ]
    ranked = sorted(
        enumerate(blocks),
        key=lambda item: (
            -len(
                draft_terms
                & {
                    term
                    for term in re.findall(r"[a-z0-9]+", item[1].casefold())
                    if len(term) >= 4
                }
            ),
            item[0],
        ),
    )

    selected = []
    used = 0
    for _, block in ranked:
        remaining = max_chars - used
        if remaining <= 0:
            break
        excerpt = block if len(block) <= remaining else block[:remaining]
        selected.append(excerpt)
        used += len(excerpt) + 2
    return "\n\n".join(selected)[:max_chars]


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

    # Pack consecutive section drafts without exceeding the reviewer input cap.
    # The configured batch count can request smaller groups, but never groups
    # large enough to silently discard later sections.
    total_draft_chars = (
        sum(len(str(batch.get("draft", ""))) for batch in usable)
        + max(0, len(usable) - 1) * 2
    )
    configured_target = (
        total_draft_chars + QUALITY_EVALUATION_BATCHES - 1
    ) // QUALITY_EVALUATION_BATCHES
    target_group_chars = min(
        QUALITY_DRAFT_MAX_CHARS,
        max(1, configured_target),
    )
    selected_groups: list[list[dict]] = []
    current_group: list[dict] = []
    current_chars = 0
    for batch in usable:
        batch_chars = len(str(batch.get("draft", "")))
        separator_chars = 2 if current_group else 0
        if (
            current_group
            and current_chars + separator_chars + batch_chars
            > target_group_chars
        ):
            selected_groups.append(current_group)
            current_group = []
            current_chars = 0
            separator_chars = 0
        current_group.append(batch)
        current_chars += separator_chars + batch_chars
    if current_group:
        selected_groups.append(current_group)

    groups = []
    for selected in selected_groups:
        group_draft = "\n\n".join(str(batch["draft"]) for batch in selected)

        def exact_or_fallback(field: str):
            candidates = [
                batch.get(field)
                for batch in selected
                if str(batch.get(field, "")).strip()
            ]
            if candidates:
                return max(candidates, key=lambda value: len(str(value)))
            return evidence.get(field, "")

        # Company knowledge comes first so a final character cap can never
        # silently remove the CV/reference proof while retaining market prose.
        group_evidence = {
            "company_knowledge": {
                "project_references": _relevant_evidence_excerpt(
                    exact_or_fallback("project_references"),
                    group_draft,
                    1200,
                ),
                "cv_excerpts": _relevant_evidence_excerpt(
                    exact_or_fallback("cv_excerpts"),
                    group_draft,
                    1200,
                ),
                "past_proposals": _relevant_evidence_excerpt(
                    exact_or_fallback("past_proposals"),
                    group_draft,
                    300,
                ),
                "provenance": (
                    "Retrieved from company AnythingLLM/Qdrant workspaces. "
                    "CVs and project references may support bidder claims. "
                    "Past proposals are tone/structure evidence unless they "
                    "explicitly contain the claimed company fact."
                ),
            },
            "requirements": _relevant_evidence_excerpt(
                exact_or_fallback("requirements"),
                group_draft,
                800,
            ),
            "section_evidence": [
                {
                    "sections": batch.get("sections", []),
                    "tender_excerpts": _relevant_evidence_excerpt(
                        batch.get("tender_excerpts", ""),
                        str(batch.get("draft", "")),
                        500,
                    ),
                    "response_template_excerpts": _relevant_evidence_excerpt(
                        batch.get("response_template_excerpts", ""),
                        str(batch.get("draft", "")),
                        300,
                    ),
                }
                for batch in selected
            ],
            "market_research": {
                "summary": _relevant_evidence_excerpt(
                    exact_or_fallback("research_summary"),
                    group_draft,
                    300,
                ),
                "provenance": (
                    "External context only; it cannot prove the bidding "
                    "company's experience, staff, projects, or certifications."
                ),
            },
        }
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


def quality_agent(state: dict, *, scanner=None) -> dict:
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
    duplicate_sections = _duplicate_sections(draft, section_order)
    incomplete_sections = _insubstantial_sections(draft, section_order)
    quality_findings = (
        scanner.scan(draft)
        if scanner is not None
        else (_run_llm_guard(draft) if _LLM_GUARD_AVAILABLE else {})
    )
    grounding_review = _evaluate_grounding_and_coherence(state, draft)
    groundedness_score = grounding_review["groundedness_score"]
    coherence_score = grounding_review["coherence_score"]
    evaluator_error = bool(grounding_review.get("evaluation_error"))
    grounding_failed = (
        evaluator_error
        or groundedness_score < MIN_GROUNDEDNESS_SCORE
        or coherence_score < MIN_COHERENCE_SCORE
        or bool(grounding_review.get("unsupported_claims"))
        or bool(grounding_review.get("contradictions"))
    )
    failed_sections = _identify_failed_sections(
        draft=draft,
        sections=section_order,
        missing_sections=missing_sections,
        out_of_order_sections=out_of_order_sections,
        duplicate_sections=duplicate_sections,
        incomplete_sections=incomplete_sections,
        quality_findings=quality_findings,
        grounding_review=grounding_review,
        word_count=word_count,
    )

    notes = []
    if word_count < MIN_WORD_COUNT:
        notes.append(f"Draft is short ({word_count} words) — may be incomplete.")
    if missing_sections:
        notes.append(f"Missing expected sections: {missing_sections}")
    if out_of_order_sections:
        notes.append(f"Template sections are out of order: {out_of_order_sections}")
    if duplicate_sections:
        notes.append(f"Duplicate template sections: {duplicate_sections}")
    if incomplete_sections:
        notes.append(
            "Sections without substantive body content "
            f"(minimum {MIN_SECTION_BODY_WORDS} words): {incomplete_sections}"
        )
    if quality_findings:
        notes.append(f"LLM Guard flagged: {quality_findings}")
    score_threshold_failed = (
        groundedness_score < MIN_GROUNDEDNESS_SCORE
        or coherence_score < MIN_COHERENCE_SCORE
    )
    if score_threshold_failed and not evaluator_error:
        notes.append(
            "Grounding/coherence review failed: "
            f"groundedness={groundedness_score:.2f} "
            f"(minimum {MIN_GROUNDEDNESS_SCORE:.2f}), "
            f"coherence={coherence_score:.2f} "
            f"(minimum {MIN_COHERENCE_SCORE:.2f})."
        )
    elif grounding_failed and not evaluator_error:
        notes.append(
            "Grounding review found unsupported or contradictory claims despite "
            "passing the numeric score thresholds."
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
        and not duplicate_sections
        and not incomplete_sections
        and not quality_findings
        and not grounding_failed
    )

    quality_report = {
        "word_count": word_count,
        "missing_sections": missing_sections,
        "out_of_order_sections": out_of_order_sections,
        "duplicate_sections": duplicate_sections,
        "incomplete_sections": incomplete_sections,
        "failed_sections": failed_sections,
        "required_sections": required_sections,
        "quality_findings": quality_findings,
        "grounding_review": grounding_review,
        "evaluation_available": not evaluator_error,
        "groundedness_threshold": MIN_GROUNDEDNESS_SCORE,
        "coherence_threshold": MIN_COHERENCE_SCORE,
        "notes": notes,
    }

    if evaluator_error:
        logger.error(
            "Quality evaluator failed: %s",
            grounding_review.get("evaluation_error"),
        )
    elif not passed:
        logger.warning("Quality check failed. Notes: %s", notes)
    else:
        logger.info("Quality check passed.")

    return {
        "quality_passed": passed,
        "quality_report": quality_report,
    }
