"""
Quality Agent
--------------
Runs after Security — but only if security passed (see the guard below;
the graph itself already routes a failed security check straight to END,
so this is a defensive double-check, not the primary gate).

Checks the draft's QUALITY, not its safety: template compliance, length,
tone, refusal detection, coherence, and hallucination risk. Unlike the
Security agent, a failure here is GRADED — it triggers a regeneration
attempt (up to MAX_GENERATION_ATTEMPTS) via the "retry_generation" status
rather than a hard stop.

Uses LLM Guard's:
  - `ToxicLanguage`      -> tone/appropriateness scoring
  - `NoRefusal`          -> catches the generation model refusing / punting
                            instead of writing the section (a common failure
                            mode for generation agents)
  - `FactualConsistency` -> flags output that CONTRADICTS the grounding
                            material (tender requirements / research / CVs /
                            references) it was supposed to be written from —
                            the practical stand-in for "hallucination
                            detection" (there's no dedicated hallucination
                            scanner in LLM Guard as of this writing).
  - `Relevance`          -> embedding-similarity check between the grounding
                            material and the output — the practical stand-in
                            for "coherence scoring."
plus the pre-existing template-compliance and word-count checks.

CHANGED: FactualConsistency and Relevance are new — the pipeline
previously had no hallucination or coherence check at all, despite that
being an explicit requirement. When generation_agent provides
`draft_sections` (dict of section key -> text) and `generation_grounding`
(the requirements/research/CVs/references actually used to write the
draft), both new scanners run PER SECTION against that grounding text —
narrower context per call, and one hallucinated section doesn't get
diluted by six clean ones the way a single whole-document score would.
If those keys aren't present (e.g. an older generation_agent, or a state
schema that dropped them), quality_agent falls back to running the two
scanners once against the whole draft with no grounding text — no
grounding text means FactualConsistency/Relevance have nothing to check
against, so that fallback is skipped entirely rather than scored against
an empty prompt.

Per-section minimum word counts were also added (`SECTION_MIN_WORDS`):
the old single `MIN_WORD_COUNT = 150` check against the whole document
couldn't have caught a shallow draft even before this file's rewrite —
150 words is far below what any real generated draft has ever measured
at, so it was a check that could never fail in practice. It's now
raised to reflect the actual combined section-length targets in
generation_agent.py's prompts, with a per-section check underneath it so
a single thin section can't hide inside an otherwise-long document.

Known limitation: FactualConsistency and Relevance run on models with a
limited input window (the default FactualConsistency model, a
deberta-v3-base NLI model, only reliably attends to roughly its ~512
token training context). `generation_grounding` is trimmed before being
used as the scan prompt to stay well under that, which means a
hallucinated detail whose only supporting/contradicting evidence fell
outside the trimmed portion can still slip through. This is a
coverage-vs-cost tradeoff, not a hard guarantee — flag it in your
evaluation write-up as a known limitation, same as the MaliciousURLs/
prompt-injection proxy already flagged in security_agent.py.

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
    "Risk Management",
    "Proposed Team",
    "Why Us",
]

# Per-section floors, keyed to match generation_agent.py's SECTION_SPECS
# keys. Set a bit below each section's stated prompt target (which is a
# midpoint/ceiling guide for the model, not a hard floor) to avoid false
# positives on a section that's simply concise rather than shallow.
# "proposed_team" is deliberately excluded: a short, exact placeholder
# sentence is the CORRECT output when no matching CVs were found, so
# penalizing it for length would be wrong.
SECTION_MIN_WORDS = {
    "executive_summary": 110,
    "understanding_of_requirements": 220,
    "proposed_approach": 380,
    "work_plan": 140,
    "risk_management": 140,
    "why_us": 180,
}

# Sum of the floors above (plus a little slack for the team section,
# which varies) — replaces the old MIN_WORD_COUNT = 150, which no real
# generated draft could ever fail.
MIN_WORD_COUNT = 1250

MAX_GENERATION_ATTEMPTS = 3

# Keep well under the ~512-token window the default FactualConsistency /
# Relevance models are trained on. ~4 chars/token is a rough English
# average, so this is a conservative ceiling, not a precise token count.
_MAX_GROUNDING_CHARS = 1800

try:
    from llm_guard.output_scanners import FactualConsistency, NoRefusal, Relevance, ToxicLanguage

    _LLM_GUARD_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when dep is missing
    _LLM_GUARD_AVAILABLE = False
    logger.warning(
        "llm-guard not installed — skipping toxicity/refusal/coherence/hallucination "
        "scanning. Run `pip install llm-guard` to enable it."
    )

_scanners_cache = None


def _get_scanners():
    global _scanners_cache
    if _scanners_cache is None:
        _scanners_cache = {
            "toxicity": ToxicLanguage(threshold=0.7),
            "no_refusal": NoRefusal(threshold=0.5),
            "factual_consistency": FactualConsistency(minimum_score=0.7),
            "relevance": Relevance(threshold=0.5),
        }
    return _scanners_cache


def _run_content_scanners(draft: str) -> dict:
    """Toxicity + refusal — pure output scanners, no grounding prompt needed."""
    scanners = _get_scanners()
    findings = {}
    for name in ("toxicity", "no_refusal"):
        try:
            _, is_valid, risk_score = scanners[name].scan(prompt="", output=draft)
        except Exception as exc:
            logger.warning("LLM Guard scanner %r failed, skipping: %s", name, exc)
            continue
        if not is_valid:
            findings[name] = round(risk_score, 3)
    return findings


def _run_grounding_scanners(grounding: str, text: str) -> dict:
    """FactualConsistency + Relevance — need the grounding material as the
    scanner's `prompt` argument to check `text` against."""
    scanners = _get_scanners()
    findings = {}
    for name in ("factual_consistency", "relevance"):
        try:
            _, is_valid, risk_score = scanners[name].scan(prompt=grounding, output=text)
        except Exception as exc:
            logger.warning("LLM Guard scanner %r failed, skipping: %s", name, exc)
            continue
        if not is_valid:
            findings[name] = round(risk_score, 3)
    return findings


def _trim_grounding(text: str) -> str:
    if len(text) <= _MAX_GROUNDING_CHARS:
        return text
    return text[:_MAX_GROUNDING_CHARS] + " ...[grounding truncated for scanner input limits]"


def _check_template_compliance(draft: str) -> list[str]:
    return [s for s in REQUIRED_SECTIONS if s.lower() not in draft.lower()]


def _check_section_lengths(draft_sections: dict) -> dict:
    short = {}
    for key, min_words in SECTION_MIN_WORDS.items():
        text = draft_sections.get(key, "")
        if not text or text.startswith("[This section could not be generated"):
            continue  # already surfaced as a generation error, not a quality finding
        word_count = len(text.split())
        if word_count < min_words:
            short[key] = word_count
    return short


def quality_agent(state: dict) -> dict:
    if not state.get("is_verified"):
        return {}
    if not state.get("security_passed", True):
        # Defensive — the graph should never route here on a security
        # failure, but don't silently score a blocked draft if it does.
        return {}

    draft = state.get("draft_proposal", "")
    draft_sections = state.get("draft_sections")
    grounding = state.get("generation_grounding")

    word_count = len(draft.split())
    missing_sections = _check_template_compliance(draft)
    content_findings = _run_content_scanners(draft) if _LLM_GUARD_AVAILABLE else {}

    # Coherence / hallucination scan — per-section against grounding text
    # when generation_agent provided it, otherwise a single whole-draft
    # pass, otherwise skipped (no grounding = nothing to check against).
    grounding_findings = {}
    if _LLM_GUARD_AVAILABLE and grounding:
        trimmed_grounding = _trim_grounding(grounding)
        if draft_sections:
            for key, text in draft_sections.items():
                if not text or text.startswith("[This section could not be generated"):
                    continue
                findings = _run_grounding_scanners(trimmed_grounding, text)
                if findings:
                    grounding_findings[key] = findings
        else:
            findings = _run_grounding_scanners(trimmed_grounding, draft)
            if findings:
                grounding_findings["_whole_draft"] = findings

    section_word_findings = _check_section_lengths(draft_sections) if draft_sections else {}

    notes = []
    if word_count < MIN_WORD_COUNT:
        notes.append(f"Draft is short ({word_count} words, expected >= {MIN_WORD_COUNT}) — may be incomplete.")
    if missing_sections:
        notes.append(f"Missing expected sections: {missing_sections}")
    if content_findings:
        notes.append(f"LLM Guard flagged tone/refusal issues: {content_findings}")
    if grounding_findings:
        notes.append(f"LLM Guard flagged coherence/hallucination risk: {grounding_findings}")
    if section_word_findings:
        notes.append(f"Sections shorter than expected: {section_word_findings}")

    passed = (
        word_count >= MIN_WORD_COUNT
        and not missing_sections
        and not content_findings
        and not grounding_findings
        and not section_word_findings
    )

    quality_report = {
        "word_count": word_count,
        "missing_sections": missing_sections,
        "quality_findings": content_findings,
        "coherence_hallucination_findings": grounding_findings,
        "short_sections": section_word_findings,
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