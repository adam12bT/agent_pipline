"""Generation agent implementation behind the packaged contract."""

import logging
import os
import re

from .prompts import GENERATION_PROMPT_TEMPLATE
from providers import get_provider
from pipeline_progress import (
    finish_generation,
    mark_batch_completed,
    mark_batch_started,
    start_generation,
)

PROPOSALS_WORKSPACE = "company-past-proposals"
CVS_WORKSPACE = "company-cvs"
REFERENCES_WORKSPACE = "company-project-references"

logger = logging.getLogger(__name__)

_DEFAULT_PROPOSAL_SECTIONS = [
    "Executive Summary",
    "Understanding of the Requirements",
    "Proposed Approach & Methodology",
    "Indicative Work Plan / Timeline",
    "Risk Management & Quality Assurance",
    "Proposed Team (Profils Proposés)",
    "Why Us",
]


def _proposal_sections(response_template_rules: dict) -> list[str]:
    rules = response_template_rules if isinstance(response_template_rules, dict) else {}
    raw_sections = rules.get("section_order") or rules.get("required_sections") or []
    sections = [str(section).strip() for section in raw_sections if str(section).strip()]
    return sections or list(_DEFAULT_PROPOSAL_SECTIONS)


def _section_batches(
    response_template_rules: dict, batch_size: int = 1
) -> list[list[str]]:
    size = max(1, batch_size)
    sections = _proposal_sections(response_template_rules)
    return [sections[index : index + size] for index in range(0, len(sections), size)]


def _batches_for_sections(sections: list[str], batch_size: int) -> list[list[str]]:
    size = max(1, batch_size)
    return [sections[index : index + size] for index in range(0, len(sections), size)]


def _rule_items(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    return []


def _template_section_word_target(
    response_template_rules: dict,
    section_count: int,
) -> tuple[int, int, str]:
    """Derive one per-section budget from the uploaded template's global limit."""
    rules = (
        response_template_rules
        if isinstance(response_template_rules, dict)
        else {}
    )
    formatting = _rule_items(rules.get("formatting_requirements"))
    instructions = _rule_items(
        rules.get("instructions") or rules.get("template_instructions")
    )
    template_text = " ".join([*formatting, *instructions])
    normalized = re.sub(r"[\s\u00a0]+", " ", template_text.casefold())

    normalized_formatting = re.sub(
        r"[\s\u00a0]+",
        " ",
        " ".join(formatting).casefold(),
    )

    total_word_limit = None
    formatting_word_match = re.search(
        r"(?:maximum|max|not exceed|ne doit pas depasser)"
        r"\D{0,12}(\d[\d ,.]*?)\s*(?:words|mots)\b",
        normalized_formatting,
    )
    if formatting_word_match:
        digits = re.sub(r"\D", "", formatting_word_match.group(1))
        total_word_limit = int(digits) if digits else None

    total_word_patterns = (
        r"(?:proposal|response|submission|document|offre|reponse)"
        r"[^.]{0,40}?(?:maximum|max|not exceed|ne doit pas depasser)"
        r"\D{0,12}(\d[\d ,.]*?)\s*(?:words|mots)\b",
        r"(?:maximum|max|not exceed|ne doit pas depasser)"
        r"\D{0,12}(\d[\d ,.]*?)\s*(?:words|mots)\b"
        r"[^.]{0,30}?(?:total|proposal|response|submission|document|offre|reponse)",
    )
    for pattern in total_word_patterns:
        if total_word_limit:
            break
        match = re.search(pattern, normalized)
        if match:
            digits = re.sub(r"\D", "", match.group(1))
            if digits:
                total_word_limit = int(digits)
                break

    maximum_pages = None
    page_match = re.search(
        r"(?:maximum|max|not exceed|ne doit pas depasser)"
        r"\D{0,12}(\d{1,3})\s*pages?\b",
        normalized,
    )
    if page_match:
        maximum_pages = int(page_match.group(1))

    count = max(1, section_count)
    if total_word_limit:
        target_total = max(count * 180, int(total_word_limit * 0.85))
        source = f"template total-word limit ({total_word_limit})"
    elif maximum_pages:
        target_total = max(count * 180, int(maximum_pages * 350 * 0.70))
        source = f"template page limit ({maximum_pages} pages)"
    else:
        target_total = count * 450
        source = "section count (no template word/page limit found)"

    words_per_section = target_total / count
    minimum_words = max(180, min(650, round(words_per_section * 0.80)))
    maximum_words = max(
        minimum_words,
        min(750, round(words_per_section * 1.05)),
    )
    return minimum_words, maximum_words, source


def _proposal_structure(
    response_template_rules: dict, sections: list[str] | None = None
) -> str:
    """Turn extracted template rules into an explicit Markdown outline.

    The old prompt always included a detailed default outline, which competed
    with an uploaded client template. Defaults are now used only when the
    template genuinely contains no section structure.
    """
    rules = response_template_rules if isinstance(response_template_rules, dict) else {}
    raw_sections = rules.get("section_order") or rules.get("required_sections") or []
    using_client_template = bool(raw_sections)
    all_sections = _proposal_sections(rules)
    selected_sections = sections or all_sections
    minimum_words, maximum_words, budget_source = _template_section_word_target(
        rules,
        len(all_sections),
    )

    lines = [
        "CLIENT TEMPLATE - USE THESE EXACT HEADINGS AND THIS EXACT ORDER:"
        if using_client_template
        else "NO CLIENT SECTION OUTLINE WAS FOUND â€” USE THESE DEFAULT HEADINGS:",
        (
            f"Per-section word budget: {minimum_words}-{maximum_words} words; "
            f"derived from {budget_source}."
        ),
    ]
    for section in selected_sections:
        lines.extend(
            [
                f"## {section}",
                f"Target length: {minimum_words}-{maximum_words} words.",
            ]
        )

    instructions = rules.get("instructions") or rules.get("template_instructions") or []
    formatting = rules.get("formatting_requirements") or []
    if isinstance(instructions, str):
        instructions = [instructions]
    if isinstance(formatting, str):
        formatting = [formatting]
    if instructions:
        lines.extend(["", "Template instructions:"])
        lines.extend(f"- {item}" for item in instructions if str(item).strip())
    if formatting:
        lines.extend(["", "Formatting requirements:"])
        lines.extend(f"- {item}" for item in formatting if str(item).strip())

    return "\n".join(lines)


def _canonical_heading(value: str) -> str:
    heading = str(value).strip().casefold()
    heading = re.sub(r"^\s{0,3}#{1,6}\s*", "", heading)
    heading = re.sub(r"[*_`]", "", heading)
    heading = re.sub(r"^\s*(?:section\s+)?\d+(?:\.\d+)*[.)\-:]?\s*", "", heading)
    return re.sub(r"\s+", " ", heading).strip(" :-–—")


def _heading_aliases(section: str) -> set[str]:
    """Return full and bilingual-half aliases for one template heading."""
    aliases = {_canonical_heading(section)}
    aliases.update(
        _canonical_heading(part)
        for part in str(section).split("/")
        if _canonical_heading(part)
    )
    return aliases


def _section_number(value: str) -> str | None:
    """Return a leading template section number, such as ``5`` or ``5.2``."""
    heading = re.sub(r"^\s{0,3}#{1,6}\s*", "", str(value).strip())
    match = re.match(r"^(?:section\s+)?(\d+(?:\.\d+)*)[.)\-:]?\s+", heading, re.I)
    return match.group(1) if match else None


def _normalize_batch_headings(draft: str, sections: list[str]) -> tuple[str, list[str]]:
    """Restore exact client titles when the model shortens bilingual headings.

    Models commonly emit only the French or English half of a bilingual title.
    The content is still the requested section, so replace that Markdown heading
    with the exact client title. Truly absent sections are returned unchanged and
    remain visible to the quality gate instead of being hidden by a placeholder.
    """
    lines = draft.splitlines()
    unmatched = []
    used_lines: set[int] = set()
    for section in sections:
        aliases = _heading_aliases(section)
        expected_number = _section_number(section)
        matched_line = next(
            (
                index
                for index, line in enumerate(lines)
                if index not in used_lines
                and re.match(r"^\s{0,3}#{1,6}\s+", line)
                and (
                    _canonical_heading(line) in aliases
                    or (
                        expected_number is not None
                        and _section_number(line) == expected_number
                    )
                )
            ),
            None,
        )
        if matched_line is None:
            unmatched.append(section)
            continue
        lines[matched_line] = f"## {section}"
        used_lines.add(matched_line)
    return "\n".join(lines), unmatched


def _split_batch_sections(draft: str, sections: list[str]) -> dict[str, str]:
    """Split a generated Markdown batch into exact template-section blocks."""
    lines = draft.splitlines()
    starts: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        if not re.match(r"^\s{0,3}#{1,6}\s+", line):
            continue
        canonical = _canonical_heading(line)
        number = _section_number(line)
        matched = next(
            (
                section
                for section in sections
                if canonical in _heading_aliases(section)
                or (number is not None and number == _section_number(section))
            ),
            None,
        )
        if matched and matched not in {title for _, title in starts}:
            starts.append((index, matched))

    content: dict[str, str] = {}
    for position, (start, section) in enumerate(starts):
        end = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
        content[section] = "\n".join(lines[start:end]).strip()
    return content


def _merge_section_drafts(
    previous_draft: str,
    all_sections: list[str],
    replacements: dict[str, str],
) -> str:
    """Merge repaired sections into a prior draft in exact template order."""
    previous = _split_batch_sections(previous_draft, all_sections)
    merged = []
    for section in all_sections:
        content = replacements.get(section) or previous.get(section)
        if content and content.strip():
            merged.append(content.strip())
    return "\n\n".join(merged)


def _rebuild_section_evidence(
    all_sections: list[str],
    final_draft: str,
    evidence_batches: list[dict],
) -> list[dict]:
    """Create ordered, current evidence records after targeted replacements."""
    final_blocks = _split_batch_sections(final_draft, all_sections)
    rebuilt = []
    for section in all_sections:
        source = next(
            (
                batch
                for batch in reversed(evidence_batches)
                if section in (batch.get("sections") or [])
            ),
            {},
        )
        record = {
            key: value
            for key, value in source.items()
            if key not in {"sections", "draft"}
        }
        record["sections"] = [section]
        record["draft"] = final_blocks.get(section, "")
        rebuilt.append(record)
    return rebuilt


def _clip(value, limit: int) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n[context truncated to {limit} characters]"


def _truncate_to_length(value: str, target_length: int) -> str:
    marker = "\n[truncated to fit Groq request budget]"
    if len(value) <= target_length:
        return value
    if target_length <= len(marker):
        return value[:target_length]
    return f"{value[: target_length - len(marker)]}{marker}"


def _fit_generation_prompt(format_values: dict, max_chars: int) -> tuple[str, dict]:
    """Render a prompt whose *total* size stays below the Groq free-tier budget.

    Per-field clipping is insufficient because the generation prompt combines
    several independent RAG fields. Keep the fixed instructions and proposal
    headings intact, then progressively trim optional evidence fields.
    """
    fitted = {key: str(value) for key, value in format_values.items()}
    prompt = GENERATION_PROMPT_TEMPLATE.format(**fitted)
    if len(prompt) <= max_chars:
        return prompt, fitted

    # Lower-priority/reference material is reduced first. Tender facts,
    # extracted requirements, template rules and headings retain larger floors.
    shrink_order = [
        ("past_proposals", 200),
        ("revision_feedback", 200),
        ("research_summary", 500),
        ("project_references", 500),
        ("cv_excerpts", 500),
        ("response_template_excerpts", 700),
        ("tender_excerpts", 1200),
        ("requirements", 1200),
        ("response_template_rules", 700),
    ]
    for field, minimum in shrink_order:
        overflow = len(prompt) - max_chars
        if overflow <= 0:
            break
        current = fitted[field]
        reducible = max(0, len(current) - minimum)
        if not reducible:
            continue
        # Leave room for the truncation marker added by the helper.
        target = len(current) - min(reducible, overflow + 50)
        fitted[field] = _truncate_to_length(current, max(minimum, target))
        prompt = GENERATION_PROMPT_TEMPLATE.format(**fitted)

    if len(prompt) > max_chars:
        raise ValueError(
            "The fixed generation instructions exceed the configured total "
            f"prompt budget of {max_chars} characters."
        )
    return prompt, fitted


def _search_knowledge_port(knowledge, workspace_slug: str, query: str, top_n: int = 3) -> str:
    try:
        results = knowledge.search(workspace_slug, query, top_n=top_n)
    except Exception as exc:
        logger.warning("Injected knowledge search failed for %r: %s", workspace_slug, exc)
        results = []
    if not results:
        return "(none found in the company knowledge base for this query)"
    return "\n".join(
        f"- From [{item.get('metadata', {}).get('title', 'unknown source')}]: "
        f"{item.get('text', '').strip()}"
        for item in results
    )


def generation_agent(state: dict, *, rag=None, knowledge=None) -> dict:
    if not state.get("is_verified"):
        return {}

    if rag is None or knowledge is None:
        raise RuntimeError("RagQuery and KnowledgeSearch dependencies are required")
    try:
        knowledge.ensure_ready()
    except Exception as exc:
        logger.warning("Injected company knowledge adapter is not ready: %s", exc)
    workspace_slug = state["workspace_slug"]
    template_workspace_slug = state["response_template_workspace_slug"]
    requirements = state.get("requirements", {})
    search_query = requirements.get("scope_summary") or "technical proposal requirements"
    previous_generation_evidence = state.get("previous_generation_evidence") or {}

    # Reuse the exact company evidence from the first attempt. A repair should
    # not spend more vector-search calls or silently change its evidence base.
    if previous_generation_evidence:
        project_references = previous_generation_evidence.get(
            "project_references", ""
        )
        cv_excerpts = previous_generation_evidence.get("cv_excerpts", "")
        past_proposals = previous_generation_evidence.get("past_proposals", "")
    else:
        project_references = _search_knowledge_port(
            knowledge, REFERENCES_WORKSPACE, search_query
        )
        cv_excerpts = _search_knowledge_port(
            knowledge, CVS_WORKSPACE, f"consultant profile relevant to: {search_query}"
        )
        past_proposals = _search_knowledge_port(
            knowledge, PROPOSALS_WORKSPACE, f"past proposal similar to: {search_query}"
        )

    response_template_rules = requirements.get("response_template", {})
    revision_feedback = state.get("quality_report") or "(first generation attempt)"
    attempt_number = state.get("generation_attempts", 0) + 1
    previous_draft = state.get("previous_draft", "")
    # One section per call prevents a detailed early section from consuming the
    # output allowance reserved for later headings. A 1,600-token response is
    # ample for the largest 700-word section and remains below hosted limits.
    batch_size = 1
    batch_max_tokens = min(
        1600,
        max(512, int(os.environ.get("GENERATION_BATCH_MAX_TOKENS", "1600"))),
    )
    context_limit = max(
        2000, int(os.environ.get("GENERATION_CONTEXT_MAX_CHARS", "6000"))
    )
    prompt_max_chars = min(
        11000,
        max(8000, int(os.environ.get("GENERATION_PROMPT_MAX_CHARS", "11000"))),
    )
    all_sections = _proposal_sections(response_template_rules)
    repair_sections = []
    if attempt_number > 1 and previous_draft.strip():
        requested_repairs = revision_feedback.get("failed_sections", []) if isinstance(
            revision_feedback, dict
        ) else []
        repair_sections = [
            section for section in all_sections if section in requested_repairs
        ]
    selected_sections = repair_sections or all_sections
    repair_mode = bool(repair_sections)
    batches = _batches_for_sections(selected_sections, batch_size=batch_size)
    run_id = state.get("run_id")
    start_generation(run_id, batches)

    retained_batches = []
    if repair_mode:
        retained_batches = [
            dict(batch)
            for batch in previous_generation_evidence.get("section_batches", [])
            if isinstance(batch, dict)
        ]
    generation_evidence = {
        "section_batches": retained_batches,
        "requirements": requirements,
        "research_summary": state.get("research_summary", "(no research available)"),
        "project_references": project_references,
        "cv_excerpts": cv_excerpts,
        "past_proposals": past_proposals,
        "repair_mode": repair_mode,
        "repaired_sections": repair_sections,
    }

    logger.info(
        "Generation attempt %d for workspace %r using %d dynamic batch(es)%s",
        attempt_number,
        workspace_slug,
        len(batches),
        f" to repair {repair_sections}" if repair_mode else "",
    )

    draft_parts = []
    replacement_sections: dict[str, str] = {}
    previous_blocks = _split_batch_sections(previous_draft, all_sections)
    for batch_number, sections in enumerate(batches, start=1):
        mark_batch_started(run_id, batch_number)
        section_names = "; ".join(sections)
        batch_query = (
            f"{search_query}; facts, constraints, evidence and instructions for sections: "
            f"{section_names}"
        )
        tender_excerpts = rag.query(workspace_slug, batch_query, top_n=4)
        response_template_excerpts = rag.query(
            template_workspace_slug,
            f"content instructions, tables and formatting for sections: {section_names}",
            top_n=4,
        )
        batch_revision_feedback = revision_feedback
        if repair_mode:
            batch_revision_feedback = {
                "quality_report": revision_feedback,
                "previous_section_content": {
                    section: previous_blocks.get(section, "") for section in sections
                },
                "instruction": (
                    "Rewrite only these failed sections. Correct every cited issue "
                    "without changing facts supported elsewhere in the proposal."
                ),
            }
        prompt, fitted_context = _fit_generation_prompt(
            {
                "batch_number": batch_number,
                "batch_count": len(batches),
                "tender_excerpts": _clip(tender_excerpts, context_limit),
                "response_template_excerpts": _clip(
                    response_template_excerpts, context_limit
                ),
                "response_template_rules": _clip(
                    response_template_rules, context_limit
                ),
                "proposal_structure": _proposal_structure(
                    response_template_rules, sections
                ),
                "revision_feedback": _clip(batch_revision_feedback, context_limit),
                "requirements": _clip(requirements, context_limit),
                "research_summary": _clip(
                    state.get("research_summary", "(no research available)"),
                    context_limit,
                ),
                "project_references": _clip(project_references, context_limit),
                "cv_excerpts": _clip(cv_excerpts, context_limit),
                "past_proposals": _clip(past_proposals, context_limit),
            },
            prompt_max_chars,
        )
        batch_evidence = {
            "sections": sections,
            "tender_excerpts": fitted_context["tender_excerpts"],
            "response_template_excerpts": fitted_context[
                "response_template_excerpts"
            ],
            "prompt_chars": len(prompt),
        }
        generation_evidence["section_batches"].append(batch_evidence)
        logger.info(
            "Generation batch %d/%d prompt fitted to %d/%d characters",
            batch_number,
            len(batches),
            len(prompt),
            prompt_max_chars,
        )

        try:
            batch_draft = get_provider().complete(
                prompt,
                max_tokens=batch_max_tokens,
                request_label=f"generation.batch_{batch_number}_of_{len(batches)}",
                reasoning_effort="low",
                include_reasoning=False,
            ).strip()
            if not batch_draft:
                raise ValueError("the model returned an empty section batch")
            batch_draft, unmatched_headings = _normalize_batch_headings(
                batch_draft, sections
            )
            if unmatched_headings:
                logger.warning(
                    "Generation batch %d/%d omitted template headings: %s",
                    batch_number,
                    len(batches),
                    unmatched_headings,
                )
            batch_evidence["draft"] = batch_draft
            draft_parts.append(batch_draft)
            generated_sections = _split_batch_sections(batch_draft, sections)
            replacement_sections.update(generated_sections)
            mark_batch_completed(
                run_id,
                batch_number,
                generated_sections,
            )
            logger.info(
                "Generation attempt %d completed batch %d/%d (%s)",
                attempt_number, batch_number, len(batches), section_names,
            )
        except Exception as e:
            error_msg = (
                f"Generation batch {batch_number}/{len(batches)} failed "
                f"for sections {sections}: {e}"
            )
            logger.error(
                "Generation attempt %d failed in batch %d/%d for workspace %r: %s",
                attempt_number, batch_number, len(batches), workspace_slug, e,
                exc_info=True,
            )
            finish_generation(run_id, failed=True)
            return {
                "draft_proposal": "",
                "generation_evidence": generation_evidence,
                "generation_attempts": attempt_number,
                "errors": [error_msg],
            }

    draft = (
        _merge_section_drafts(previous_draft, all_sections, replacement_sections)
        if repair_mode
        else "\n\n".join(draft_parts)
    )
    generation_evidence["section_batches"] = _rebuild_section_evidence(
        all_sections,
        draft,
        generation_evidence["section_batches"],
    )
    finish_generation(run_id)

    return {
        "draft_proposal": draft,
        "generation_evidence": generation_evidence,
        "generation_attempts": attempt_number,
    }
