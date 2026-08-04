"""
Generation Agent
-----------------
Produces the full technical proposal.

CHANGED: this used to make ONE `client.chat(...)` call with a single giant
prompt asking for all 7 sections (~1,900-2,500 words total) at once. In
practice the model treated the per-section word-count targets as vague
suggestions rather than real floors — every section came back at roughly
40-50% of its requested length, filled with generic bid-writer boilerplate
("we will conduct thorough testing") instead of analysis grounded in the
actual requirements/research/CVs, even though those were present in the
prompt. Classic "asked to do too much in one shot, so it satisfices."

Fix: generate each section with its OWN `client.chat(...)` call. Every
call still gets the full shared context (requirements, research, CV
excerpts, references, past proposals) so sections stay consistent with
each other, but each call only has to satisfy ONE section's depth target
instead of competing with six others for the model's attention/token
budget. A section that fails is replaced with a clearly-marked
placeholder instead of silently dragging down or aborting the whole
report, and the failure is recorded in `errors`.

Two things this required, from reading anythingllm_client.py:

- `chat()` defaults to a fixed `session_id="rfp-pipeline"`, and
  `mode="chat"` uses rolling history (per that method's own docstring).
  Calling `chat()` 7 times against the same workspace with the default
  session id would make AnythingLLM accumulate history across all 7
  calls — so by the last section, the "one focused ask" this rewrite is
  built around would actually be competing with the full history of
  every prior section's prompt AND response, reintroducing the same
  overload problem via a different path. Every section call below gets
  its own unique `session_id` (workspace + section + attempt number) so
  each one is genuinely independent and stateless, and retries after a
  quality-agent "retry_generation" don't inherit a prior attempt's
  history either.
- `chat()` has no `max_tokens`/length override to lean on, so section
  depth can only be managed by keeping each individual request small —
  there's no server-side knob to raise instead.

The 7 section calls have no data dependency on each other, so they're
fired concurrently via a thread pool instead of sequentially — otherwise
splitting one call into seven would turn a single request's latency into
up to 7x that (and worse on every regeneration retry).

CHANGED (2): now also returns `draft_sections` (dict of section key ->
section text) and `generation_grounding` (the requirements/research/
references/CVs actually fed to the generation calls, concatenated) so
quality_agent can run LLM Guard's FactualConsistency/Relevance scanners
per section against the material each section was supposed to be
grounded in, instead of only being able to see the final stitched
Markdown string. `draft_proposal` is unchanged and still the full
document, for anything downstream that only needs the final text.

NOTE FOR WHOEVER OWNS state.py/graph.py: `draft_sections` and
`generation_grounding` are new state keys. If the LangGraph state schema
is a strict TypedDict, these need to be added there or LangGraph may
silently drop them, and quality_agent's new per-section scanning will
just no-op back to whole-draft mode (harmless, but you lose the
granularity).
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from anythingllm_client import AnythingLLMClient
from company_knowledge import PROPOSALS_WORKSPACE, CVS_WORKSPACE, REFERENCES_WORKSPACE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared instructions injected into EVERY section call, so each one stays
# grounded in the real material instead of drifting into generic filler.
# ---------------------------------------------------------------------------
_SHARED_PREAMBLE = """You are a senior bid writer producing ONE section of a FULL technical \
proposal report in response to the tender document in this workspace. This is a formal \
deliverable that will be submitted to the client for evaluation. Write in complete, \
substantive paragraphs with real analysis and detail grounded in the material provided below. \
Shallow, generic filler is worse than a shorter passage that is actually specific to this \
tender — do not pad with boilerplate sentences ("we will ensure quality throughout") that \
could apply to any project. Every claim should trace back to something in the requirements, \
the research, the CV excerpts, or the tender document itself.

Do not invent specific figures, dates, project names, or consultant names that are not \
present in the material below or the tender document. Leave a clear placeholder like \
[TO BE CONFIRMED] instead of making something up. Write in a professional, confident \
register appropriate for a formal procurement submission.

TENDER REQUIREMENTS (extracted from the tender document):
{requirements}

MARKET / COMPETITOR RESEARCH:
{research_summary}

RELEVANT PAST PROJECT REFERENCES (from our company's own project history):
{project_references}

RELEVANT CONSULTANT CVs (company knowledge base):
{cv_excerpts}

RELEVANT PAST PROPOSALS (for tone/structure reference only — do not copy verbatim):
{past_proposals}

---

Now write ONLY the following section. Do not write a heading yourself — start directly with \
the section content. Do not write any other section, and do not add a preamble like "Here is \
the section" — respond with the section text only.

SECTION: {section_title}
{section_instructions}"""


def _fmt_requirements(requirements: dict) -> str:
    """Render the requirements dict as readable prose/bullets instead of
    dumping raw Python dict syntax (str(dict)) into the prompt — that was
    harder for the model to parse cleanly and easy to skim past."""
    if not requirements:
        return "(no structured requirements were extracted — rely on general tender context)"

    lines = []
    scope = requirements.get("scope_summary")
    if scope:
        lines.append(f"- Scope: {scope}")

    deliverables = requirements.get("deliverables") or []
    if deliverables:
        lines.append("- Deliverables:")
        lines.extend(f"    - {d}" for d in deliverables)

    deadlines = requirements.get("deadlines") or {}
    if deadlines.get("submission_deadline"):
        lines.append(f"- Submission deadline: {deadlines['submission_deadline']}")
    if deadlines.get("project_duration"):
        lines.append(f"- Project duration: {deadlines['project_duration']}")

    if requirements.get("budget"):
        lines.append(f"- Budget: {requirements['budget']}")

    criteria = requirements.get("evaluation_criteria") or []
    if criteria:
        lines.append("- Evaluation criteria:")
        lines.extend(f"    - {c}" for c in criteria)

    if requirements.get("selection_method"):
        lines.append(f"- Selection method: {requirements['selection_method']}")

    return "\n".join(lines) if lines else "(no structured requirements were extracted)"


def _build_grounding_text(shared_context: dict) -> str:
    """Concatenate the material each section was generated against into one
    reference text, for quality_agent's hallucination/coherence scanners.
    Labeled blocks rather than a raw dump, so a scanner (or a human
    reading a quality report) can tell which part came from where."""
    labeled = [
        ("TENDER REQUIREMENTS", shared_context.get("requirements")),
        ("MARKET RESEARCH", shared_context.get("research_summary")),
        ("PROJECT REFERENCES", shared_context.get("project_references")),
        ("CV EXCERPTS", shared_context.get("cv_excerpts")),
    ]
    blocks = [f"{label}:\n{text}" for label, text in labeled if text]
    return "\n\n".join(blocks)


def _search_company_knowledge(client: AnythingLLMClient, workspace_slug: str, query: str,
                                top_n: int = 3) -> str:
    """Search one company knowledge workspace and format results as readable text.
    Returns a clear "none found" message instead of an empty string, so the LLM
    prompt reads naturally either way."""
    try:
        results = client.vector_search(workspace_slug, query, top_n=top_n)
    except Exception:
        results = []

    if not results:
        return "(none found in the company knowledge base for this query)"

    formatted = []
    for r in results:
        title = r.get("metadata", {}).get("title", "unknown source")
        text = r.get("text", "").strip()
        formatted.append(f"- From [{title}]: {text}")
    return "\n".join(formatted)


def _section_specs(has_cv_excerpts: bool, has_references: bool) -> list[dict]:
    """Build the per-section instruction list. Team/Why-Us instructions branch
    on whether we actually found matching CVs/references, same logic the old
    single-prompt template used, just applied per-section now."""

    if has_cv_excerpts:
        team_instructions = (
            "Based ONLY on the CV excerpts provided above, give each person a substantive "
            "paragraph: their role on this project, relevant background, and specifically why "
            "they fit this tender's requirements (not a generic 'well qualified' statement — "
            "tie their background to a named deliverable or requirement above). Do not invent "
            "anyone not present in the excerpts. If a person's name isn't given, refer to them "
            "by role only (e.g. 'the Backend & Systems Engineer') rather than using a gendered "
            "pronoun for a placeholder."
        )
    else:
        team_instructions = (
            'Write exactly this text and nothing else: "[TEAM PROFILES TO BE COMPLETED — no '
            'matching CVs found in the company knowledge base]"'
        )

    if has_references:
        why_us_instructions = (
            "Reference the past project references above concretely (what was delivered, for "
            "whom, and how it's relevant here) and use the market research to position us "
            "against the likely competitors it names. Make a substantive, specific case — "
            "~250-350 words."
        )
    else:
        why_us_instructions = (
            "No past project references were found in the company knowledge base — do not "
            "invent any. Keep this section general but still substantive: argue from "
            "methodology strengths, team depth (per the Proposed Team section), and "
            "demonstrated understanding of this tender's sector, and use the market research "
            "to position us against the likely competitors it names. ~250-350 words."
        )

    return [
        {
            "key": "executive_summary",
            "title": "Executive Summary",
            "instructions": (
                "~150-250 words. Who we are, what we're proposing, and the single strongest, "
                "most specific reason we're the right fit for THIS tender (not a generic "
                "capability statement). Flowing prose, no sub-bullets."
            ),
        },
        {
            "key": "understanding_of_requirements",
            "title": "Understanding of the Requirements",
            "instructions": (
                "~300-400 words. Restate the scope, deliverables, deadlines, budget, and "
                "evaluation criteria in your own words to demonstrate genuine comprehension — "
                "reference the actual figures/dates/deliverables above, not vague summaries. "
                "Explicitly call out at least one real ambiguity or risk you notice in the "
                "tender itself (e.g. missing detail on current systems, a tight deadline "
                "relative to scope, an unclear integration requirement)."
            ),
        },
        {
            "key": "proposed_approach",
            "title": "Proposed Approach & Methodology",
            "instructions": (
                "~500-700 words. Break into named phases/workstreams that map onto the actual "
                "deliverables listed above (e.g. Discovery & Requirements, Design & "
                "Architecture, Build, Data Migration, Testing & UAT, Training, Deployment, "
                "Support — adapt to what this tender actually needs). For EACH phase: describe "
                "concretely what will be done, the key activities, and what 'done' looks like. "
                "Explain the reasoning behind the approach, don't just name the phases. This is "
                "the longest, most technical section — go deep, not wide."
            ),
        },
        {
            "key": "work_plan",
            "title": "Indicative Work Plan / Timeline",
            "instructions": (
                "~200-300 words plus a Markdown table with columns Phase | Duration | Key "
                "Milestones, consistent with the project duration stated in the requirements "
                "above. After the table, write a paragraph on sequencing and dependencies "
                "between phases — be specific about which phases block which, not just 'each "
                "phase builds on the last.'"
            ),
        },
        {
            "key": "risk_management",
            "title": "Risk Management & Quality Assurance",
            "instructions": (
                "~200-300 words. Identify 3-5 CONCRETE risks specific to THIS tender (drawing "
                "on its actual scope/deliverables/integration requirements above — not a "
                "generic risk-register list) and a specific mitigation for each. Then describe "
                "how quality will be assured throughout (testing strategy, review gates, "
                "acceptance criteria) — specific to the deliverables above, not boilerplate."
            ),
        },
        {
            "key": "proposed_team",
            "title": "Proposed Team (Profils Proposés)",
            "instructions": team_instructions,
        },
        {
            "key": "why_us",
            "title": "Why Us",
            "instructions": why_us_instructions,
        },
    ]


def _generate_section(client: AnythingLLMClient, workspace_slug: str, spec: dict,
                       shared_context: dict, attempt: int) -> str:
    prompt = _SHARED_PREAMBLE.format(
        section_title=spec["title"],
        section_instructions=spec["instructions"],
        **shared_context,
    )
    # Unique session per section+attempt so mode="chat"'s rolling history
    # never carries content over from another section or a prior retry —
    # see module docstring for why this matters here.
    session_id = f"gen-{workspace_slug}-{spec['key']}-{attempt}"
    response = client.chat(workspace_slug, prompt, mode="chat", session_id=session_id)
    return response.strip()


def generation_agent(state: dict) -> dict:
    if not state.get("is_verified"):
        return {}

    client = AnythingLLMClient()
    workspace_slug = state["workspace_slug"]
    requirements = state.get("requirements", {})
    search_query = requirements.get("scope_summary") or "technical proposal requirements"

    project_references = _search_company_knowledge(client, REFERENCES_WORKSPACE, search_query)
    cv_excerpts = _search_company_knowledge(
        client, CVS_WORKSPACE, f"consultant profile relevant to: {search_query}"
    )
    past_proposals = _search_company_knowledge(
        client, PROPOSALS_WORKSPACE, f"past proposal similar to: {search_query}"
    )

    has_cv_excerpts = "none found" not in cv_excerpts
    has_references = "none found" not in project_references

    shared_context = {
        "requirements": _fmt_requirements(requirements),
        "research_summary": state.get("research_summary", "(no research available)"),
        "project_references": project_references,
        "cv_excerpts": cv_excerpts,
        "past_proposals": past_proposals,
    }

    grounding_text = _build_grounding_text(shared_context)

    sections = _section_specs(has_cv_excerpts, has_references)
    attempts = state.get("generation_attempts", 0) + 1
    section_texts = {}
    section_errors = []

    # Sections have no data dependency on each other, so fire all 7 calls
    # concurrently instead of waiting on each one sequentially.
    with ThreadPoolExecutor(max_workers=len(sections)) as pool:
        future_to_spec = {
            pool.submit(_generate_section, client, workspace_slug, spec, shared_context, attempts): spec
            for spec in sections
        }
        for future in as_completed(future_to_spec):
            spec = future_to_spec[future]
            try:
                text = future.result()
                if not text:
                    raise ValueError("empty response")
            except Exception as e:
                error_msg = f"Generation failed for section '{spec['title']}': {e}"
                logger.warning(error_msg)
                section_errors.append(error_msg)
                text = f"[This section could not be generated: {e}]"
            section_texts[spec["key"]] = text

    # Reassemble in the original, fixed section order (thread completion
    # order is nondeterministic).
    rendered = [f"## {spec['title']}\n\n{section_texts[spec['key']]}" for spec in sections]
    draft = "\n\n".join(rendered)

    if section_errors and len(section_errors) == len(sections):
        # Every single section failed — treat this the same as the old
        # hard-failure path so the graph's retry/error handling still works.
        return {
            "draft_proposal": "",
            "draft_sections": section_texts,
            "generation_grounding": grounding_text,
            "generation_attempts": attempts,
            "errors": section_errors,
        }

    result = {
        "draft_proposal": draft,
        "draft_sections": section_texts,
        "generation_grounding": grounding_text,
        "generation_attempts": attempts,
    }
    if section_errors:
        # Partial failure — keep the draft (other sections are still good)
        # but surface which section(s) need a human look or a re-run.
        result["errors"] = section_errors
    return result