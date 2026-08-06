"""
Prompt Depot
-------------
Single source of truth for every LLM-facing prompt used across the agent
pipeline. Every agent that needs to call an LLM imports its prompt(s) from
here instead of defining its own inline string.
"""

# Extraction Agent
EXTRACTION_PROMPT = """Based ONLY on the tender document excerpts provided, extract the following \
information and respond with ONLY a valid JSON object (no markdown fences, no extra text):

{
  "scope_summary": "2-3 sentence summary of what work is being requested",
  "deliverables": ["list", "of", "expected", "deliverables"],
  "deadlines": {"submission_deadline": "date if stated, else null", "project_duration": "if stated, else null"},
  "budget": "budget or price range if stated, else null",
  "evaluation_criteria": ["list of how proposals will be scored"],
  "selection_method": "e.g. QCBS, QBS, LCS, if stated, else null"
}

If a field cannot be found in the document, use null or an empty list — do not guess."""


# Research Agent
RESEARCH_SCOPE_PROMPT = """Based ONLY on the tender document provided, answer in ONE short \
sentence (max ~40 words, no markdown, no preamble): what specific product, service, \
or work is being procured, in what sector/domain, and what are the 1-2 most technically \
or regulatorily distinctive requirements (e.g. a specific integration, an offline/mobile \
requirement, a named compliance regime)? Be concrete rather than generic — prefer \
"a national health-exchange API integration" over "system integration"."""

RESEARCH_BUDGET_PROMPT = """Based ONLY on the tender document provided, state the total \
budget or price ceiling if one is mentioned (include currency and amount only, \
e.g. "USD 380,000"). If no budget or price range is stated anywhere in the \
document, respond with exactly: none stated"""

RESEARCH_FALLBACK_SCOPE = "the scope of this tender"
RESEARCH_FALLBACK_BUDGET = "none stated"

RESEARCH_QUERY_BASE = (
    "market landscape and competing firms/consultants for a project involving: {scope}."
)
RESEARCH_QUERY_BUDGET_CLAUSE = (
    " The project budget is approximately {budget} — prioritize firms and "
    "consultancies realistically sized to compete for a contract at this budget "
    "level, not large enterprise vendors whose typical engagements are far larger."
)
RESEARCH_QUERY_SELECTION_METHOD_CLAUSE = " Procurement is via {selection_method}."
RESEARCH_QUERY_GUARDRAILS = (
    " Identify likely competitors and their typical positioning. For the 'recent "
    "similar awarded projects' section: only name a specific project, client, or "
    "contract if you can point to a real, findable source confirming it (a news "
    "article, press release, or procurement notice with a date) — do not infer or "
    "guess a plausible-sounding award from a firm's homepage or service description. "
    "If no verifiable recent award can be found for a given firm, say so explicitly "
    "rather than describing a generic, undated project. In the references/sources "
    "list, include ONLY sources that are actually cited inline in the report body — "
    "do not list every page visited during research if it wasn't used as a citation."
)


# Generation Agent
GENERATION_PROMPT_TEMPLATE = """You are a senior bid writer producing a FULL technical proposal \
report in response to the tender document below. This is a formal deliverable that \
will be submitted to the client for evaluation — not a cover letter and not an executive summary. \
Each section below must be written in complete, substantive paragraphs with real analysis and \
detail grounded in the material provided. Shallow, generic filler is worse than a shorter section \
that is actually specific to this tender.

RELEVANT TENDER DOCUMENT EXCERPTS:
{tender_excerpts}

EXTRACTED REQUIREMENTS:
{requirements}

MARKET / COMPETITOR RESEARCH:
{research_summary}

RELEVANT PAST PROJECT REFERENCES (from our company's own project history — use these for \
the "Why Us" / track record section):
{project_references}

RELEVANT CONSULTANT CVs (use these — and ONLY these — to write the "Proposed Team / Profils \
Proposés" section; do not invent names, titles, or years of experience that aren't in this list):
{cv_excerpts}

RELEVANT PAST PROPOSALS (for tone/structure reference only — do not copy text verbatim, \
just match the general style and level of detail):
{past_proposals}

Write the proposal in Markdown with these sections, each meeting the stated minimum depth. \
Treat the minimums as a floor: expand further wherever the tender's requirements give you real \
material to work with.

1. **Executive Summary** (~150-250 words) — who we are, what we're proposing, and the single \
strongest reason we're the right fit. No sub-bullets here; flowing prose.

2. **Understanding of the Requirements** (~300-400 words) — restate the scope, deliverables, \
deadlines, budget, and evaluation criteria in your own words to demonstrate genuine \
comprehension. Explicitly call out any ambiguities or risks you notice in the tender itself.

3. **Proposed Approach & Methodology** (~500-700 words) — break this into named phases or \
workstreams (e.g. Discovery & Requirements, Design & Architecture, Build, Data Migration, \
Testing & UAT, Training, Deployment, Support) that map onto the actual deliverables listed \
above. For each phase, describe concretely what will be done, the key activities, and what \
"done" looks like. Do not just name the phases — explain the reasoning behind the approach.

4. **Indicative Work Plan / Timeline** (~200-300 words plus a Markdown table) — provide a table \
with columns Phase | Duration | Key Milestones, consistent with the project duration stated in \
the requirements. Follow the table with a short paragraph on sequencing and dependencies.

5. **Risk Management & Quality Assurance** (~200-300 words) — identify 3-5 concrete risks \
specific to this tender (e.g. data migration integrity, staff adoption across many sites, \
integration with external systems) and the mitigation for each, plus how quality will be \
assured throughout (testing strategy, review gates, acceptance criteria).

6. **Proposed Team (Profils Proposés)** — based ONLY on the CV excerpts above. If no CV \
excerpts were provided, write exactly: "[TEAM PROFILES TO BE COMPLETED — no matching CVs found \
in the company knowledge base]" instead of inventing anyone. If excerpts were provided, give \
each person a short paragraph: role on this project, relevant background, and why they fit \
this specific tender.

7. **Why Us** (~250-350 words) — reference the past project references above if any were found, \
and the market research for competitive positioning against likely competitors. If no past \
references were found, keep this section general rather than inventing specific past projects, \
but still make a substantive case (methodology strengths, team depth, understanding of the \
sector) rather than a one-line platitude.

Do not invent specific figures, dates, project names, or consultant names that are not present \
in the material above or the tender document itself — leave a clear placeholder like \
[TO BE CONFIRMED] instead of making something up. Write in a professional, confident register \
appropriate for a formal procurement submission."""