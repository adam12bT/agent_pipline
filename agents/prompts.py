"""
Prompt Depot
-------------
Single source of truth for every LLM-facing prompt used across the agent
pipeline. Every agent that needs to call an LLM imports its prompt(s) from
here instead of defining its own inline string.
"""

# Extraction Agent
EXTRACTION_PROMPT = """Two separately labelled sources are provided above: tender excerpts and \
response-template excerpts. Extract tender facts ONLY from the tender excerpts and template rules \
ONLY from the response-template excerpts. Respond with ONLY a valid JSON object (no markdown \
fences, no extra text):

{
  "scope_summary": "2-3 sentence summary of what work is being requested",
  "deliverables": ["list", "of", "expected", "deliverables"],
  "technical_constraints": ["technologies, integrations, security, hosting, standards, or performance constraints"],
  "contractual_constraints": ["eligibility, legal, commercial, warranty, SLA, or contractual obligations"],
  "mandatory_requirements": ["requirements explicitly described as mandatory, required, shall, or must"],
  "deadlines": {"submission_deadline": "date if stated, else null", "project_duration": "if stated, else null"},
  "budget": "budget or price range if stated, else null",
  "evaluation_criteria": ["list of how proposals will be scored"],
  "selection_method": "e.g. QCBS, QBS, LCS, if stated, else null",
  "response_template": {
    "required_sections": ["exact section titles required by the response template"],
    "section_order": ["exact section titles in their required order"],
    "instructions": ["content instructions attached to individual sections"],
    "formatting_requirements": ["page limits, fonts, tables, annexes, language, or other formatting rules"]
  }
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

CLIENT RESPONSE TEMPLATE EXCERPTS:
{response_template_excerpts}

EXTRACTED RESPONSE TEMPLATE RULES:
{response_template_rules}

REVISION FEEDBACK FROM THE PREVIOUS QUALITY REVIEW:
{revision_feedback}

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

MANDATORY PROPOSAL STRUCTURE:
{proposal_structure}

Write a complete, substantive Markdown section beneath every heading above. Reproduce each \
heading exactly, including any numbering, accents, and wording supplied by the client. Do not \
replace client headings with the default English structure, omit sections, merge sections, or \
add an alternative top-level outline. Map the tender requirements, deliverables, methodology, \
timeline, risks, quality controls, security measures, team evidence, and commercial constraints \
into the most appropriate client-defined sections.

For team content, use ONLY the CV excerpts above. If no CV excerpts were provided, include the \
clearly labelled placeholder "[TEAM PROFILES TO BE COMPLETED — no matching CVs found in the \
company knowledge base]" in the relevant client section instead of inventing names, roles, or \
experience.

Do not invent specific figures, dates, project names, or consultant names that are not present \
in the material above or the tender document itself — leave a clear placeholder like \
[TO BE CONFIRMED] instead of making something up. Write in a professional, confident register \
appropriate for a formal procurement submission."""


# Quality Agent: groundedness and coherence evaluator
QUALITY_GROUNDING_PROMPT_TEMPLATE = """You are an evidence-grounding reviewer for a formal \
technical proposal. Compare the proposal against the exact evidence that was supplied to its \
writer. Do not use outside knowledge. Distinguish factual claims (dates, budgets, requirements, \
credentials, named people/projects, competitor facts) from clearly labelled recommendations, \
plans, assumptions, and placeholders.

EVIDENCE AVAILABLE TO THE WRITER:
{evidence}

PROPOSAL TO REVIEW:
{draft}

Return ONLY valid JSON using this schema:
{{
  "groundedness_score": 0.0,
  "coherence_score": 0.0,
  "unsupported_claims": [
    {{"claim": "exact or concise claim", "reason": "why the evidence does not support it"}}
  ],
  "contradictions": [
    {{"claim": "proposal claim", "evidence": "conflicting evidence"}}
  ],
  "coherence_issues": ["internal inconsistency, impossible sequence, or requirement mismatch"],
  "notes": ["short reviewer note"]
}}

Scores must be numbers from 0 to 1. Groundedness measures whether factual claims are supported \
by the evidence. Coherence measures internal consistency and consistency with tender constraints. \
Facts copied from the tender requirements (including its duration, warranty, budget, dates, and \
constraints) are supported claims; do not demand separate proof that the bidder can comply with \
them. A contradiction requires evidence that directly conflicts with the proposal—not merely an \
absence of capability evidence. Do not penalize future-tense delivery plans merely because they \
have not happened yet. Explicit placeholders in square brackets, including TEAM PROFILES TO BE \
COMPLETED and TO BE CONFIRMED, are disclosures of missing information and must not be listed as \
unsupported factual claims. Do penalize invented company experience, CV details, contract facts, \
dates, amounts, certifications, and named projects. Keep each list concise and include only \
material issues."""
