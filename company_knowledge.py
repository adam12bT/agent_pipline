"""
Company Knowledge Base
------------------------
Three PERSISTENT AnythingLLM workspaces, created once and reused across
every tender run — separate from the single-use "rfp-<hash>" workspace
the Verifier agent creates fresh for each tender document.

This is what was missing before: the Generation agent had no access to
the company's actual past proposals, CVs, or project references, so it
could only write generic prose. Now it can search these three workspaces
for genuinely relevant past material.

Split into 3 separate workspaces (not 1 combined one) so retrieval stays
precise — e.g. searching for "similar past projects" won't accidentally
pull back an unrelated CV paragraph just because it shares some wording.

NOTE: these names are deliberately already-valid-slugs (lowercase,
hyphens, no spaces) — see the big comment in
AnythingLLMClient.get_or_create_workspace() for why that matters.
"""

from anythingllm_client import AnythingLLMClient

PROPOSALS_WORKSPACE = "company-past-proposals"
CVS_WORKSPACE = "company-cvs"
REFERENCES_WORKSPACE = "company-project-references"

ALL_COMPANY_WORKSPACES = [PROPOSALS_WORKSPACE, CVS_WORKSPACE, REFERENCES_WORKSPACE]


def ensure_company_workspaces(client: AnythingLLMClient | None = None) -> dict:
    """
    Idempotent — safe to call at the start of every pipeline run.
    Creates the 3 workspaces on first run only; on every later run it
    finds them already existing and does nothing.
    Returns {workspace_slug: {"created": bool}} for visibility/logging.
    """
    client = client or AnythingLLMClient()
    result = {}
    for slug in ALL_COMPANY_WORKSPACES:
        outcome = client.get_or_create_workspace(slug)
        result[slug] = {"created": outcome["created"]}
    return result
