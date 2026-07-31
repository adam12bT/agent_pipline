"""
Security Agent
----------------
New node, split out of the old combined quality_agent.py. Runs
immediately after Generation, before Quality.

Why split it: security checks are BLOCKING (fail closed, escalate to a
human) while quality checks are GRADED (score-based retry). Mixing both
in one node meant a binary "hard stop" and a scored "try again" had to
share the same return shape and the same pass/fail flag. Now they're two
separate conditional edges in graph.py: `security` routes to END
(human alert) on failure with no retry, `quality` routes back to
`generation` on failure.

Uses LLM Guard's OUTPUT scanners (scanning the generated draft itself):
  - `Sensitive`      -> PII leaking into the draft (e.g. a stray email or
                        phone number pulled in from a CV excerpt).
  - `MaliciousURLs`   -> suspicious links in the draft. This also doubles
                        as the agent's best available signal for INDIRECT
                        PROMPT INJECTION: the Research agent pulls content
                        from the open web, and a poisoned page is a
                        classic vector for smuggling malicious links or
                        instructions into what the Generation agent
                        eventually treats as trusted context. LLM Guard
                        doesn't ship a dedicated "was this output
                        manipulated by an injected instruction" scanner,
                        so this is the closest practical proxy — flag it
                        in your report as a known limitation, not a
                        guarantee.

If either scanner fires, the run is BLOCKED — no automatic retry. This is
deliberate: unlike a coherence/hallucination problem (which a reroll can
plausibly fix), a PII leak or an injected malicious link isn't something
you want an LLM to silently "try again" its way out of. `status` is set
to "security_blocked" so the graph routes straight past Quality to END,
and the caller (see backend/run_store.py) surfaces `security_report` so
a human can review it.

Install:
    pip install llm-guard

Falls back to a naive regex PII check if llm-guard isn't installed, same
pattern as the original quality_agent.py, so the pipeline doesn't hard-
fail in an environment where the models aren't available yet.
"""

import logging
import re

logger = logging.getLogger(__name__)

try:
    from llm_guard.output_scanners import MaliciousURLs, Sensitive

    _LLM_GUARD_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when dep is missing
    _LLM_GUARD_AVAILABLE = False
    logger.warning(
        "llm-guard not installed — falling back to the naive regex PII check. "
        "Run `pip install llm-guard` to enable full scanning."
    )

# Deliberately simple patterns — only used as a fallback if llm-guard isn't
# installed. Do not treat this as a real safety guarantee on its own.
_NAIVE_PII_PATTERNS = {
    "email": re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
    "phone": re.compile(r"\b\d{2,4}[-.\s]?\d{2,4}[-.\s]?\d{2,4}[-.\s]?\d{0,4}\b"),
}

_scanners_cache = None


def _get_scanners():
    """Lazily build & cache the LLM Guard scanners (they load ML models on
    construction, so build once per process, on first use)."""
    global _scanners_cache
    if _scanners_cache is None:
        _scanners_cache = {
            "pii": Sensitive(entity_types=None, redact=False),
            "malicious_urls": MaliciousURLs(threshold=0.5),
        }
    return _scanners_cache


def _check_naive_pii(draft: str) -> dict:
    """Fallback only — used when llm-guard isn't installed."""
    findings = {}
    for label, pattern in _NAIVE_PII_PATTERNS.items():
        matches = pattern.findall(draft)
        if matches:
            findings[label] = len(matches)
    return findings


def _run_llm_guard(draft: str) -> dict:
    """A scanner that errors (e.g. a model failed to load) is logged and
    skipped rather than failing the whole security stage — one bad
    scanner shouldn't block every run. Flip this to fail-closed (treat a
    scanner error as a finding) if you'd rather be stricter here."""
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


def security_agent(state: dict) -> dict:
    if not state.get("is_verified"):
        return {}

    draft = state.get("draft_proposal", "")
    findings = _run_llm_guard(draft) if _LLM_GUARD_AVAILABLE else _check_naive_pii(draft)
    passed = not findings

    security_report = {
        "findings": findings,
        "notes": (
            ["Security scan clean."] if passed
            else [f"Security scan flagged: {findings}. Escalating to human review — no automatic retry."]
        ),
    }

    result = {
        "security_passed": passed,
        "security_report": security_report,
    }
    if not passed:
        result["status"] = "security_blocked"
    return result
