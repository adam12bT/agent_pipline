"""
Security Agent Implementation
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

When the optional LLM Guard integration is enabled, this uses bilingual
Presidio PII detection plus LLM Guard's OUTPUT scanner:
  - Presidio (en/fr) -> PII leaking into the draft (e.g. a stray email or
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
and the API run store surfaces `security_report` so
a human can review it.

LLM Guard is disabled by default for lightweight deployments. The active
fallback checks emails and phone numbers with local regular expressions,
so it needs no model download and cannot fail because a remote model was
removed.
"""

import logging
import os
import re

logger = logging.getLogger(__name__)

LLM_GUARD_ENABLED = os.environ.get(
    "LLM_GUARD_ENABLED", "false"
).strip().lower() in {"1", "true", "yes", "on"}
SECURITY_FALLBACK_ENABLED = os.environ.get(
    "SECURITY_FALLBACK_ENABLED", "false"
).strip().lower() in {"1", "true", "yes", "on"}

if LLM_GUARD_ENABLED:
    try:
        from llm_guard.output_scanners import MaliciousURLs
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        _LLM_GUARD_AVAILABLE = True
    except ImportError:  # pragma: no cover - depends on deployment extras
        _LLM_GUARD_AVAILABLE = False
        logger.warning(
            "LLM Guard was enabled but its dependencies are unavailable; "
            "using the lightweight PII fallback."
        )
else:
    _LLM_GUARD_AVAILABLE = False
    logger.info(
        "LLM Guard is disabled; lightweight PII fallback is %s.",
        "enabled" if SECURITY_FALLBACK_ENABLED else "disabled",
    )

# Deliberately simple patterns — only used as a fallback if llm-guard isn't
# installed. Do not treat this as a real safety guarantee on its own.
_NAIVE_PII_PATTERNS = {
    "email": re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
    "phone": re.compile(r"\b\d{2,4}[-.\s]?\d{2,4}[-.\s]?\d{2,4}[-.\s]?\d{0,4}\b"),
}

_scanners_cache = None
LLM_GUARD_FAIL_CLOSED = os.environ.get(
    "LLM_GUARD_FAIL_CLOSED", "true"
).strip().lower() not in {"0", "false", "no", "off"}


def llm_guard_available() -> bool:
    return _LLM_GUARD_AVAILABLE


def security_scanner_status() -> dict:
    if _LLM_GUARD_AVAILABLE:
        mode = "llm_guard"
    elif SECURITY_FALLBACK_ENABLED:
        mode = "regex_fallback"
    else:
        mode = "disabled"
    return {
        "mode": mode,
        "llm_guard_enabled": LLM_GUARD_ENABLED,
        "llm_guard_available": _LLM_GUARD_AVAILABLE,
        "fallback_enabled": SECURITY_FALLBACK_ENABLED,
    }


def _get_scanners():
    """Lazily build & cache the LLM Guard scanners (they load ML models on
    construction, so build once per process, on first use)."""
    global _scanners_cache
    if _scanners_cache is None:
        _scanners_cache = {
            "pii": _PresidioSensitiveScanner(),
            "malicious_urls": MaliciousURLs(threshold=0.5),
        }
    return _scanners_cache


class _PresidioSensitiveScanner:
    """Expose LLM Guard's scan contract using only the required en/fr models.

    LLM Guard 0.3.16's Sensitive scanner initializes an internal language
    list that includes Chinese. Explicit Presidio configuration avoids that
    runtime download and keeps PII analysis bilingual.
    """

    def __init__(self, threshold: float = 0.5):
        configuration = {
            "nlp_engine_name": "spacy",
            "models": [
                {"lang_code": "en", "model_name": "en_core_web_sm"},
                {"lang_code": "fr", "model_name": "fr_core_news_sm"},
            ],
        }
        provider = NlpEngineProvider(nlp_configuration=configuration)
        nlp_engine = provider.create_engine()
        self._analyzer = AnalyzerEngine(
            nlp_engine=nlp_engine,
            supported_languages=["en", "fr"],
        )
        self._threshold = threshold

    def scan(self, prompt: str, output: str):
        del prompt
        findings = []
        for language in ("en", "fr"):
            findings.extend(
                self._analyzer.analyze(
                    text=output,
                    language=language,
                    score_threshold=self._threshold,
                )
            )
        risk_score = max((item.score for item in findings), default=0.0)
        return output, not findings, risk_score


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
    findings = {}

    try:
        scanners = _get_scanners()
    except Exception as exc:
        logger.exception("Security scanners failed to initialize")
        findings.update(_check_naive_pii(draft))
        if LLM_GUARD_FAIL_CLOSED:
            findings["scanner_initialization_error"] = str(exc)[:300]
        return findings

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


def security_agent(state: dict, *, scanner=None) -> dict:
    if not state.get("is_verified"):
        return {}

    draft = state.get("draft_proposal", "")
    scanner_status = security_scanner_status()
    if scanner is not None:
        findings = scanner.scan(draft)
        scanner_status = (
            scanner.status()
            if hasattr(scanner, "status")
            else {"mode": "injected", "adapter": type(scanner).__name__}
        )
    elif scanner_status["mode"] == "llm_guard":
        findings = _run_llm_guard(draft)
    elif scanner_status["mode"] == "regex_fallback":
        findings = _check_naive_pii(draft)
    else:
        findings = {}
    passed = not findings

    if scanner_status["mode"] == "disabled":
        notes = [
            "Security scanning is disabled. No automated security checks were performed."
        ]
    elif passed:
        notes = ["Security scan clean."]
    else:
        notes = [
            f"Security scan flagged: {findings}. Escalating to human review — "
            "no automatic retry."
        ]

    security_report = {
        "findings": findings,
        "notes": notes,
        "scanner": scanner_status,
        "scan_performed": scanner_status["mode"] != "disabled",
    }

    result = {
        "security_passed": passed,
        "security_report": security_report,
    }
    if not passed:
        logger.warning("Security scan BLOCKED this draft, escalating to human review: %s", findings)
    else:
        logger.info("Security scan passed.")
    return result
