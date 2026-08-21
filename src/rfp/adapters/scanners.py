"""Output-scanner adapters for the security and quality agents."""


class ConfiguredSecurityScanner:
    def status(self) -> dict:
        from rfp.agents.security.implementation import security_scanner_status

        return security_scanner_status()

    def scan(self, text: str) -> dict:
        from rfp.agents.security.implementation import (
            _check_naive_pii,
            _run_llm_guard,
            security_scanner_status,
        )

        mode = security_scanner_status()["mode"]
        if mode == "llm_guard":
            return _run_llm_guard(text)
        if mode == "regex_fallback":
            return _check_naive_pii(text)
        return {}


class ConfiguredQualityScanner:
    def scan(self, text: str) -> dict:
        from rfp.agents.quality.implementation import _LLM_GUARD_AVAILABLE, _run_llm_guard

        return _run_llm_guard(text) if _LLM_GUARD_AVAILABLE else {}
