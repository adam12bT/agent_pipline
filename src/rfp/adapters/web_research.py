"""WebResearch port implemented by the GPT Researcher flow."""

import asyncio
from contextlib import nullcontext
import logging
import os

from providers.telemetry import record_llm_usage_aggregate

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.environ.get(name, str(default))))
    except ValueError:
        logger.warning("Ignoring invalid %s value; using %d", name, default)
        return default


def configure_research_groq_credentials(uses_groq: bool) -> bool:
    """Map the dedicated research key for GPT Researcher's own clients."""
    research_key = os.environ.get("RESEARCH_GROQ_API_KEY")
    if not uses_groq or not research_key:
        return False
    os.environ["GROQ_API_KEY"] = research_key
    logger.info("GPT Researcher configured with RESEARCH_GROQ_API_KEY")
    return True


async def _run_research(query: str) -> str:
    groq_models = (
        os.environ.get("FAST_LLM", ""),
        os.environ.get("SMART_LLM", ""),
        os.environ.get("STRATEGIC_LLM", ""),
    )
    uses_groq = any(model.strip().lower().startswith("groq:") for model in groq_models)
    configure_research_groq_credentials(uses_groq)

    from gpt_researcher import GPTResearcher

    researcher = GPTResearcher(query=query, report_type="research_report")
    # Local models are not constrained by hosted TPM quotas, so give report
    # synthesis enough room to finish. Every value remains configurable.
    hosted_defaults = uses_groq
    researcher.cfg.max_search_results_per_query = _env_int(
        "RESEARCH_MAX_RESULTS_PER_QUERY", 4
    )
    researcher.cfg.max_subtopics = _env_int("RESEARCH_MAX_SUBTOPICS", 3)
    researcher.cfg.total_words = _env_int(
        "RESEARCH_TOTAL_WORDS", 900 if hosted_defaults else 1800
    )
    researcher.cfg.fast_token_limit = _env_int(
        "RESEARCH_FAST_TOKEN_LIMIT", 1200 if hosted_defaults else 3000
    )
    researcher.cfg.smart_token_limit = _env_int(
        "RESEARCH_SMART_TOKEN_LIMIT", 1800 if hosted_defaults else 6000
    )
    researcher.cfg.strategic_token_limit = _env_int(
        "RESEARCH_STRATEGIC_TOKEN_LIMIT", 1200 if hosted_defaults else 3000
    )
    researcher.cfg.summary_token_limit = _env_int(
        "RESEARCH_SUMMARY_TOKEN_LIMIT", 500 if hosted_defaults else 2200
    )

    groq_interval = max(
        0.0, float(os.environ.get("GROQ_MIN_INTERVAL_SECONDS", "30"))
    )
    if uses_groq and groq_interval > 0:
        from langchain_core.rate_limiters import InMemoryRateLimiter

        researcher.cfg.llm_kwargs["rate_limiter"] = InMemoryRateLimiter(
            requests_per_second=1.0 / groq_interval,
            check_every_n_seconds=min(1.0, groq_interval),
            max_bucket_size=1,
        )
        await asyncio.sleep(groq_interval)

    try:
        from langchain_core.callbacks import get_usage_metadata_callback

        usage_context = get_usage_metadata_callback("rfp_research_usage")
    except ImportError:
        usage_context = nullcontext(None)

    with usage_context as usage_callback:
        await researcher.conduct_research()
        report = await researcher.write_report()

    if usage_callback is not None:
        provider_name = "groq" if uses_groq else "research_llm"
        for model, usage in usage_callback.usage_metadata.items():
            record_llm_usage_aggregate(
                provider=provider_name,
                model=str(model),
                prompt_tokens=usage.get("input_tokens"),
                completion_tokens=usage.get("output_tokens"),
                total_tokens=usage.get("total_tokens"),
                source="gpt_researcher_langchain_callback",
            )
    if uses_groq and groq_interval > 0:
        await asyncio.sleep(groq_interval)
    return report


class GPTResearcherAdapter:
    def research(self, query: str) -> str:
        return asyncio.run(_run_research(query))
