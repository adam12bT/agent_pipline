"""WebResearch port implemented by the GPT Researcher flow."""

import asyncio


class GPTResearcherAdapter:
    def research(self, query: str) -> str:
        # Lazy import keeps the optional research dependency out of agents that
        # never request this adapter.
        from rfp.agents.research.implementation import _run_research

        return asyncio.run(_run_research(query))
