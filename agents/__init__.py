"""
Makes the 5 pipeline nodes importable as `from agents import verifier_agent, ...`
(exactly what graph.py expects) while keeping each agent in its own file.
"""

from .verifier_agent import verifier_agent
from .extraction_agent import extraction_agent
from .research_agent import research_agent
from .generation_agent import generation_agent
from .quality_agent import quality_agent
from .security_agent import security_agent
__all__ = [
    "verifier_agent",
    "extraction_agent",
    "research_agent",
    "generation_agent",
    "quality_agent",
    "security_agent",
]
