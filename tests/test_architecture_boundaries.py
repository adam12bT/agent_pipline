import importlib
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
AGENTS_ROOT = ROOT / "src" / "rfp" / "agents"
AGENTS = ("verifier", "extraction", "research", "generation", "security", "quality")
REQUIRED_FILES = (
    "__init__.py",
    "__main__.py",
    "agent.py",
    "contract.py",
    "config.py",
    "main.py",
    "agent.toml",
)


class ArchitectureBoundaryTests(unittest.TestCase):
    def test_every_agent_has_packaging_surface_and_matching_extra(self):
        pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        extras = pyproject["project"]["optional-dependencies"]
        for name in AGENTS:
            with self.subTest(agent=name):
                folder = AGENTS_ROOT / name
                for filename in REQUIRED_FILES:
                    self.assertTrue((folder / filename).is_file(), filename)
                manifest = tomllib.loads((folder / "agent.toml").read_text(encoding="utf-8"))
                self.assertEqual(manifest["name"], name)
                self.assertEqual(manifest["extra"], name)
                self.assertIn(name, extras)
                self.assertEqual(manifest["contract"], f"rfp.agents.{name}.contract")
                config = importlib.import_module(f"rfp.agents.{name}.config")
                self.assertEqual(manifest["env"], config.ENV_KEYS)

    def test_agents_do_not_import_sibling_agents(self):
        for name in AGENTS:
            source = "\n".join(
                path.read_text(encoding="utf-8")
                for path in (AGENTS_ROOT / name).glob("*.py")
            )
            for sibling in set(AGENTS) - {name}:
                self.assertNotIn(f"rfp.agents.{sibling}", source)

    def test_research_has_only_web_side_effect_port(self):
        manifest = tomllib.loads(
            (AGENTS_ROOT / "research" / "agent.toml").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["ports"], ["WebResearch"])
        source = (AGENTS_ROOT / "research" / "implementation.py").read_text(encoding="utf-8")
        self.assertNotIn("AnythingLLMClient", source)
        self.assertNotIn("GPTResearcher", source)

    def test_agents_do_not_return_orchestration_status(self):
        for name in AGENTS:
            source = (AGENTS_ROOT / name / "implementation.py").read_text(encoding="utf-8")
            self.assertNotIn('"status":', source, name)

    def test_legacy_root_integration_modules_are_removed(self):
        for filename in (
            "anythingllm_client.py",
            "company_knowledge.py",
            "extractor_client.py",
            "retrieval.py",
        ):
            self.assertFalse((ROOT / filename).exists(), filename)

    def test_pyproject_extras_are_the_only_dependency_install_source(self):
        self.assertFalse((ROOT / "requirements.txt").exists())
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn('pip install --no-cache-dir ".[full]"', dockerfile)
        self.assertNotIn("requirements.txt", dockerfile)


if __name__ == "__main__":
    unittest.main()
