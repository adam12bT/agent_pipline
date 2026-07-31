"""
Thin Python client for the stripped-down AnythingLLM server (the Node.js
backend in ../anything-llm-lightweight/server).

Routes and request/response shapes below were confirmed directly against
that server's source code (server/endpoints/api/workspace/index.js and
server/endpoints/api/document/index.js) — not guessed.

NOTE: API key auth was disabled on that server fork (see validApiKey.js),
so no Authorization header is required here. If you re-enable auth on
the server later, add the Bearer header back in _headers().
"""

import os
import requests

ANYTHINGLLM_BASE_URL = os.environ.get("ANYTHINGLLM_BASE_URL", "http://localhost:3001/api")


class AnythingLLMClient:
    def __init__(self, base_url: str = ANYTHINGLLM_BASE_URL):
        self.base_url = base_url.rstrip("/")

    def _headers(self):
        # Auth disabled on the server fork — nothing to add here for now.
        return {}

    def get_workspace(self, slug: str) -> dict | None:
        """
        GET /v1/workspace/:slug — NOTE: despite the name, this returns a
        LIST under the "workspace" key (confirmed from the server source),
        not a single object. Empty list = doesn't exist. Returns the first
        match dict, or None if not found.
        """
        resp = requests.get(
            f"{self.base_url}/v1/workspace/{slug}",
            headers=self._headers(),
            timeout=30,
        )
        resp.raise_for_status()
        matches = resp.json().get("workspace", [])
        return matches[0] if matches else None

    def get_or_create_workspace(self, name: str) -> dict:
        """
        Idempotent workspace creation. IMPORTANT: AnythingLLM's Workspace.new()
        does NOT let you set a custom slug — it always derives the slug from
        `name` via slugify(name, {lower:true}), and if that slug is already
        taken it silently appends a random suffix instead of reusing it
        (confirmed in models/workspace.js). So calling create_workspace()
        twice with the same name creates two DIFFERENT workspaces, not one.

        To make repeated runs safe (e.g. for the shared "company knowledge"
        workspaces that should persist across every tender), always use a
        `name` that is ALREADY a valid slug (lowercase, hyphens, no spaces —
        e.g. "company-past-proposals") so slugify(name) == name, then check
        for an existing workspace at that exact slug before creating.
        """
        existing = self.get_workspace(name)
        if existing:
            return {"workspace": existing, "created": False}

        created = self.create_workspace(name)
        return {"workspace": created["workspace"], "created": True}

    def create_workspace(self, name: str) -> dict:
        """POST /v1/workspace/new -> returns the created workspace, including its slug."""
        resp = requests.post(
            f"{self.base_url}/v1/workspace/new",
            json={"name": name},
            headers=self._headers(),
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def upload_document(self, file_path: str, workspace_slug: str) -> dict:
        """
        POST /v1/document/upload (multipart) — uploads a file and, via
        addToWorkspaces, embeds it into the given workspace in one call.
        Returns the document metadata list, including each doc's `location`
        (needed if you ever want to add/remove it from other workspaces later).
        """
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f)}
            data = {"addToWorkspaces": workspace_slug}
            resp = requests.post(
                f"{self.base_url}/v1/document/upload",
                files=files,
                data=data,
                headers=self._headers(),
                timeout=120,
            )
        resp.raise_for_status()
        return resp.json()

    def vector_search(self, workspace_slug: str, query: str, top_n: int = 4,
                       score_threshold: float = 0.5) -> list[dict]:
        """
        POST /v1/workspace/:slug/vector-search — direct similarity search,
        no LLM call. Returns a list of {text, metadata, score, ...} chunks.
        Use this when you just need raw relevant passages (e.g. Extraction agent).
        """
        resp = requests.post(
            f"{self.base_url}/v1/workspace/{workspace_slug}/vector-search",
            json={"query": query, "topN": top_n, "scoreThreshold": score_threshold},
            headers=self._headers(),
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("results", [])

    def chat(self, workspace_slug: str, message: str, mode: str = "query",
              session_id: str = "rfp-pipeline") -> str:
        """
        POST /v1/workspace/:slug/chat — runs the message through the LLM,
        grounded in the workspace's embedded documents (RAG).
        mode="query": only answers using retrieved doc chunks, no chit-chat.
        mode="chat": general LLM knowledge + doc context + rolling history.
        Returns just the text response.
        """
        resp = requests.post(
            f"{self.base_url}/v1/workspace/{workspace_slug}/chat",
            json={"message": message, "mode": mode, "sessionId": session_id},
            headers=self._headers(),
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("textResponse", "")
