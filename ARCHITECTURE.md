# RFP pipeline architecture

The synchronous application now separates orchestration policy from agent
work. Redis/Celery is deliberately deferred; replacing the synchronous graph
runner with a worker will not change agent contracts.

```text
FastAPI / CLI
      |
      v
rfp.orchestration.graph
      |
      +--> Verifier --> Extraction --> Research --> Generation
      |                                      --> Security --> Quality
      |
      +--> projections.py --> Pydantic Input --> agent --> Pydantic Output
      |
      +--> routing.py (branching, retries, terminal status)
      |
      +--> contracts/ports.py <--- adapters (AnythingLLM, research, scanners)
```

## Dependency rules

1. Agent callers pass only the fields declared by the agent input contract.
2. Agent results are validated before entering pipeline state.
3. Agents return facts and verdicts; `orchestration/routing.py` owns retries,
   branches, and terminal statuses.
4. External systems are represented by protocols in `contracts/ports.py` and
   implemented by adapters.
5. `rfp.orchestration.graph` is the single call site for every agent. Replacing
   one local `run()` function with an HTTP client changes only this composition.
6. Research receives extracted scope through its contract and owns only web
   research; it has no AnythingLLM or extractor dependency.

## Package layout

The canonical implementation is installable from `src/rfp`. Every agent owns
`agent.py`, `contract.py`, `config.py`, `agent.toml`, `main.py`, `__main__.py`,
and a test package. Run one independently with:

```powershell
python -m rfp.agents.quality --in quality-input.json --out quality-output.json
```

Every saved real-run pair can also be verified without external side effects:

```powershell
python -m rfp.agents.quality `
  --in tests/fixtures/agents/quality/input.json `
  --expected tests/fixtures/agents/quality/output.json `
  --contract-only
```

Dependencies are declared as `pyproject.toml` extras, such as
`.[research]`, `.[quality]`, `.[api]`, or `.[full]`.

Saved contract-compatible input/output examples live under
`tests/fixtures/agents/<agent>/`. A golden namespaced-to-public state snapshot
under `tests/fixtures/pipeline/` protects the existing API response shape.

## API projection

Internal state is namespaced by agent (`verifier`, `extraction`, `research`,
`generation`, `security`, `quality`, and `control`). The API flattens that state
at its boundary, so the existing React frontend and download endpoint keep the
same response contract. Application code imports the canonical `rfp.*`
packages.

## Deferred worker step

The future Celery task will call the same orchestrator. Redis, persistent run
storage, worker queues, and SSE delivery are not part of this architecture
slice.
