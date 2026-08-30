import os

os.environ.setdefault("INTERNAL_API_KEY", "test-key")
os.environ.setdefault("OPENROUTER_API_KEY", "test-openrouter-key")

# `financial_semantic_reasoning_enabled` defaults to True in production
# (app/config.py) now that a real local Ollama provider is confirmed
# reachable and the semantic financial prompt/schema has been validated
# against it. The regression suite, however, must stay fast and
# deterministic by default -- most test files patch get_llm_client to
# None only for understanding/investigation_planner/core_synthesis
# (see e.g. tests/test_golden_20_scenarios.py's _run_agent_pipeline),
# never for report_generator.py's own financial-reasoning call, so
# leaving the production default on here would make every test that
# reaches report generation silently attempt a real ~20s Ollama call.
# Dedicated real-Ollama financial tests (tests/test_financial_semantic_
# real_ollama.py) explicitly re-enable this for themselves.
os.environ.setdefault("FINANCIAL_SEMANTIC_REASONING_ENABLED", "false")

# Same rationale for Remediation Cost Estimation (app.remediation): its
# report_generator.py call would otherwise attempt a real LLM request on every
# test that reaches report generation. Dedicated remediation tests
# (tests/test_remediation_cost_*.py) call the engine directly with a
# FakeLLMClient and are unaffected by this flag.
os.environ.setdefault("REMEDIATION_COST_ESTIMATION_ENABLED", "false")

# `canonical_semantic_llm_primary` is ON in production (app/config.py) -- the
# LLM canonical interpretation is the primary semantic authority and the
# deterministic pipeline is the fail-closed floor. The regression suite pins it
# OFF so the ~3000-test baseline stays deterministic and fast (otherwise every
# test reaching understand_finding_node would attempt a real ~8s canonical LLM
# call). Dedicated LLM-primary tests (tests/test_llm_primary_*.py) monkeypatch
# it True and install recorded responses.
os.environ.setdefault("CANONICAL_SEMANTIC_LLM_PRIMARY", "false")

import pytest


@pytest.fixture(autouse=True)
def _reset_llm_execution_context():
    """The per-request frozen LLM route + correlation id + last-call metadata are
    ContextVars. Reset them around every test so one test's `begin_request` can
    never leak its provider/model into the next (which would break the factory's
    'resolve from settings' path)."""
    from app.services.llm import execution as _exec
    from app.services.llm import call_metadata as _meta
    _exec._execution_config.set(None)
    _exec._request_id.set(None)
    _meta._last_call_metadata.set({})
    yield
    _exec._execution_config.set(None)
    _exec._request_id.set(None)
    _meta._last_call_metadata.set({})
