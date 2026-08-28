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
