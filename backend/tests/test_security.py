from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_requires_no_key():
    resp = client.get("/health")
    assert resp.status_code == 200


def test_missing_api_key_rejected():
    resp = client.post("/api/v1/analyze-finding", json={"finding_text": "x"})
    assert resp.status_code == 401


def test_invalid_api_key_rejected():
    resp = client.post(
        "/api/v1/analyze-finding",
        json={"finding_text": "x"},
        headers={"X-Internal-Api-Key": "wrong-key"},
    )
    assert resp.status_code == 401


# --- provider-switch mutates global runtime state -- must not be callable by
# an unauthenticated caller (it directly controls which LLM provider/model
# every subsequent investigation uses, a governance-sensitive operation). ---

def test_provider_switch_requires_api_key():
    resp = client.post("/api/v1/provider", json={"provider": "ollama"})
    assert resp.status_code == 401


def test_provider_switch_rejects_wrong_key():
    resp = client.post(
        "/health/provider", json={"provider": "ollama"},
        headers={"X-Internal-Api-Key": "wrong-key"},
    )
    assert resp.status_code == 401


def test_openrouter_key_never_in_openapi_schema():
    resp = client.get("/openapi.json")
    assert "OPENROUTER_API_KEY" not in resp.text
    assert "test-openrouter-key" not in resp.text
