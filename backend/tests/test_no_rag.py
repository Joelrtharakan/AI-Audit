"""RAG/embeddings/vector-store were removed entirely -- these guard against
any of it creeping back into the LLM-only pipeline or its dependencies."""

import pathlib

BACKEND_ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_no_rag_modules_on_disk():
    removed_paths = [
        "app/services/rag_service.py",
        "app/services/similarity_service.py",
        "app/services/vector_store.py",
        "app/services/embedding_service.py",
        "app/services/document_ingestion.py",
        "app/services/chunking.py",
        "app/services/official_source_ingestion.py",
        "app/services/capa_suggester.py",
        "app/services/synthetic_data_service.py",
        "app/routers/rag.py",
        "app/routers/similarity.py",
        "app/routers/capa.py",
        "app/routers/documents.py",
        "app/models/rag.py",
        "app/models/responses.py",
        "app/config",
        "data",
    ]
    for rel_path in removed_paths:
        assert not (BACKEND_ROOT / rel_path).exists(), f"{rel_path} should have been removed"


def test_finding_analysis_pipeline_has_no_rag_imports():
    for module_path in ["app/services/finding_analysis_service.py", "app/services/prompt_builder.py"]:
        source = (BACKEND_ROOT / module_path).read_text(encoding="utf-8")
        for forbidden in ("rag_service", "vector_store", "embedding_service", "chromadb", "sentence_transformers"):
            assert forbidden not in source, f"{module_path} references {forbidden}"


def test_only_approved_routes_exist():
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    paths = set(client.get("/openapi.json").json()["paths"].keys())

    assert "/health" in paths
    assert "/api/v1/analyze-finding" in paths
    assert "/api/v1/investigate" in paths
    for forbidden in ("/api/v1/suggest-capa", "/api/v1/rag/search", "/api/v1/rag/answer", "/api/v1/check-similar-findings"):
        assert forbidden not in paths
