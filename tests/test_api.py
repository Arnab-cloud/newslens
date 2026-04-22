import pytest
from fastapi.testclient import TestClient

from newslens.core.mock_engine import MockEngine
from newslens.core.model import NewsLens
from newslens.server.api import app, lens_storage


@pytest.fixture(autouse=True)
def setup_mock_model():
    mock_engine = MockEngine()
    lens_storage["model"] = NewsLens(engine=mock_engine)
    yield
    lens_storage.clear()


client = TestClient(app)


def test_api_health():
    """Test the base health check."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_api_summarize_success():
    """Test standard summarization with the correct schema."""
    payload = {
        "article": "This is a test article content.",
        "max_tokens": 512,
        "temperature": 0.7,
    }
    response = client.post("/summarize", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert "MOCK_SUMMARY" in data["summary"]


def test_api_batch_summarize():
    """Verify batch processing works."""
    payload = {"articles": ["Article 1", "Article 2"], "max_tokens": 512}
    response = client.post("/summarize/batch", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert len(data["summaries"]) == 2
    assert "MOCK_SUMMARY" in data["summaries"][0]
