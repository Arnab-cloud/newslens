# from fastapi.testclient import TestClient
# from mock_engine import MockEngine

# from newslens.core.model import NewsLens
# from newslens.server.api import app, lens_storage

# # Inject mock into the app storage
# lens_storage["model"] = NewsLens(engine=MockEngine())

# client = TestClient(app)


# def test_api_summarize():
#     response = client.post("/summarize", json={"text": "This is a test"})
#     print(response.json())
#     assert response.status_code == 200
#     assert "summary" in response.json()
#     assert "MOCK_SUMMARY" in response.json()["summary"]


# def test_api_health():
#     response = client.get("/health")
#     assert response.status_code == 200
#     assert response.json()["status"] == "ready"
#
import pytest
from fastapi.testclient import TestClient
from mock_engine import MockEngine

from newslens.core.model import NewsLens
from newslens.server.api import app, lens_storage


# 1. Setup the client
# We inject the mock engine before initializing the TestClient
@pytest.fixture(autouse=True)
def setup_mock_model():
    mock_engine = MockEngine()
    # Ensure the model is injected into the server's storage
    lens_storage["model"] = NewsLens(engine=mock_engine)
    yield
    # Cleanup after tests
    lens_storage.clear()


client = TestClient(app)


def test_api_health():
    """Test the base health check."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_api_summarize_success():
    """Test standard summarization with the correct schema."""
    # Note: Using 'article' as defined in SummarizeRequest schema
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
