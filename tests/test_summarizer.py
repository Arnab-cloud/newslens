import pytest

from newslens.core.mock_engine import MockEngine
from newslens.core.model import NewsLens


@pytest.fixture
def lens():
    mock = MockEngine()
    return NewsLens(engine=mock)


def test_summarization_flow(lens):
    result = lens.summarize("This is a long article.")
    assert "MOCK_SUMMARY" in result


@pytest.mark.asyncio
async def test_asummarize_batch():
    mock = MockEngine()
    lens = NewsLens(engine=mock)

    articles = ["Article A", "Article B", "Article C"]
    summaries = await lens.asummarize_batch(articles)

    assert len(summaries) == 3
    assert all("MOCK_SUMMARY" in s for s in summaries)
