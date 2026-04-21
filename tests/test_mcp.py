import pytest
from mock_engine import MockEngine

from newslens.agents.mcp import create_mcp_server
from newslens.core.model import NewsLens


@pytest.mark.asyncio
async def test_mcp_tool_registration():
    # 1. Setup with injected Mock Engine
    mock_engine = MockEngine()
    lens = NewsLens(engine=mock_engine)

    # 2. Create server
    mcp_server = create_mcp_server(lens)

    # 3. Verify tool exists
    # FastMCP exposes a dictionary of tools
    assert "summarize_news" in await mcp_server.list_tools()

    # 4. Invoke tool manually to verify logic
    result = await mcp_server.call_tool(
        "summarize_news", arguments={"text": "Test article"}
    )
    assert "MOCK_SUMMARY" in result
