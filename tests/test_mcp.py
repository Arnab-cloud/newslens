import json

import pytest
from mock_engine import MockEngine

from newslens.agents.mcp_server import create_mcp_server
from newslens.core.model import NewsLens


@pytest.mark.asyncio
async def test_mcp_tool_registration():
    mock_engine = MockEngine()
    lens = NewsLens(engine=mock_engine)

    mcp_server = create_mcp_server(lens)

    assert "summarize_news" in "".join(
        [tool.model_dump_json() for tool in await mcp_server.list_tools()]
    )

    result = await mcp_server.call_tool(
        "summarize_news", arguments={"text": "Test article"}
    )

    if isinstance(result, dict):
        assert "MOCK_SUMMARY" in json.dumps(result)
    else:
        assert "MOCK_SUMMARY" in "".join(
            [
                json.dumps(res)
                if isinstance(res, dict)
                else "".join([r.model_dump_json() for r in res])
                for res in list(result)
            ]
        )
