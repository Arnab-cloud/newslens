from newslens.agents.mcp import create_mcp_server
from newslens.core.model import NewsLens
from tests.mock_engine import MockEngine

mock_engine = MockEngine()
lens = NewsLens(engine=mock_engine)

mcp_server = create_mcp_server(lens)
mcp_server.run(transport="stdio")
