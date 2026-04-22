from newslens.agents.mcp_server import create_mcp_server
from newslens.core.mock_engine import MockEngine
from newslens.core.model import NewsLens

mock_engine = MockEngine()
lens = NewsLens(engine=mock_engine)

mcp_server = create_mcp_server(lens)
mcp_server.run(transport="stdio")
