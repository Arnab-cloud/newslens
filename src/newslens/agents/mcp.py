try:
    from mcp.server import FastMCP
except ImportError:
    raise ImportError(
        "MCP is not installed. Please install it with: pip install 'newslens[mcp]'"
    )


from newslens.core.engine import InferenceEngine
from newslens.core.model import NewsLens


def create_mcp_server(lens: NewsLens):
    # Initialize the FastMCP server
    mcp = FastMCP("NewsLens Service")

    @mcp.tool()
    async def summarize_news(text: str) -> str:
        """
        Summarizes a news article using the local Qwen-0.6B LoRA model.
        Extremely fast and context-aware.
        """
        # We use the async version to keep the MCP server responsive
        return await lens.asummarize(text)

    return mcp


def serve_mcp(adapter_path: str | None = None):
    """Entry point to launch the MCP server via CLI."""
    engine = InferenceEngine(adapter_path=adapter_path)
    lens = NewsLens(engine)
    server = create_mcp_server(lens)
    server.run(transport="stdio")
