try:
    from langchain.tools import tool
except ImportError:
    raise ImportError(
        "LangChain is not installed. Please install it with: pip install 'newslens[agents]'"
    )

from pydantic import BaseModel, Field

from newslens.core.model import NewsLens


class SummarizeInput(BaseModel):
    text: str = Field(description="The full text of the news article to summarize.")


def get_langchain_tool(lens: NewsLens):
    @tool("newslens_summarizer", args_schema=SummarizeInput)
    def summarize_tool(text: str) -> str:
        """Use this tool to distill long news articles into concise, factual summaries."""
        return lens.summarize(text)

    return summarize_tool
