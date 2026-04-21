from typing import Annotated

from newslens.core.model import NewsLens


def register_newslens_skill(agent, lens: NewsLens):
    """Registers the NewsLens summarizer as a skill for an AutoGen agent."""

    @agent.register_for_execution()
    @agent.register_for_llm(description="A high-performance news summarization skill.")
    def summarize_article(
        text: Annotated[str, "The news content to be summarized"],
    ) -> str:
        return lens.summarize(text)

    return summarize_article
