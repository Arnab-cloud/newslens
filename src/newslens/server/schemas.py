from pydantic import BaseModel, Field


class SummarizeRequest(BaseModel):
    article: str = Field(..., description="The news article to summarize")
    max_tokens: int | None = Field(1024)
    temperature: float | None = Field(0.7, ge=0.0, le=2.0)
    top_p: float | None = Field(0.9, ge=0.0, le=1.0)


# V2 adds the URL field from your previous implementation
class SummarizeRequestV2(SummarizeRequest):
    # article: str | None = Field("", description="The news article to summarize")
    url: str | None = Field("", description="The news article url to be summarized")


class SummarizeResponse(BaseModel):
    summary: str
    input_length: int
    output_length: int


class BatchSummarizeRequest(BaseModel):
    articles: list[str]
    max_tokens: int | None = 1024
    temperature: float | None = 0.7
    top_p: float | None = 0.9


class BatchSummarizeResponse(BaseModel):
    summaries: list[str]
    total_articles: int
