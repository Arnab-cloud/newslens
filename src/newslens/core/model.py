import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol


class EngineProtocol(Protocol):
    """Defines the interface that any engine must implement."""

    @property
    def tokenizer(self) -> Any: ...

    def generate(self, prompt: str, **kwargs) -> str: ...


class NewsLens:
    def __init__(self, engine: EngineProtocol, max_workers: int = 2):
        self.engine = engine
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def summarize(self, text: str, **kwargs) -> str:
        """Synchronous summarization call."""
        prompt = self._build_prompt(text)
        return self.engine.generate(prompt, **kwargs)

    async def asummarize(self, text: str, **kwargs) -> str:
        """Asynchronous summarization call for Agents/Web Servers."""
        func = functools.partial(self.summarize, text, **kwargs)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, func)

    def summarize_batch(self, texts: list[str], **kwargs) -> list[str]:
        """Simple loop for batching.
        Note: Future versions could implement true continuous batching here.
        """
        return [self.summarize(t, **kwargs) for t in texts]

    async def asummarize_batch(self, texts: list[str], **kwargs) -> list[str]:
        """
        High-performance batch summarization using asyncio.gather.
        This schedules all tasks concurrently on the executor.
        """
        tasks = [self.asummarize(text, **kwargs) for text in texts]
        return await asyncio.gather(*tasks)

    def _build_prompt(self, text: str) -> str:
        """Standardizes the chat template for Qwen."""
        messages = [
            {"role": "system", "content": "You are a professional news summarizer."},
            {"role": "user", "content": f"Summarize this article:\n\n{text}"},
        ]
        # In 2026, we use the tokenizer's built-in chat template logic
        return self.engine.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
