import time

from newslens.core.model import EngineProtocol


class MockTockenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool,
        add_generation_prompt: bool,
    ):
        prompt = ""
        for msg in messages:
            for key, val in msg.items():
                prompt += f"{key}:{val}\n"
        return prompt


class MockEngine(EngineProtocol):
    def generate(self, prompt: str, **kwargs) -> str:
        time.sleep(1.0)  # Simulate latency
        return f"MOCK_SUMMARY: {prompt[:20]}..."

    @property
    def tokenizer(self):
        # Here we return a mock tockenizer
        return MockTockenizer()
