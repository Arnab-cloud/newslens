from typing import Any

import requests

from newslens.core.model import EngineProtocol


class RemoteEngine(EngineProtocol):
    def __init__(self, api_url: str, tokenizer: Any):
        self.api_url = api_url
        self._tokenizer = tokenizer

    @property
    def tokenizer(self):
        # We assume the remote server uses the same tokenizer
        # or we return the local one for prompt formatting
        return self._tokenizer

    def generate(self, prompt: str, **kwargs) -> str:
        payload = {"text": prompt, **kwargs}
        response = requests.post(f"{self.api_url}/summarize", json=payload)
        response.raise_for_status()
        return response.json()["summary"]
