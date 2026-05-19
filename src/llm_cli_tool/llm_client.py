import logging
import os
from typing import Generator

from openai import OpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, model: str):
        self.model = model
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    @retry(
        retry=retry_if_exception_type(RateLimitError),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    def _call(self, messages: list[dict], stream: bool, temperature: float, max_tokens: int):
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=stream,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def stream(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2_000) -> Generator[str, None, None]:
        response = self._call(
            [{"role": "user", "content": prompt}],
            stream=True,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def complete(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2_000) -> tuple[str, int, int]:
        response = self._call(
            [{"role": "user", "content": prompt}],
            stream=False,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        return content, response.usage.prompt_tokens, response.usage.completion_tokens

