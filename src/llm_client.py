"""
LLM client abstraction.

Wraps the OpenAI SDK so the rest of the application is not tightly coupled to
a single provider.  Any OpenAI-compatible endpoint (OpenAI, Azure OpenAI,
Ollama, LM Studio, …) works by setting ``OPENAI_BASE_URL`` in the environment.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

__all__ = ["LLMClient", "LLMResponse"]

# ---------------------------------------------------------------------------
# Per-model token pricing (USD per 1 000 tokens)
# Update these as pricing changes.
# ---------------------------------------------------------------------------
_PRICE_TABLE: dict[str, dict[str, float]] = {
    "gpt-4o":          {"input": 0.005,   "output": 0.015},
    "gpt-4o-mini":     {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo":     {"input": 0.01,    "output": 0.03},
    "gpt-4":           {"input": 0.03,    "output": 0.06},
    "gpt-3.5-turbo":   {"input": 0.0005,  "output": 0.0015},
}
_DEFAULT_PRICE = {"input": 0.001, "output": 0.002}


def _price_for(model: str) -> dict[str, float]:
    for key, price in _PRICE_TABLE.items():
        if model.startswith(key):
            return price
    return _DEFAULT_PRICE


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class LLMResponse:
    """Result returned by a single LLM call."""

    content: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    input_cost_usd: float = field(init=False)
    output_cost_usd: float = field(init=False)
    total_cost_usd: float = field(init=False)

    def __post_init__(self) -> None:
        price = _price_for(self.model)
        self.input_cost_usd = self.input_tokens / 1000 * price["input"]
        self.output_cost_usd = self.output_tokens / 1000 * price["output"]
        self.total_cost_usd = self.input_cost_usd + self.output_cost_usd


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class LLMClient:
    """Thin wrapper around the OpenAI chat completions API."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        self._api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self._base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        kwargs: dict[str, Any] = {"api_key": self._api_key}
        if self._base_url:
            kwargs["base_url"] = self._base_url
        self._client = OpenAI(**kwargs)

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Send a chat completion request and return a structured response."""
        start = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        latency = time.perf_counter() - start

        usage = response.usage
        return LLMResponse(
            content=response.choices[0].message.content or "",
            model=response.model,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            latency_seconds=latency,
        )
