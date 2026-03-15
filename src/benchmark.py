"""
Benchmark scenarios and orchestration.

Each scenario has:
* A *dataset* – a list of Python dicts.
* A *task description* used as the system prompt.
* An *expected output schema* used to build the user message.

The orchestrator runs each scenario twice – once with JSON-formatted input and
once with Toon-formatted input – and collects :class:`BenchmarkResult` objects
for comparison.
"""

from __future__ import annotations

import json
import textwrap
import time
from dataclasses import dataclass, field
from typing import Any

import tiktoken

from .llm_client import LLMClient, LLMResponse
from .toon_format import dumps as to_toon

__all__ = [
    "BenchmarkScenario",
    "BenchmarkResult",
    "BenchmarkRun",
    "run_benchmark",
    "BUILTIN_SCENARIOS",
]

# ---------------------------------------------------------------------------
# Token counting helper
# ---------------------------------------------------------------------------


def _count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Count the number of tokens in *text* for *model* using tiktoken."""
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkScenario:
    """A single benchmark scenario definition."""

    name: str
    description: str
    system_prompt: str
    data: list[dict[str, Any]]
    expected_output_hint: str = ""


@dataclass
class BenchmarkResult:
    """Metrics for a single format × scenario combination."""

    scenario: str
    fmt: str                    # "json" or "toon"
    input_text: str
    response: LLMResponse
    local_input_tokens: int     # token count measured locally before the call

    @property
    def total_tokens(self) -> int:
        return self.response.input_tokens + self.response.output_tokens

    @property
    def total_cost_usd(self) -> float:
        return self.response.total_cost_usd


@dataclass
class BenchmarkRun:
    """Container for a pair of results (JSON + Toon) for one scenario."""

    scenario_name: str
    json_result: BenchmarkResult
    toon_result: BenchmarkResult

    # --- derived metrics ---------------------------------------------------

    @property
    def input_token_savings(self) -> int:
        """Positive number = Toon used fewer input tokens."""
        return self.json_result.response.input_tokens - self.toon_result.response.input_tokens

    @property
    def input_token_savings_pct(self) -> float:
        base = self.json_result.response.input_tokens
        if base == 0:
            return 0.0
        return self.input_token_savings / base * 100

    @property
    def cost_savings_usd(self) -> float:
        return self.json_result.total_cost_usd - self.toon_result.total_cost_usd

    @property
    def latency_diff_seconds(self) -> float:
        """Positive = Toon was faster."""
        return (
            self.json_result.response.latency_seconds
            - self.toon_result.response.latency_seconds
        )

    @property
    def local_input_token_savings(self) -> int:
        return (
            self.json_result.local_input_tokens
            - self.toon_result.local_input_tokens
        )

    @property
    def local_input_token_savings_pct(self) -> float:
        base = self.json_result.local_input_tokens
        if base == 0:
            return 0.0
        return self.local_input_token_savings / base * 100


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _format_input(data: list[dict[str, Any]], fmt: str) -> str:
    if fmt == "json":
        return json.dumps(data, indent=2)
    else:
        return to_toon(data)


def run_benchmark(
    scenario: BenchmarkScenario,
    client: LLMClient,
    *,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> BenchmarkRun:
    """Run *scenario* with JSON and Toon inputs and return a :class:`BenchmarkRun`."""

    results = {}
    for fmt in ("json", "toon"):
        input_text = _format_input(scenario.data, fmt)
        user_prompt = textwrap.dedent(f"""
            Format: {fmt.upper()}

            Data:
            {input_text}

            {scenario.expected_output_hint}
        """).strip()

        local_tokens = _count_tokens(
            scenario.system_prompt + user_prompt, model=client.model
        )

        response = client.complete(
            system_prompt=scenario.system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        results[fmt] = BenchmarkResult(
            scenario=scenario.name,
            fmt=fmt,
            input_text=input_text,
            response=response,
            local_input_tokens=local_tokens,
        )

    return BenchmarkRun(
        scenario_name=scenario.name,
        json_result=results["json"],
        toon_result=results["toon"],
    )


# ---------------------------------------------------------------------------
# Built-in scenarios
# ---------------------------------------------------------------------------

BUILTIN_SCENARIOS: list[BenchmarkScenario] = [
    BenchmarkScenario(
        name="User Profile Extraction",
        description="Extract structured data from a list of user profiles.",
        system_prompt=(
            "You are a data extraction assistant. "
            "Given a list of user profiles in the provided format, "
            "return a JSON array where each element contains only the fields: "
            "'name', 'email', and 'country'. "
            "Return ONLY the JSON array, no explanation."
        ),
        data=[
            {
                "id": 1,
                "name": "Alice Johnson",
                "email": "alice@example.com",
                "age": 29,
                "country": "USA",
                "subscription": "premium",
                "last_login": "2024-01-15",
            },
            {
                "id": 2,
                "name": "Bob Smith",
                "email": "bob@example.com",
                "age": 34,
                "country": "UK",
                "subscription": "free",
                "last_login": "2024-01-10",
            },
            {
                "id": 3,
                "name": "Carlos Rivera",
                "email": "carlos@example.com",
                "age": 41,
                "country": "Mexico",
                "subscription": "premium",
                "last_login": "2024-01-20",
            },
        ],
        expected_output_hint=(
            "Return a JSON array with name, email, and country for each user."
        ),
    ),
    BenchmarkScenario(
        name="Product Catalog Summary",
        description="Summarize a product catalog into a short text description.",
        system_prompt=(
            "You are an e-commerce copywriter. "
            "Given a product catalog, write a single concise paragraph "
            "(3–5 sentences) that summarizes the key highlights: "
            "types of products, price range, and notable features. "
            "Do NOT list every product individually."
        ),
        data=[
            {
                "sku": "EL-001",
                "name": "Wireless Headphones",
                "category": "Electronics",
                "price": 79.99,
                "rating": 4.5,
                "in_stock": True,
                "features": ["Bluetooth 5.0", "30h battery", "Noise cancellation"],
            },
            {
                "sku": "EL-002",
                "name": "Smart Watch",
                "category": "Electronics",
                "price": 199.99,
                "rating": 4.7,
                "in_stock": True,
                "features": ["Heart rate monitor", "GPS", "Waterproof"],
            },
            {
                "sku": "HM-001",
                "name": "Yoga Mat",
                "category": "Home & Fitness",
                "price": 29.99,
                "rating": 4.3,
                "in_stock": False,
                "features": ["Non-slip", "Eco-friendly", "6mm thick"],
            },
            {
                "sku": "HM-002",
                "name": "Coffee Maker",
                "category": "Home & Kitchen",
                "price": 49.99,
                "rating": 4.1,
                "in_stock": True,
                "features": ["Programmable", "12-cup capacity", "Auto shut-off"],
            },
        ],
        expected_output_hint=(
            "Write a short paragraph summarizing the catalog."
        ),
    ),
    BenchmarkScenario(
        name="Sales Report Analysis",
        description="Analyze a sales report and answer questions about it.",
        system_prompt=(
            "You are a business analyst. "
            "Given a monthly sales report, answer the following questions "
            "in a structured JSON object with keys: "
            "'top_product', 'total_revenue', 'average_order_value', 'insight'. "
            "Return ONLY the JSON object."
        ),
        data=[
            {"month": "January 2024", "region": "North America", "currency": "USD"},
            {
                "product": "Widget A",
                "units_sold": 1200,
                "revenue": 36000,
                "returns": 45,
            },
            {
                "product": "Widget B",
                "units_sold": 850,
                "revenue": 42500,
                "returns": 20,
            },
            {
                "product": "Gadget C",
                "units_sold": 300,
                "revenue": 15000,
                "returns": 5,
            },
            {
                "product": "Service Plan",
                "units_sold": 500,
                "revenue": 25000,
                "returns": 0,
            },
        ],
        expected_output_hint=(
            "Provide top_product, total_revenue, average_order_value, and a one-sentence insight."
        ),
    ),
    BenchmarkScenario(
        name="Config File Transformation",
        description="Transform a configuration object into environment-variable format.",
        system_prompt=(
            "You are a DevOps engineer. "
            "Given a nested application configuration object, "
            "convert it to a list of KEY=VALUE environment variable pairs. "
            "Use double underscores (__) to denote nesting. "
            "Return ONLY the plain-text KEY=VALUE lines, one per line."
        ),
        data=[
            {
                "app": {
                    "name": "my-service",
                    "version": "2.1.0",
                    "debug": False,
                    "port": 8080,
                },
                "database": {
                    "host": "db.internal",
                    "port": 5432,
                    "name": "production",
                    "pool_size": 10,
                    "ssl": True,
                },
                "cache": {
                    "backend": "redis",
                    "host": "cache.internal",
                    "ttl": 3600,
                },
                "logging": {
                    "level": "INFO",
                    "format": "json",
                },
            }
        ],
        expected_output_hint=(
            "Return flat KEY=VALUE pairs using __ for nesting, e.g. APP__NAME=my-service."
        ),
    ),
]
