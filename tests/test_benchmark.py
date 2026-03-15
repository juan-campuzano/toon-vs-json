"""Unit tests for benchmark helpers (no LLM calls required)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.benchmark import (
    BUILTIN_SCENARIOS,
    BenchmarkResult,
    BenchmarkRun,
    _count_tokens,
    _format_input,
)
from src.llm_client import LLMResponse
from src.toon_format import dumps as to_toon, loads as from_toon


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(input_tokens: int, output_tokens: int, latency: float) -> LLMResponse:
    return LLMResponse(
        content="mock output",
        model="gpt-4o-mini",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_seconds=latency,
    )


def _make_result(
    scenario: str,
    fmt: str,
    input_tokens: int,
    output_tokens: int = 50,
    latency: float = 1.0,
    local_input_tokens: int = 100,
) -> BenchmarkResult:
    return BenchmarkResult(
        scenario=scenario,
        fmt=fmt,
        input_text="",
        response=_make_response(input_tokens, output_tokens, latency),
        local_input_tokens=local_input_tokens,
    )


# ---------------------------------------------------------------------------
# _count_tokens
# ---------------------------------------------------------------------------


class TestCountTokens:
    """Token counting using tiktoken.  The encoding BPE files are fetched from
    the network on first use; in environments without internet access we mock
    the encoder so the logic is still exercised."""

    def _make_mock_encoding(self, token_map: dict[str, list[int]] | None = None):
        """Return a mock tiktoken encoding whose encode() splits on whitespace."""
        enc = MagicMock()
        enc.encode.side_effect = lambda text: text.split() if text else []
        return enc

    def test_returns_positive_int(self):
        with patch("tiktoken.encoding_for_model", return_value=self._make_mock_encoding()):
            count = _count_tokens("Hello, world!")
            assert isinstance(count, int)
            assert count > 0

    def test_longer_text_more_tokens(self):
        with patch("tiktoken.encoding_for_model", return_value=self._make_mock_encoding()):
            short = _count_tokens("Hi")
            long = _count_tokens("Hello world this is a longer string with more tokens.")
            assert long > short

    def test_empty_string(self):
        with patch("tiktoken.encoding_for_model", return_value=self._make_mock_encoding()):
            assert _count_tokens("") == 0

    def test_unknown_model_falls_back(self):
        with patch("tiktoken.encoding_for_model", side_effect=KeyError("unknown")), \
             patch("tiktoken.get_encoding", return_value=self._make_mock_encoding()):
            count = _count_tokens("hello", model="unknown-model-xyz")
            assert count > 0


# ---------------------------------------------------------------------------
# _format_input
# ---------------------------------------------------------------------------


class TestFormatInput:
    _DATA = [{"name": "Alice", "active": True}]

    def test_json_format_is_valid_json(self):
        text = _format_input(self._DATA, "json")
        parsed = json.loads(text)
        assert parsed == self._DATA

    def test_toon_format_round_trips(self):
        text = _format_input(self._DATA, "toon")
        parsed = from_toon(text)
        assert parsed == self._DATA

    def test_toon_shorter_than_json(self):
        data = [
            {"id": i, "name": f"User {i}", "active": True, "role": "admin"}
            for i in range(5)
        ]
        json_text = _format_input(data, "json")
        toon_text = _format_input(data, "toon")
        assert len(toon_text) < len(json_text)


# ---------------------------------------------------------------------------
# BenchmarkResult
# ---------------------------------------------------------------------------


class TestBenchmarkResult:
    def test_total_tokens(self):
        result = _make_result("s", "json", input_tokens=100, output_tokens=50)
        assert result.total_tokens == 150

    def test_total_cost_usd(self):
        result = _make_result("s", "json", input_tokens=1000, output_tokens=1000)
        assert result.total_cost_usd > 0


# ---------------------------------------------------------------------------
# BenchmarkRun
# ---------------------------------------------------------------------------


class TestBenchmarkRun:
    def _make_run(
        self,
        json_input: int = 200,
        toon_input: int = 150,
        json_latency: float = 2.0,
        toon_latency: float = 1.5,
    ) -> BenchmarkRun:
        return BenchmarkRun(
            scenario_name="test",
            json_result=_make_result(
                "test", "json", input_tokens=json_input, latency=json_latency, local_input_tokens=json_input
            ),
            toon_result=_make_result(
                "test", "toon", input_tokens=toon_input, latency=toon_latency, local_input_tokens=toon_input
            ),
        )

    def test_input_token_savings_positive_when_toon_wins(self):
        run = self._make_run(json_input=200, toon_input=150)
        assert run.input_token_savings == 50

    def test_input_token_savings_negative_when_json_wins(self):
        run = self._make_run(json_input=100, toon_input=200)
        assert run.input_token_savings == -100

    def test_input_token_savings_pct(self):
        run = self._make_run(json_input=200, toon_input=150)
        assert run.input_token_savings_pct == pytest.approx(25.0)

    def test_cost_savings_positive_when_toon_wins(self):
        run = self._make_run(json_input=2000, toon_input=1000)
        assert run.cost_savings_usd > 0

    def test_latency_diff_positive_when_toon_faster(self):
        run = self._make_run(json_latency=2.0, toon_latency=1.5)
        assert run.latency_diff_seconds == pytest.approx(0.5)

    def test_local_input_token_savings(self):
        run = self._make_run(json_input=200, toon_input=150)
        assert run.local_input_token_savings == 50

    def test_zero_base_pct(self):
        run = self._make_run(json_input=0, toon_input=0)
        assert run.input_token_savings_pct == 0.0


# ---------------------------------------------------------------------------
# Built-in scenarios
# ---------------------------------------------------------------------------


class TestBuiltinScenarios:
    def test_at_least_one_scenario(self):
        assert len(BUILTIN_SCENARIOS) >= 1

    def test_all_have_names(self):
        for s in BUILTIN_SCENARIOS:
            assert s.name, "Scenario must have a non-empty name"

    def test_all_have_data(self):
        for s in BUILTIN_SCENARIOS:
            assert s.data, f"Scenario '{s.name}' has no data"

    def test_all_data_round_trips(self):
        for s in BUILTIN_SCENARIOS:
            toon_text = to_toon(s.data)
            restored = from_toon(toon_text)
            assert restored == s.data, f"Round-trip failed for scenario '{s.name}'"

    def test_toon_is_shorter_than_json_for_all_scenarios(self):
        for s in BUILTIN_SCENARIOS:
            json_text = json.dumps(s.data)
            toon_text = to_toon(s.data)
            assert len(toon_text) < len(json_text), (
                f"Expected Toon to be shorter than JSON for scenario '{s.name}'\n"
                f"Toon len={len(toon_text)}, JSON len={len(json_text)}"
            )


# ---------------------------------------------------------------------------
# LLMResponse cost calculation
# ---------------------------------------------------------------------------


class TestLLMResponseCost:
    def test_cost_is_non_negative(self):
        r = _make_response(100, 100, 1.0)
        assert r.total_cost_usd >= 0

    def test_more_tokens_more_cost(self):
        cheap = _make_response(100, 100, 1.0)
        expensive = _make_response(10000, 10000, 1.0)
        assert expensive.total_cost_usd > cheap.total_cost_usd

    def test_input_output_cost_sum(self):
        r = _make_response(100, 200, 1.0)
        assert r.total_cost_usd == pytest.approx(
            r.input_cost_usd + r.output_cost_usd
        )
