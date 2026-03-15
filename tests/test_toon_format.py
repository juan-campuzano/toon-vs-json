"""Unit tests for the Toon format encoder and decoder."""

from __future__ import annotations

import json
import pytest

from src.toon_format import dumps, loads


# ---------------------------------------------------------------------------
# Encoding tests
# ---------------------------------------------------------------------------


class TestEncode:
    def test_null(self):
        assert dumps(None) == "~"

    def test_bool_true(self):
        assert dumps(True) == "T"

    def test_bool_false(self):
        assert dumps(False) == "F"

    def test_integer(self):
        assert dumps(42) == "42"

    def test_negative_integer(self):
        assert dumps(-7) == "-7"

    def test_float(self):
        assert dumps(3.14) == "3.14"

    def test_simple_string(self):
        assert dumps("hello") == "hello"

    def test_string_with_spaces(self):
        # Strings with spaces are quoted so they round-trip correctly
        result = dumps("hello world")
        assert result == "'hello world'"

    def test_string_leading_space(self):
        result = dumps(" hello")
        assert result.startswith("'")

    def test_string_trailing_space(self):
        result = dumps("hello ")
        assert result.startswith("'")

    def test_reserved_true_string(self):
        # The string "T" must be quoted to avoid being decoded as boolean
        result = dumps("T")
        assert result == "'T'"

    def test_reserved_false_string(self):
        result = dumps("F")
        assert result == "'F'"

    def test_reserved_null_string(self):
        result = dumps("~")
        assert result == "'~'"

    def test_empty_string(self):
        assert dumps("") == "''"

    def test_string_with_colon(self):
        result = dumps("key:value")
        assert result.startswith("'")

    def test_string_with_comma(self):
        result = dumps("a,b")
        assert result.startswith("'")

    def test_string_with_single_quote(self):
        result = dumps("it's")
        assert "\\'" in result or result.startswith("'")

    def test_simple_list(self):
        assert dumps([1, 2, 3]) == "[1,2,3]"

    def test_empty_list(self):
        assert dumps([]) == "[]"

    def test_nested_list(self):
        assert dumps([[1, 2], [3, 4]]) == "[[1,2],[3,4]]"

    def test_simple_dict(self):
        result = dumps({"a": 1})
        assert result == "{a:1}"

    def test_empty_dict(self):
        assert dumps({}) == "{}"

    def test_dict_boolean_values(self):
        result = dumps({"active": True, "deleted": False})
        assert result == "{active:T,deleted:F}"

    def test_dict_null_value(self):
        result = dumps({"x": None})
        assert result == "{x:~}"

    def test_nested_dict(self):
        result = dumps({"user": {"name": "Alice", "age": 30}})
        assert result == "{user:{name:Alice,age:30}}"

    def test_dict_with_list_value(self):
        result = dumps({"tags": ["a", "b"]})
        assert result == "{tags:[a,b]}"

    def test_key_with_hyphen_is_quoted(self):
        # Keys with hyphens are not simple word chars
        result = dumps({"my-key": "val"})
        assert "'my-key'" in result

    def test_numeric_string_is_quoted(self):
        # "123" as a string value must not round-trip as number 123
        result = dumps("123")
        assert result.startswith("'")


# ---------------------------------------------------------------------------
# Decoding tests
# ---------------------------------------------------------------------------


class TestDecode:
    def test_null(self):
        assert loads("~") is None

    def test_true(self):
        assert loads("T") is True

    def test_false(self):
        assert loads("F") is False

    def test_integer(self):
        assert loads("42") == 42

    def test_negative_integer(self):
        assert loads("-7") == -7

    def test_float(self):
        assert loads("3.14") == pytest.approx(3.14)

    def test_bare_string(self):
        assert loads("hello") == "hello"

    def test_quoted_string(self):
        assert loads("'hello world'") == "hello world"

    def test_quoted_escape(self):
        assert loads("'it\\'s'") == "it's"

    def test_empty_dict(self):
        assert loads("{}") == {}

    def test_simple_dict(self):
        assert loads("{a:1}") == {"a": 1}

    def test_dict_booleans(self):
        result = loads("{active:T,deleted:F}")
        assert result == {"active": True, "deleted": False}

    def test_dict_null(self):
        assert loads("{x:~}") == {"x": None}

    def test_nested_dict(self):
        result = loads("{user:{name:Alice,age:30}}")
        assert result == {"user": {"name": "Alice", "age": 30}}

    def test_empty_list(self):
        assert loads("[]") == []

    def test_simple_list(self):
        assert loads("[1,2,3]") == [1, 2, 3]

    def test_list_of_strings(self):
        assert loads("[a,b,c]") == ["a", "b", "c"]

    def test_mixed_list(self):
        result = loads("[1,T,~,hello]")
        assert result == [1, True, None, "hello"]

    def test_invalid_trailing_data(self):
        with pytest.raises(ValueError):
            loads("{a:1} extra")

    def test_empty_input(self):
        with pytest.raises(ValueError):
            loads("")


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------


class TestRoundTrip:
    _CASES = [
        None,
        True,
        False,
        0,
        42,
        -3,
        3.14,
        "",
        "hello",
        "T",
        "F",
        "~",
        "hello world",
        [],
        [1, 2, 3],
        {},
        {"a": 1},
        {"active": True, "deleted": False, "score": None},
        {"tags": ["admin", "editor"]},
        {"user": {"name": "Alice", "age": 30, "active": True}},
        [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}],
    ]

    @pytest.mark.parametrize("obj", _CASES)
    def test_round_trip(self, obj):
        assert loads(dumps(obj)) == obj

    def test_complex_nested(self):
        original = {
            "company": "Acme Corp",
            "founded": 1990,
            "public": False,
            "ceo": None,
            "products": [
                {"name": "Widget", "price": 9.99, "available": True},
                {"name": "Gadget", "price": 49.99, "available": False},
            ],
            "address": {
                "street": "123 Main St",
                "city": "Springfield",
                "country": "USA",
            },
        }
        assert loads(dumps(original)) == original


# ---------------------------------------------------------------------------
# Token count comparison (no LLM needed)
# ---------------------------------------------------------------------------


class TestTokenReduction:
    """Verify that Toon encoding is always shorter in character count than
    the equivalent compact JSON (no indentation)."""

    _CASES = [
        {"name": "Alice", "age": 30, "active": True},
        {"tags": ["admin", "editor", "viewer"]},
        [{"id": i, "value": f"item-{i}", "active": True} for i in range(5)],
    ]

    @pytest.mark.parametrize("obj", _CASES)
    def test_toon_shorter_than_json(self, obj):
        toon_len = len(dumps(obj))
        json_len = len(json.dumps(obj))
        assert toon_len < json_len, (
            f"Expected Toon ({toon_len}) < JSON ({json_len})\n"
            f"Toon: {dumps(obj)}\n"
            f"JSON: {json.dumps(obj)}"
        )
